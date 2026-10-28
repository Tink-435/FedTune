"""Coordinator round loop (Alg. 10, FR 19-24, 27, 33-34), CommAccountant and codebook.

The coordinator package imports nothing from ``fedtune.client`` or ``fedtune.data``
(checked by a static import test). Clients reach it only as endpoints that turn a
``GlobalBroadcast`` into a ``ClientUpdate``; evaluation is injected as a callable.
"""
from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import torch

from ..infra.artifacts import ArtifactStore, CheckpointManager
from ..protocol import ClientUpdate, GlobalBroadcast, adapter_bytes, broadcast_bytes, summary_bytes, update_bytes
from ..rng import RngService
from . import linalg
from .participation import ParticipationScheduler
from .strategies import RoundContext, Strategy


class ClientEndpoint(Protocol):
    def train(self, b: GlobalBroadcast) -> ClientUpdate: ...
    def validate(self, adapter, head) -> dict: ...


def build_codebook(emb: torch.Tensor, Q: int, seed: int, iters: int = 50) -> torch.Tensor:
    """k-means over embeddings of the coordinator's own reference set (disjoint from shards/test)."""
    g = np.random.default_rng(seed)
    X = emb.double()
    Q = min(Q, len(X))
    C = X[torch.as_tensor(g.choice(len(X), Q, replace=False))].clone()
    for _ in range(iters):
        a = torch.cdist(X, C).argmin(1)
        newC = torch.stack([X[a == q].mean(0) if (a == q).any() else C[q] for q in range(Q)])
        if torch.allclose(newC, C):
            break
        C = newC
    return C.float()


@dataclass
class CommAccountant:
    up: int = 0
    down: int = 0
    per_round: list = field(default_factory=list)

    def add(self, up: int, down: int) -> None:
        self.up += up
        self.down += down
        self.per_round.append((up, down))


def _peak_mem_mb() -> float:
    """Peak resident memory in MB; works on Linux, macOS and Windows (NFR 8)."""
    try:
        import resource                                   # Linux / macOS
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    except ImportError:                                   # Windows
        try:
            import psutil
            mi = psutil.Process().memory_info()
            return getattr(mi, "peak_wset", mi.rss) / (1024 * 1024)
        except ImportError:
            return float("nan")


class Coordinator:
    def __init__(self, cfg: dict, strategy: Strategy, clients: dict[int, ClientEndpoint],
                 schedule: np.ndarray, init_state: tuple, evaluate_val: Callable[[dict, dict], dict],
                 store: ArtifactStore, rng: RngService, num_classes: int, full_model_bytes: int,
                 codebook: torch.Tensor | None = None, config_hash: str = "",
                 evaluate_test: Callable[[dict, dict], dict] | None = None):
        self.cfg, self.strategy, self.clients = cfg, strategy, clients
        self.sched = ParticipationScheduler(schedule)
        self.global_adapter, self.global_head = init_state
        # V_g drives every per-round decision/plot; T is only read when evaluate_test is called
        self.evaluate_val, self.evaluate_test = evaluate_val, evaluate_test
        self.store, self.rng = store, rng
        self.ckpt = CheckpointManager(store)
        self.num_classes, self.full_model_bytes = num_classes, full_model_bytes
        self.codebook, self.config_hash = codebook, config_hash
        self.comm = CommAccountant()
        self.codebook_sent: set[int] = set()
        self.stale: dict[int, tuple[int, ClientUpdate]] = {}
        self.r0 = 0
        self.stats = {"completed": 0, "skipped": 0, "participation": [], "excluded": 0}
        self.last_metrics, self.last_client_metrics = {}, {}
        self.final_test_metrics = None

    # ------------------------------------------------------------------ checkpoints
    def _state(self, r: int) -> dict:
        return {"round": r, "global_adapter": self.global_adapter, "global_head": self.global_head,
                "rng": self.rng.state(), "strategy": self.strategy.get_state(),
                "rounds_offset": self.store.rounds_offset(), "config_hash": self.config_hash,
                "codebook_sent": sorted(self.codebook_sent), "codebook": self.codebook,
                "stale": {k: (age, u) for k, (age, u) in self.stale.items()},
                "comm": (self.comm.up, self.comm.down), "stats": self.stats,
                "last_metrics": self.last_metrics, "last_client_metrics": self.last_client_metrics}

    def restore(self, ck: dict) -> None:
        if ck["config_hash"] != self.config_hash:
            raise RuntimeError("checkpoint was written by a different configuration (config_hash mismatch)")
        self.r0 = ck["round"]
        self.global_adapter, self.global_head = ck["global_adapter"], ck["global_head"]
        self.rng.restore(ck["rng"])
        self.strategy.set_state(ck["strategy"])
        # client runtimes are rebuilt on resume, so the codebook is re-sent once to each client
        self.codebook_sent, self.codebook = set(), ck["codebook"]
        self.stale = dict(ck["stale"])
        self.comm.up, self.comm.down = ck["comm"]
        self.stats = ck["stats"]
        self.last_metrics, self.last_client_metrics = ck["last_metrics"], ck["last_client_metrics"]
        self.store.truncate_rounds(ck["rounds_offset"])

    # ------------------------------------------------------------------ Algorithm 10
    def run(self) -> dict:
        t_start = time.perf_counter()
        try:
            for r in range(self.r0 + 1, self.cfg["experiment"]["rounds"] + 1):
                self._round(r)
            if self.cfg["evaluation"]["final_test"] and self.evaluate_test is not None:
                self.final_test_metrics = self.evaluate_test(self.global_adapter, self.global_head)
            summary = self._summary("completed", time.perf_counter() - t_start)
            self.store.write_json("run_summary.json", summary)
            return summary
        except BaseException as e:            # FR 34: flush, mark failed with reason, re-raise
            self.store.write_json("run_summary.json", {
                **self._summary("failed", time.perf_counter() - t_start),
                "reason": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=5)})
            raise

    def _broadcast(self, r: int, k: int) -> GlobalBroadcast:
        tr = self.cfg["train"]
        send_cb = self.strategy.needs_summary and k not in self.codebook_sent
        if send_cb:
            self.codebook_sent.add(k)
        return GlobalBroadcast(
            round=r, adapter=self.global_adapter, head=self.global_head,
            hparams={"lr": tr["lr"], "batch_size": tr["batch_size"], "local_epochs": tr["local_epochs"],
                     "max_steps": tr["max_steps"], "grad_clip": tr["grad_clip"], "weight_decay": tr["weight_decay"]},
            client_seed=self.rng.int_seed(f"client/{k}/{r}"),
            codebook=self.codebook if send_cb else None, needs_summary=self.strategy.needs_summary)

    def _round(self, r: int) -> None:
        t0 = time.perf_counter()
        dc, ec = self.cfg["dropout"], self.cfg["experiment"]
        m_min = dc["min_participants"]
        S_r, dropped = self.sched.participants(r), self.sched.dropped(r)
        self.stats["participation"].append(len(S_r))
        rec = {"run_id": self.store.dir.name, "strategy": self.strategy.name, "round": r,
               "participating_clients": S_r, "dropped_clients": dropped, "excluded_clients": [],
               "stale_clients": [], "skipped": False}

        if len(S_r) < m_min:
            return self._skip(rec, "too_few_participants", t0)

        valid, up, down = [], 0, 0
        for k in S_r:                                       # sequential, increasing id (one accelerator)
            b = self._broadcast(r, k)
            down += broadcast_bytes(b)
            u = self.clients[k].train(b)
            up += update_bytes(u)
            if u.is_finite():                               # FR 33
                valid.append((u, float(u.n_examples)))
                self.stale[k] = (r, u)
            else:
                rec["excluded_clients"].append({"client": k, "reason": "non-finite"})
                self.stats["excluded"] += 1
        if dc["stale_updates"]:                             # FR 22 (optional)
            for k in dropped:
                if k in self.stale and 1 <= r - self.stale[k][0] <= dc["stale_max_age"]:
                    age = r - self.stale[k][0]
                    u = self.stale[k][1]
                    valid.append((u, u.n_examples * dc["stale_discount"] ** age))
                    rec["stale_clients"].append({"client": k, "age": age})
        rec["comm_bytes"] = {"up": up, "down": down}
        if len(valid) < m_min:
            self.comm.add(up, down)
            return self._skip(rec, "too_few_valid_updates", t0)

        ups = [u for u, _ in valid]
        ids = [u.client_id for u in ups]
        n = np.array([nv for _, nv in valid])
        sim_cache = {}

        def similarity():
            if "S" not in sim_cache:
                sim_cache["S"] = linalg.update_similarity([u.adapter for u in ups], self.global_adapter)
            return sim_cache["S"]

        ctx = RoundContext(round=r, client_ids=ids, n=n,
                           summaries_c=[u.summary_c for u in ups] if self.strategy.needs_summary else None,
                           summaries_y=[u.summary_y for u in ups] if self.strategy.needs_summary else None,
                           similarity=similarity, num_classes=self.num_classes,
                           codebook_size=len(self.codebook) if self.codebook is not None else 0)
        self.strategy.round_info = {}
        w, clusters = self.strategy.compute_weights(ctx)
        self.global_adapter, self.global_head, eps = linalg.merge(
            [u.adapter for u in ups], [u.head for u in ups], w, self.cfg["aggregation"]["lora_method"])

        if self.strategy.needs_summary:
            for u in ups:
                if u.round == r:
                    self.store.append_summary({"round": r, "client": u.client_id, "bytes": summary_bytes(u),
                                               "fields": {"h_c": list(u.summary_c), "h_y": list(u.summary_y)},
                                               "reveals": "coarse topical (codebook-cell) and label make-up of the shard"})

        self.comm.add(up, down)
        rec.update({
            "aggregation_weights": {str(i): round(float(x), 6) for i, x in zip(ids, w)},
            "clusters": clusters, "merge_residual": eps,
            "local_loss": {str(u.client_id): u.local_loss for u in ups if u.round == r},
            "upload_fraction_of_model": (up / max(len(S_r), 1)) / self.full_model_bytes,
            "cumulative_bytes": {"up": self.comm.up, "down": self.comm.down},
        })
        if self.strategy.round_info:
            rec["strategy_info"] = self.strategy.round_info
        rec["val_metrics"] = self.last_metrics = self.evaluate_val(self.global_adapter, self.global_head)
        if self.cfg["evaluation"]["test_every_round"] and self.evaluate_test is not None:
            rec["test_metrics"] = self.evaluate_test(self.global_adapter, self.global_head)
        if r % ec["eval_every"] == 0 or r == ec["rounds"]:
            self.last_client_metrics = {str(k): c.validate(self.global_adapter, self.global_head)
                                        for k, c in self.clients.items()}
            rec["client_metrics"] = self.last_client_metrics
        rec["round_time_s"] = round(time.perf_counter() - t0, 3)
        self.stats["completed"] += 1
        self.store.append_round(rec)
        if r % ec["checkpoint_every"] == 0 or r == ec["rounds"]:
            self.ckpt.save(r, self._state(r))

    def _skip(self, rec: dict, reason: str, t0: float) -> None:
        rec.update({"skipped": True, "skip_reason": reason, "val_metrics": self.last_metrics,
                    "aggregation_weights": {}, "clusters": None, "merge_residual": None,
                    "cumulative_bytes": {"up": self.comm.up, "down": self.comm.down},
                    "round_time_s": round(time.perf_counter() - t0, 3)})
        rec.setdefault("comm_bytes", {"up": 0, "down": 0})
        self.stats["skipped"] += 1
        self.store.append_round(rec)          # global adapter unchanged (FR 21)

    def _summary(self, status: str, seconds: float) -> dict:
        accs = [m["accuracy"] for m in self.last_client_metrics.values() if m and np.isfinite(m["accuracy"])]
        part = self.stats["participation"]
        return {
            "status": status, "strategy": self.strategy.name, "config_hash": self.config_hash,
            "rounds_completed": self.stats["completed"], "rounds_skipped": self.stats["skipped"],
            "excluded_updates": self.stats["excluded"],
            "participation": {"mean": float(np.mean(part)) if part else 0.0,
                              "min": int(min(part)) if part else 0, "max": int(max(part)) if part else 0},
            "final_val_metrics": self.last_metrics,              # global validation V_g
            "final_test_metrics": self.final_test_metrics,       # test T (None if final_test is off)
            "client_dispersion": {"variance": float(np.var(accs)) if accs else None,
                                  "worst_client_accuracy": float(min(accs)) if accs else None},
            "cumulative_bytes": {"up": self.comm.up, "down": self.comm.down},
            "adapter_bytes": adapter_bytes(self.global_adapter, self.global_head),
            "full_model_bytes": self.full_model_bytes,
            "total_time_s": round(seconds, 2), "peak_memory_mb": round(_peak_mem_mb(), 1),
        }
