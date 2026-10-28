"""ExperimentRunner, GridRunner and Exporter (UC3, UC5, UC6, UC9, UC10; FR 17, 26-28).

This is the only module that wires client-side objects to the coordinator, so the
client/coordinator boundary stays explicit everywhere else.
"""
from __future__ import annotations

import copy
import csv
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .client.runtime import ClientRuntime, SummaryBuilder
from .config import dump_resolved, effective_config, run_hash
from .coordinator.coordinator import Coordinator, build_codebook
from .coordinator.participation import build_schedule
from .coordinator.strategies import REGISTRY
from .data.loader import load_dataset_from_config
from .data.partition import partition, write_report
from .evaluation import Evaluator
from .infra.artifacts import ArtifactStore
from .infra.model_host import ModelHost
from .rng import RngService


def _fmt(x: float) -> str:
    return f"{x:g}"


def run_id(cfg: dict, strategy: str) -> str:
    """``<name>-<strategy>-a<alpha>-p<rate>-s<seed>-<hash8>``.

    The hash covers every setting that can change the run's numbers (``config.run_hash``),
    so a changed lr / kappa / tau lands in a new directory and can never pick up an old
    completed summary. Changing only output_dir, checkpoint_every or the grid lists keeps it.
    """
    eff = effective_config(cfg, strategy)
    e = eff["experiment"]
    return (f"{e['name']}-{strategy}-a{_fmt(eff['data']['partition']['alpha'])}-p{_fmt(eff['dropout']['rate'])}"
            f"-s{e['seed']}-{run_hash(eff, strategy)[:8]}")


def reproducibility(cfg: dict, device: str) -> dict:
    """What FR 3 guarantees for this run, stated honestly (see README, Reproducibility)."""
    det_cpu = device == "cpu" and cfg["model"]["deterministic"]
    return {"partitions_and_participation": "bitwise (seeded numpy streams, independent of device)",
            "training": ("bitwise on the same machine and library versions (CPU, deterministic kernels)" if det_cpu
                         else f"statistical only ({device}: non-deterministic accelerator kernels; "
                              "compare runs as seed distributions, not bit-for-bit)"),
            "device": device}


def run_single(cfg: dict, strategy: str, resume: bool = False, fault_hook=None) -> dict:
    """One strategy, one seed. Returns the run summary.

    Any failure before the round loop starts (missing dataset file, out of memory while
    loading the model, ...) also leaves run_summary.json with status "failed" (FR 34, UC8).
    """
    store = ArtifactStore(Path(cfg["experiment"]["output_dir"]) / run_id(cfg, strategy))
    try:
        return _run_single(cfg, strategy, resume, fault_hook)
    except BaseException as e:
        s = store.read_json("run_summary.json")
        if not s or s.get("status") != "failed":          # the coordinator writes its own, richer record
            store.write_json("run_summary.json", {"status": "failed", "strategy": strategy,
                                                  "reason": f"{type(e).__name__}: {e}"})
        raise


def _run_single(cfg: dict, strategy: str, resume: bool, fault_hook) -> dict:
    cfg = effective_config(cfg, strategy)                   # strategy_overrides applied
    h = run_hash(cfg, strategy)
    store = ArtifactStore(Path(cfg["experiment"]["output_dir"]) / run_id(cfg, strategy))
    done = store.read_json("run_summary.json")
    if resume and done and done.get("status") == "completed":
        if done.get("config_hash") == h:                    # reuse only a result of this exact config
            return done
        print(f"[fedtune] {store.dir.name}: stored summary has config_hash {done.get('config_hash')!r}, "
              f"expected {h!r}; re-running from scratch")
        resume = False
    if cfg["model"]["deterministic"]:
        torch.use_deterministic_algorithms(True, warn_only=True)

    rng = RngService(cfg["experiment"]["seed"])
    K, R = cfg["experiment"]["num_clients"], cfg["experiment"]["rounds"]
    data = load_dataset_from_config(cfg["data"], seed=rng.int_seed("data"))
    part = partition(data, K, cfg["data"]["partition"], cfg["data"], rng.fresh("partition"))   # deterministic: rebuilt on resume

    ckpt = None
    if resume:
        from .infra.artifacts import CheckpointManager
        ckpt = CheckpointManager(store).latest()
        if ckpt is not None and ckpt.get("config_hash") != h:
            print(f"[fedtune] {store.dir.name}: checkpoint was written by another config; starting fresh")
            ckpt = None
    if ckpt is None:
        for f in ("rounds.jsonl", "summary_log.jsonl", "run_summary.json"):
            (store.dir / f).unlink(missing_ok=True)
        for old in (store.dir / "checkpoints").glob("ckpt_*.pt"):
            old.unlink()
        dump_resolved(cfg, store.dir / "config.resolved.yaml")
        write_report(part, store.dir / "partition_report.json")
        P = build_schedule(K, R, cfg["dropout"]["rate"], rng.fresh("participation"))   # independent of the strategy
        store.write_participation(P)
    else:
        P = store.read_participation()                      # read, never regenerated, on resume

    host = ModelHost(cfg["model"], data.num_classes, init_seed=rng.int_seed("init"), max_length=cfg["data"]["max_length"])
    init_state = host.export_adapter()                      # PEFT init: B = 0, the adapter does nothing yet
    dv = cfg["aggregation"]["diversity"]
    hidden = host.model.base_model.model.config.hidden_size
    projection = torch.from_numpy(rng.fresh("projection").standard_normal((hidden, dv["embed_dim"]))).float() / np.sqrt(dv["embed_dim"])

    strat = REGISTRY[strategy](cfg["aggregation"])
    codebook = None
    if strat.needs_summary and ckpt is None:
        emb = host.embed(part.reference.texts, projection)
        codebook = build_codebook(emb, dv["codebook_size"], rng.int_seed("codebook"))

    sb = SummaryBuilder(projection)
    clients = {k: ClientRuntime(k, part.train[k], part.val[k], host, sb, data.num_classes) for k in range(K)}
    if fault_hook:
        fault_hook(clients)
    coord = Coordinator(cfg, strat, clients, P, init_state, Evaluator(host, part.global_val, data.num_classes),
                        store, rng, data.num_classes, host.full_model_bytes, codebook, h,
                        evaluate_test=Evaluator(host, part.test, data.num_classes))
    if ckpt is not None:
        coord.restore(ckpt)
    summary = coord.run()
    summary.update({"run_id": store.dir.name, "H": part.H, "alpha": cfg["data"]["partition"]["alpha"],
                    "dropout_rate": cfg["dropout"]["rate"], "seed": cfg["experiment"]["seed"],
                    "reproducibility": reproducibility(cfg, host.device.type)})
    store.write_json("run_summary.json", summary)
    return summary


def compare(cfg: dict, resume: bool = False) -> list[dict]:
    """UC5: every listed strategy on the same partition, seed and participation sequence."""
    rows = [run_single(cfg, s, resume) for s in cfg["experiment"]["strategy"]]
    out = Path(cfg["experiment"]["output_dir"]) / f"{cfg['experiment']['name']}-comparison.csv"
    _write_csv(out, [_flat(r) for r in rows])
    return rows


def _flat(s: dict) -> dict:
    g = s.get("final_test_metrics") or {}
    v = s.get("final_val_metrics") or {}
    d = s.get("client_dispersion") or {}
    return {"run_id": s.get("run_id"), "config_hash": s.get("config_hash"), "strategy": s["strategy"],
            "alpha": s.get("alpha"), "dropout_rate": s.get("dropout_rate"), "seed": s.get("seed"), "H": s.get("H"),
            "val_accuracy": v.get("accuracy"), "accuracy": g.get("accuracy"), "macro_f1": g.get("macro_f1"),
            "worst_client_accuracy": d.get("worst_client_accuracy"), "client_variance": d.get("variance"),
            "bytes_up": s["cumulative_bytes"]["up"], "bytes_down": s["cumulative_bytes"]["down"],
            "rounds_skipped": s["rounds_skipped"], "status": s["status"]}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")                     # e.g. paired_vs_fedavg.csv when only FedAvg ran
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- GridRunner (FR 26, UC6)
def grid_cells(cfg: dict) -> list[tuple]:
    g = cfg["grid"]
    alphas = g["alpha"] or [cfg["data"]["partition"]["alpha"]]
    rates = g["dropout_rate"] or [cfg["dropout"]["rate"]]
    seeds = g["seeds"] or [cfg["experiment"]["seed"]]
    # seed first: a partly finished grid already holds complete comparisons for the first seeds
    return list(itertools.product(seeds, alphas, rates, cfg["experiment"]["strategy"]))


def run_grid(cfg: dict, shard: str | None = None) -> Path:
    cells = grid_cells(cfg)
    if shard:                                        # "i/n": run cells i, i+n, ... on this machine
        i, n = map(int, shard.split("/"))
        cells = cells[i::n]
    for seed, alpha, rate, strat in cells:
        c = copy.deepcopy(cfg)
        c["experiment"]["seed"], c["data"]["partition"]["alpha"], c["dropout"]["rate"] = seed, float(alpha), float(rate)
        try:
            run_single(c, strat, resume=True)        # skips completed cells, resumes checkpointed ones
        except Exception as e:                       # one failed cell must not stop the grid
            print(f"[grid] cell seed={seed} alpha={alpha} p={rate} {strat} failed: {e}")
    return summarise_grid(cfg)


def summarise_grid(cfg: dict) -> Path:
    out_dir = Path(cfg["experiment"]["output_dir"])
    rows, missing = [], []
    for seed, alpha, rate, strat in grid_cells(cfg):
        c = copy.deepcopy(cfg)
        c["experiment"]["seed"], c["data"]["partition"]["alpha"], c["dropout"]["rate"] = seed, float(alpha), float(rate)
        s = ArtifactStore(out_dir / run_id(c, strat)).read_json("run_summary.json")
        if s and s["status"] == "completed":
            rows.append(_flat(s))
        else:
            missing.append({"strategy": strat, "alpha": alpha, "dropout_rate": rate, "seed": seed})
    keys = ("accuracy", "macro_f1", "worst_client_accuracy", "bytes_up")
    cells = {}
    for r in rows:
        cells.setdefault((r["strategy"], r["alpha"], r["dropout_rate"]), []).append(r)
    summary = []
    for (strat, a, p), rs in sorted(cells.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = {"strategy": strat, "alpha": a, "dropout_rate": p, "n_seeds": len(rs),
               "H_mean": float(np.mean([r["H"] for r in rs]))}
        for k in keys:
            v = [r[k] for r in rs if r[k] is not None]
            row[f"{k}_mean"] = float(np.mean(v)) if v else None
            row[f"{k}_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        summary.append(row)
    gdir = out_dir.parent / "grids" / cfg["experiment"]["name"]       # grids/<grid_id>/ next to runs/
    _write_csv(gdir / "grid_summary.csv", summary)
    _write_csv(gdir / "paired_vs_fedavg.csv", paired_vs_baseline(rows))
    with open(gdir / "missing_cells.json", "w") as f:
        json.dump(missing, f, indent=2)
    return gdir


PAIRED_METRICS = ("accuracy", "macro_f1", "worst_client_accuracy")


def paired_stats(diffs: list[float]) -> dict:
    """Paired t statistics on per-seed differences (strategy - baseline, same seed = same partition
    and participation). With n = 3 seeds the 95% CI uses t_{0.975,2} = 4.30, so only large or very
    consistent gaps register; that is the honest answer, not a defect."""
    from scipy import stats
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    out = {"n_pairs": n, "mean_diff": float(d.mean()) if n else None, "sd_diff": None, "ci95_low": None,
           "ci95_high": None, "t": None, "p_value": None, "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
    if n >= 2:
        sd = float(d.std(ddof=1))
        se = sd / np.sqrt(n)
        half = float(stats.t.ppf(0.975, n - 1)) * se
        out.update(sd_diff=sd, ci95_low=out["mean_diff"] - half, ci95_high=out["mean_diff"] + half)
        if se > 0:
            t = out["mean_diff"] / se
            out.update(t=float(t), p_value=float(2 * stats.t.sf(abs(t), n - 1)))
        elif out["mean_diff"] != 0:
            out.update(t=float("inf") * np.sign(out["mean_diff"]), p_value=0.0)
    lo, hi = out["ci95_low"], out["ci95_high"]
    out["verdict"] = ("insufficient pairs" if n < 2 else "identical on every seed" if not d.any()
                      else "better" if lo > 0 else "worse" if hi < 0 else "inconclusive (CI spans 0)")
    return out


def paired_vs_baseline(rows: list[dict], baseline: str = "fedavg") -> list[dict]:
    """Per cell (alpha, p) and pooled over all cells: strategy - baseline on matched seeds."""
    base = {(r["alpha"], r["dropout_rate"], r["seed"]): r for r in rows if r["strategy"] == baseline}
    out = []
    strategies = sorted({r["strategy"] for r in rows} - {baseline})
    cells = sorted({(r["alpha"], r["dropout_rate"]) for r in rows}, key=lambda c: (float(c[0]), float(c[1])))
    for strat in strategies:
        mine = [r for r in rows if r["strategy"] == strat]
        for metric in PAIRED_METRICS:
            pooled = []
            for a, p in cells:
                diffs = [r[metric] - base[(a, p, r["seed"])][metric] for r in mine
                         if r["alpha"] == a and r["dropout_rate"] == p and (a, p, r["seed"]) in base
                         and r[metric] is not None and base[(a, p, r["seed"])][metric] is not None]
                if not diffs:
                    continue
                pooled += diffs
                out.append({"strategy": strat, "baseline": baseline, "metric": metric, "alpha": a,
                            "dropout_rate": p, **paired_stats(diffs)})
            if pooled:
                out.append({"strategy": strat, "baseline": baseline, "metric": metric, "alpha": "all",
                            "dropout_rate": "all", **paired_stats(pooled)})
    return out


# --------------------------------------------------------------------------- Exporter (FR 28, UC10)
def export_run(run_dir: Path, out: Path) -> Path:
    store = ArtifactStore(run_dir)
    out = out / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    rounds = store.read_rounds()
    flat = [{"round": r["round"], "skipped": r["skipped"], "n_participants": len(r["participating_clients"]),
             "val_accuracy": (r.get("val_metrics") or {}).get("accuracy"),
             "val_macro_f1": (r.get("val_metrics") or {}).get("macro_f1"),
             "test_accuracy": (r.get("test_metrics") or {}).get("accuracy"),
             "bytes_up": r["comm_bytes"]["up"], "bytes_down": r["comm_bytes"]["down"],
             "merge_residual": r.get("merge_residual"), "round_time_s": r["round_time_s"]} for r in rounds]
    _write_csv(out / "rounds.csv", flat)
    (out / "rounds.json").write_text(json.dumps(rounds, indent=1))
    for f in ("config.resolved.yaml", "partition_report.json", "run_summary.json", "participation.npy"):
        if (run_dir / f).exists():
            shutil.copy(run_dir / f, out / f)   # the exported config alone regenerates partition + schedule
    _plots(rounds, store, out)
    return out


def _plots(rounds, store, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    xs = [r["round"] for r in rounds]
    ax[0].plot(xs, [(r.get("val_metrics") or {}).get("accuracy") for r in rounds], "o-", label="accuracy")
    ax[0].plot(xs, [(r.get("val_metrics") or {}).get("macro_f1") for r in rounds], "s--", label="macro-F1")
    ft = (store.read_json("run_summary.json") or {}).get("final_test_metrics")
    if ft and xs:
        ax[0].plot([xs[-1]], [ft["accuracy"]], "k*", ms=12, label=f"final test acc {ft['accuracy']:.3f}")
    ax[0].set_xlabel("round"); ax[0].legend(); ax[0].set_title("Global validation metrics (V_g)")
    P = store.read_participation()
    ax[1].imshow(P.T, aspect="auto", cmap="Greys", interpolation="nearest",
                 extent=[0.5, P.shape[0] + 0.5, P.shape[1] - 0.5, -0.5])   # rounds are 1-based
    ax[1].set_xlabel("round"); ax[1].set_ylabel("client"); ax[1].set_title("Participation (black = present)")
    fig.tight_layout()
    fig.savefig(out / "plots.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- TuningRunner (Design 2.3.4)
def tuning_trials(cfg: dict, strategy: str) -> list[dict]:
    """The trial settings for one strategy: the full product of its search space if it has at most
    ``tune.trials`` points, else a seeded sample of exactly ``tune.trials`` of them (same budget for
    every strategy). An empty space gives one trial at the base settings."""
    from .rng import RngService
    t = cfg["tune"]
    space = t["space"].get(strategy, {})
    if not space:
        return [{}]
    keys = sorted(space)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(space[k] for k in keys))]
    if len(combos) <= t["trials"]:
        return combos
    g = RngService(t["seed"]).fresh(f"tune/{strategy}")
    pick = sorted(g.choice(len(combos), t["trials"], replace=False).tolist())
    return [combos[i] for i in pick]


def _tune_cell(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    t = c["tune"]
    c["experiment"]["seed"] = t["seed"]
    if t["alpha"] is not None:
        c["data"]["partition"]["alpha"] = float(t["alpha"])
    if t["dropout_rate"] is not None:
        c["dropout"]["rate"] = float(t["dropout_rate"])
    c["evaluation"]["final_test"] = False            # the test set is never evaluated while tuning
    c["evaluation"]["test_every_round"] = False
    return c


def trial_score(rounds: list[dict], metric: str, last: int) -> float | None:
    """Mean global-validation metric over the last ``last`` completed (non-skipped) rounds."""
    v = [r["val_metrics"][metric] for r in rounds if not r["skipped"] and r.get("val_metrics")
         and r["val_metrics"].get(metric) is not None and np.isfinite(r["val_metrics"][metric])]
    return float(np.mean(v[-last:])) if v else None


def run_tuning(cfg: dict) -> Path:
    """Fair tuning protocol: same trial budget per strategy, on the tuning seed, selected on the
    global validation set V_g. Writes ``tunes/<name>/{tuning_results.csv, best_params.json, tuned.yaml}``;
    ``tuned.yaml`` is the input config with the winners frozen in ``strategy_overrides``, ready
    for ``fedtune grid`` / ``fedtune run``."""
    import yaml
    if cfg["data"]["global_val_frac"] <= 0:
        raise ValueError("tuning needs a global validation set: set data.global_val_frac > 0")
    t = cfg["tune"]
    base = _tune_cell(cfg)
    results, best = [], {}
    for strat in cfg["experiment"]["strategy"]:
        for i, params in enumerate(tuning_trials(cfg, strat)):
            c = copy.deepcopy(base)
            c["strategy_overrides"] = {strat: {**cfg["strategy_overrides"].get(strat, {}), **params}}
            rid = run_id(c, strat)
            print(f"[tune] {strat} trial {i + 1}: {params or '(base settings)'}  -> {rid}")
            status, score = "failed", None
            try:
                s = run_single(c, strat, resume=True)
                status = s["status"]
                score = trial_score(ArtifactStore(Path(c["experiment"]["output_dir"]) / rid).read_rounds(),
                                    t["select_metric"], t["select_last"])
            except Exception as e:                   # a failed trial must not stop tuning
                print(f"[tune] {strat} trial {i + 1} failed: {e}")
            results.append({"strategy": strat, "trial": i + 1, "params": json.dumps(params, sort_keys=True),
                            f"val_{t['select_metric']}_last{t['select_last']}": score, "status": status, "run_id": rid})
            if score is not None and (strat not in best or score > best[strat]["score"]):
                best[strat] = {"params": params, "score": score, "trial": i + 1, "run_id": rid}
    out = Path(cfg["experiment"]["output_dir"]).parent / "tunes" / cfg["experiment"]["name"]
    _write_csv(out / "tuning_results.csv", results)
    (out / "best_params.json").write_text(json.dumps(
        {"tuning_seed": t["seed"], "select": f"mean val {t['select_metric']} over last {t['select_last']} rounds",
         "best": best}, indent=2))
    tuned = copy.deepcopy(cfg)
    tuned["strategy_overrides"] = {s: {**cfg["strategy_overrides"].get(s, {}), **b["params"]} for s, b in best.items()}
    tuned["evaluation"]["final_test"] = True
    tuned["experiment"]["name"] = f"{cfg['experiment']['name']}-final"
    tuned["tune"]["space"] = {}                     # frozen: this config is for the final runs
    header = (f"# Frozen by `fedtune tune` (tuning seed {t['seed']}, selected on global validation, "
              f"never on test).\n# Run the final grid with:  fedtune grid -c {out / 'tuned.yaml'}\n")
    (out / "tuned.yaml").write_text(header + yaml.safe_dump(tuned, sort_keys=False))
    missing = [s for s in cfg["experiment"]["strategy"] if s not in best]
    if missing:
        print(f"[tune] WARNING: no successful trial for {missing}; they keep their base settings in tuned.yaml")
    return out
