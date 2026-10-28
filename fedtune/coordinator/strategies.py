"""Aggregation strategies (R3, FR 13-15a, 18, Alg. 5-8) and the StrategyRegistry.

A strategy only returns per-client weights (and optional cluster labels); the
separate LoRA merger builds the adapter, so every strategy works with both merge
methods (FR 16). Add a new strategy without touching core code (UC11)::

    from fedtune.coordinator.strategies import Strategy, register

    @register("uniform")
    class Uniform(Strategy):
        def compute_weights(self, ctx):
            m = len(ctx.n)
            return np.full(m, 1.0 / m), None
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform

REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        REGISTRY[name] = cls
        return cls
    return deco


@dataclass
class RoundContext:
    """What a strategy may see: updates' volumes, summaries and factors - never data."""
    round: int
    client_ids: list[int]
    n: np.ndarray                          # effective volumes (stale updates discounted)
    summaries_c: list | None               # h_k^c per client (diversity strategies)
    summaries_y: list | None               # h_k^y
    similarity: Callable[[], np.ndarray]   # lazy Alg. 4
    num_classes: int
    codebook_size: int


class Strategy:
    name = "base"
    needs_summary = False
    # aggregation.* blocks this strategy reads; they (plus lora_method) enter the run hash.
    # None (the default for plug-ins) = every block, which is always safe.
    config_sections: tuple | None = None

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.state: dict = {}
        self.round_info: dict = {}          # per-round diagnostics, copied into the round record

    def compute_weights(self, ctx: RoundContext) -> tuple[np.ndarray, list[list[int]] | None]:
        raise NotImplementedError

    # strategy state is checkpointed (FR 27)
    def get_state(self) -> dict:
        return self.state

    def set_state(self, s: dict) -> None:
        self.state = s or {}


# --------------------------------------------------------------------------- Algorithm 5
def cap(w: np.ndarray, w_max: float) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    w_max = max(w_max, 1.0 / len(w))
    capped = np.zeros(len(w), dtype=bool)
    while True:
        over = (w > w_max + 1e-12) & ~capped
        if not over.any():
            return w
        capped |= over
        w[capped] = w_max
        free = ~capped
        if not free.any():
            return w / w.sum()
        w[free] = w[free] / w[free].sum() * (1.0 - w_max * capped.sum())


# --------------------------------------------------------------------------- FedAvg (FR 13)
@register("fedavg")
class FedAvg(Strategy):
    config_sections = ()

    def compute_weights(self, ctx):
        return ctx.n / ctx.n.sum(), None


# --------------------------------------------------------------------------- Algorithm 6 (FR 14)
def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def coverage(n, hc, hy, kappa, Q, C) -> float:
    n = np.asarray(n, dtype=float)
    hbar_c = (n[:, None] * np.asarray(hc)).sum(0) / n.sum()
    hbar_y = (n[:, None] * np.asarray(hy)).sum(0) / n.sum()
    return kappa * _entropy(hbar_c / hbar_c.sum()) / np.log(Q) + (1 - kappa) * _entropy(hbar_y / hbar_y.sum()) / np.log(C)


def diversity_weights(n, hc, hy, Q, C, lam, kappa, eps, w_max):
    n = np.asarray(n, dtype=float)
    m = len(n)
    if m == 1:
        return np.ones(1), np.ones(1)
    f_all = coverage(n, hc, hy, kappa, Q, C)
    s = np.empty(m)
    for k in range(m):                          # leave-one-out coverage gain
        keep = np.arange(m) != k
        s[k] = max(f_all - coverage(n[keep], np.asarray(hc)[keep], np.asarray(hy)[keep], kappa, Q, C), 0.0) + eps
    w = (1 - lam) * n / n.sum() + lam * s / s.sum()
    return (cap(w, w_max) if lam > 0 else w), s   # lam = 0 reduces exactly to FedAvg (built-in ablation)


@register("diversity")
class DiversityAware(Strategy):
    needs_summary = True
    config_sections = ("diversity",)

    def compute_weights(self, ctx):
        d = self.cfg["diversity"]
        w, s = diversity_weights(ctx.n, ctx.summaries_c, ctx.summaries_y, ctx.codebook_size, ctx.num_classes,
                                 d["lam"], d["kappa"], d["eps"], d["w_max"])
        self.state["last_scores"] = s.tolist()
        return w, None


# --------------------------------------------------------------------------- Algorithm 7 (FR 15)
def cluster_clients(S: np.ndarray, tau: float, J: int | None) -> np.ndarray:
    """Average-linkage clustering on D = 1 - S, cut at distance ``tau`` (or into J clusters)."""
    D = np.clip(1.0 - S, 0, 2)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    return (fcluster(Z, t=J, criterion="maxclust") if J else fcluster(Z, t=tau, criterion="distance")) - 1


def similarity_stats(S: np.ndarray) -> dict:
    off = S[~np.eye(len(S), dtype=bool)]
    return {"mean": round(float(off.mean()), 5), "std": round(float(off.std()), 5),
            "min": round(float(off.min()), 5), "max": round(float(off.max()), 5)}


def cut_distance(S: np.ndarray, ccfg: dict) -> float:
    """Dendrogram cut. ``absolute``: tau. ``quantile``: the tau_quantile of this round's pairwise
    distances, so the cut follows the actual spread of update cosines instead of a fixed scale."""
    if ccfg.get("tau_mode", "absolute") == "absolute":
        return float(ccfg["tau"])
    D = np.clip(1.0 - S[~np.eye(len(S), dtype=bool)], 0, 2)
    return float(np.quantile(D, ccfg["tau_quantile"]))


class ClusterTracker:
    """Align labels to last round's with Hungarian matching on 1 - Jaccard (display only)."""

    @staticmethod
    def align(clusters: list[list[int]], prev: list[list[int]] | None) -> list[list[int]]:
        if not prev:
            return clusters
        cost = np.array([[1 - len(set(c) & set(p)) / len(set(c) | set(p)) for p in prev] for c in clusters])
        rows, cols = linear_sum_assignment(cost)
        matched = dict(zip(cols.tolist(), rows.tolist()))           # prev slot -> new cluster
        order = [matched[j] for j in range(len(prev)) if j in matched]
        order += [i for i in range(len(clusters)) if i not in order]
        return [clusters[i] for i in order]


def clustered_groups(ctx, ccfg, info: dict | None = None) -> tuple[list[np.ndarray], float | None]:
    """Groups of positions into ctx.n, and the cut distance used (None = no clustering this round).

    Degeneracy guard: when update cosines carry no usable structure (spread below
    ``min_spread``) or the cut leaves every client a singleton, the round falls back to a
    single group and says so in ``info["cluster_status"]``, instead of silently producing
    m singleton "clusters" (which made clustered = uniform mean and hybrid damping inert).
    """
    info = info if info is not None else {}
    m = len(ctx.n)
    if ctx.round <= ccfg["warmup_rounds"] or m < 3:
        info["cluster_status"] = "warmup" if ctx.round <= ccfg["warmup_rounds"] else "too_few_clients"
        return [np.arange(m)], None
    S = ctx.similarity()
    st = similarity_stats(S)
    info["similarity"] = st
    if ccfg["num_clusters"]:
        labels = cluster_clients(S, 0.0, ccfg["num_clusters"])
        cut = None
        info["cluster_status"] = "fixed_J"
    else:
        if st["std"] < ccfg.get("min_spread", 0.0):
            info["cluster_status"] = "degenerate_no_spread"
            return [np.arange(m)], None
        cut = cut_distance(S, ccfg)
        info["cut_distance"] = round(cut, 5)
        labels = cluster_clients(S, cut, None)
        info["cluster_status"] = "ok"
    groups = [np.flatnonzero(labels == g) for g in np.unique(labels) if (labels == g).any()]   # drop empty
    if len(groups) == m and not ccfg["num_clusters"]:
        info["cluster_status"] = "degenerate_all_singletons"
        return [np.arange(m)], None
    return groups, cut


def cluster_omega(nG: np.ndarray, rule: str) -> np.ndarray:
    """Cluster weights. volume: N_G/N (makes clustered == FedAvg); equal: 1/|G|; sqrt: ~ sqrt(N_G)."""
    if rule == "volume":
        o = nG.astype(float)
    elif rule == "sqrt":
        o = np.sqrt(nG.astype(float))
    else:
        o = np.ones(len(nG))
    return o / o.sum()


@register("clustered")
class Clustered(Strategy):
    config_sections = ("clustered",)

    def compute_weights(self, ctx):
        c = self.cfg["clustered"]
        self.round_info = {}
        groups, _ = clustered_groups(ctx, c, self.round_info)
        nG = np.array([ctx.n[G].sum() for G in groups])
        omega = cluster_omega(nG, c["rule"])
        w = np.zeros(len(ctx.n))
        for gi, G in enumerate(groups):
            w[G] = omega[gi] * ctx.n[G] / nG[gi]
        self.round_info["num_clusters"] = len(groups)
        return w, self._clusters(ctx, groups)

    def _clusters(self, ctx, groups):
        clusters = [[ctx.client_ids[i] for i in G] for G in groups]
        clusters = ClusterTracker.align(clusters, self.state.get("prev_clusters"))
        self.state["prev_clusters"] = clusters
        return clusters


# --------------------------------------------------------------------------- Algorithm 8 (FR 15a)
@register("hybrid")
class Hybrid(Clustered):
    needs_summary = True
    config_sections = ("diversity", "clustered", "hybrid")

    def compute_weights(self, ctx):
        d, c, h = self.cfg["diversity"], self.cfg["clustered"], self.cfg["hybrid"]
        Q, C = ctx.codebook_size, ctx.num_classes
        self.round_info = {}
        groups, cut = clustered_groups(ctx, c, self.round_info)
        self.round_info["num_clusters"] = len(groups)
        if len(groups) == 1:
            w, s = diversity_weights(ctx.n, ctx.summaries_c, ctx.summaries_y, Q, C,
                                     d["lam"], d["kappa"], d["eps"], d["w_max"])
            self.round_info["damping"] = "n/a (one cluster)"
            return w, self._clusters(ctx, groups)
        hc, hy = np.asarray(ctx.summaries_c, float), np.asarray(ctx.summaries_y, float)
        nG = np.array([ctx.n[G].sum() for G in groups])
        hG_c = np.stack([(ctx.n[G, None] * hc[G]).sum(0) / ctx.n[G].sum() for G in groups])
        hG_y = np.stack([(ctx.n[G, None] * hy[G]).sum(0) / ctx.n[G].sum() for G in groups])
        omega, s = diversity_weights(nG, hG_c, hG_y, Q, C, d["lam"], d["kappa"], d["eps"], h["omega_max"])
        # Outlier damping (SRS 4.9 limitation). A cluster is an outlier if it is a singleton or its
        # mean intra-cluster cosine is below 1 - cut, i.e. it would not hold together at this round's
        # own cut (same scale as the clustering, so it adapts with tau_mode=quantile). Damping is
        # relative: if every cluster (or none) is flagged it would only rescale omega uniformly and
        # cancel in the renormalisation, so it is skipped and logged as inert.
        S = ctx.similarity()
        thr = 1.0 - (cut if cut is not None else c["tau"])
        flagged = []
        for gi, G in enumerate(groups):
            if len(G) == 1:
                flagged.append(gi)
                continue
            intra = S[np.ix_(G, G)][~np.eye(len(G), dtype=bool)].mean()
            if intra < thr:
                flagged.append(gi)
        if 0 < len(flagged) < len(groups):
            omega = omega.copy()
            omega[flagged] *= h["rho"]
            omega = omega / omega.sum()
            self.round_info["damping"] = "applied"
        else:
            self.round_info["damping"] = "inert (all clusters flagged)" if flagged else "none flagged"
        self.round_info["damped_clusters"] = [[ctx.client_ids[i] for i in groups[g]] for g in flagged]
        w = np.zeros(len(ctx.n))
        for gi, G in enumerate(groups):
            w[G] = omega[gi] * ctx.n[G] / nG[gi]
        self.state["cluster_scores"] = s.tolist()
        return w, self._clusters(ctx, groups)
