"""Partitioner: non-IID client shards and the partition report (FR 5-7, UC2, Alg. 1).

Heterogeneity knob. The grid key ``alpha`` is mapped per mode so that a *smaller*
value always means *more skew* (the heatmap axis reads one way):
  dirichlet : alpha  -> Dirichlet concentration alpha            (label skew)
  domain/   : alpha  -> eta = 1 / (1 + alpha), the home-attribute fraction
  language
  quantity  : alpha  -> beta, Dirichlet concentration of shard sizes
  iid       : ignored
``quantity_beta`` (optional) layers quantity skew on top of any other mode.

Heterogeneity measure: H = (1/K) sum_k JS(p_k || p), in [0, ln 2] (natural log),
over the label distribution (dirichlet/quantity/iid) or the attribute distribution
(domain/language).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .loader import TextDataset


@dataclass
class Partition:
    train: list[TextDataset]          # D_k (90%)
    val: list[TextDataset]            # V_k (10%)
    test: TextDataset                 # T (touched once, after training, unless test_every_round)
    reference: TextDataset            # Q (coordinator-only, disjoint; builds the codebook)
    H: float
    report: dict = field(default_factory=dict)
    global_val: TextDataset | None = None   # V_g (coordinator-held, disjoint; model selection / tuning)


# --------------------------------------------------------------------------- helpers
def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def heterogeneity(groups: list[np.ndarray], values: np.ndarray) -> float:
    """Mean JS divergence between each client's distribution over ``values`` and the pooled one."""
    cats = np.unique(values)
    pos = {c: i for i, c in enumerate(cats)}
    hists = []
    for g in groups:
        h = np.zeros(len(cats))
        for v in values[g]:
            h[pos[v]] += 1
        hists.append(h)
    pooled = np.sum(hists, axis=0)
    return float(np.mean([js_divergence(h, pooled) for h in hists])) if hists else 0.0


def stratified_split(labels: np.ndarray, frac: float, rng: np.random.Generator):
    take, keep = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n = int(round(frac * len(idx)))
        take.extend(idx[:n]); keep.extend(idx[n:])
    return np.sort(np.array(take, dtype=np.int64)), np.sort(np.array(keep, dtype=np.int64))


# --------------------------------------------------------------------------- shard builders
def dirichlet_label(labels: np.ndarray, K: int, alpha: float, rng) -> list[np.ndarray]:
    shards = [[] for _ in range(K)]
    alpha = max(alpha, 1e-3)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        q = rng.dirichlet(alpha * np.ones(K))
        cuts = np.floor(len(idx) * np.cumsum(q)).astype(int)[:-1]
        for k, piece in enumerate(np.split(idx, cuts)):
            shards[k].extend(piece.tolist())
    return [np.array(s, dtype=np.int64) for s in shards]


def attribute_skew(attrs: np.ndarray, K: int, eta: float, rng) -> list[np.ndarray]:
    values = sorted(np.unique(attrs).tolist())
    pools = {v: list(rng.permutation(np.flatnonzero(attrs == v))) for v in values}
    everything = set(range(len(attrs)))
    size = len(attrs) // K
    shards = []
    for k in range(K):
        home = values[k % len(values)]                      # round-robin home value
        n_home = int(round(eta * size))
        mine = [pools[home].pop() for _ in range(min(n_home, len(pools[home])))]
        while len(mine) < size:                             # rest uniformly from all values
            avail = [v for v in values if pools[v]]
            if not avail:
                break
            mine.append(pools[avail[int(rng.integers(len(avail)))]].pop())
        shards.append(np.array(sorted(mine), dtype=np.int64))
        everything -= set(mine)
    return shards


def quantity_sizes(n: int, K: int, beta: float, rng) -> np.ndarray:
    pi = rng.dirichlet(max(beta, 1e-3) * np.ones(K))
    sizes = np.floor(pi * n).astype(int)
    sizes[np.argmax(sizes)] += n - sizes.sum()
    return sizes


def quantity_skew(n: int, K: int, beta: float, rng) -> list[np.ndarray]:
    perm = rng.permutation(n)
    cuts = np.cumsum(quantity_sizes(n, K, beta, rng))[:-1]
    return [np.sort(s) for s in np.split(perm, cuts)]


def enforce_min_size(shards: list[np.ndarray], n_min: int, rng) -> list[dict]:
    """Algorithm 1 line 5: move random examples from the largest shards to deficient ones."""
    adjustments = []
    while True:
        sizes = [len(s) for s in shards]
        k_small = int(np.argmin(sizes))
        if sizes[k_small] >= n_min:
            break
        k_big = int(np.argmax(sizes))
        if sizes[k_big] - 1 < n_min:
            raise ValueError("not enough data to give every client min_size examples")
        j = int(rng.integers(sizes[k_big]))
        moved = shards[k_big][j]
        shards[k_big] = np.delete(shards[k_big], j)
        shards[k_small] = np.sort(np.append(shards[k_small], moved))
        adjustments.append({"from": k_big, "to": k_small})
    return adjustments


# --------------------------------------------------------------------------- Algorithm 1 (+ modes)
def partition(data: TextDataset, K: int, pcfg: dict, dcfg: dict, rng: np.random.Generator) -> Partition:
    mode, a = pcfg["mode"], float(pcfg["alpha"])
    test_idx, rest = stratified_split(data.labels, dcfg["test_frac"], rng)
    ref_rel, pool_rel = stratified_split(data.labels[rest], dcfg["reference_frac"] / (1 - dcfg["test_frac"]), rng)
    ref_idx, rest = rest[ref_rel], rest[pool_rel]
    # global validation V_g: stratified, drawn from the IID remainder before sharding, so it follows
    # the pooled distribution (like T) and is disjoint from T, Q and every client shard
    gv_frac = dcfg.get("global_val_frac", 0.0) / (1 - dcfg["test_frac"] - dcfg["reference_frac"])
    if gv_frac > 0:
        gv_rel, pool_rel = stratified_split(data.labels[rest], gv_frac, rng)
        gval_idx, pool_idx = rest[gv_rel], rest[pool_rel]
    else:
        gval_idx, pool_idx = np.zeros(0, dtype=np.int64), rest
    pool = data.subset(pool_idx)

    attempts = 0
    for attempts in range(1, pcfg["max_tries"] + 1):
        if mode == "dirichlet":
            shards = dirichlet_label(pool.labels, K, a, rng)
        elif mode in ("domain", "language"):
            shards = attribute_skew(pool.attributes, K, 1.0 / (1.0 + a), rng)
        elif mode == "quantity":
            shards = quantity_skew(len(pool), K, a, rng)
        else:  # iid
            shards = [np.sort(s) for s in np.array_split(rng.permutation(len(pool)), K)]
        if pcfg.get("quantity_beta") and mode != "quantity":   # combine quantity skew with any mode
            target = quantity_sizes(sum(map(len, shards)), K, pcfg["quantity_beta"], rng)
            scale = target / max(target.max(), 1)
            shards = [np.sort(rng.choice(s, max(1, int(len(s) * f)), replace=False)) if len(s) else s
                      for s, f in zip(shards, scale)]
        if min(map(len, shards)) >= pcfg["min_size"]:
            break
    adjustments = enforce_min_size(shards, pcfg["min_size"], rng)

    train, val, groups = [], [], []
    for s in shards:
        s = rng.permutation(s)
        n_val = max(1, int(round(dcfg["val_frac"] * len(s)))) if dcfg["val_frac"] > 0 else 0
        val.append(pool.subset(np.sort(s[:n_val])))
        train.append(pool.subset(np.sort(s[n_val:])))
        groups.append(s)

    measure_on = pool.attributes if mode in ("domain", "language") else pool.labels
    H = heterogeneity(groups, measure_on)
    report = {
        "mode": mode, "alpha": a, "num_clients": K, "H": H, "H_over": "attribute" if mode in ("domain", "language") else "label",
        "attempts": attempts, "min_size_adjustments": len(adjustments),
        "test_size": len(test_idx), "reference_size": len(ref_idx), "global_val_size": len(gval_idx),
        "label_names": data.label_names,
        "clients": [{
            "client": k, "train_size": len(train[k]), "val_size": len(val[k]),
            "label_hist": np.bincount(train[k].labels, minlength=data.num_classes).tolist(),
            "attribute_hist": {str(v): int(c) for v, c in zip(*np.unique(train[k].attributes, return_counts=True))},
        } for k in range(K)],
    }
    return Partition(train, val, data.subset(test_idx), data.subset(ref_idx), H, report, data.subset(gval_idx))


def write_report(part: Partition, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(part.report, f, indent=2)
