#!/usr/bin/env python3
"""Do the clusters found by `clustered` / `hybrid` line up with the true skew of the clients?

For every clustering round, the clusters are compared with each client's dominant label (label-skew
partitions) or dominant attribute such as language or domain (attribute-skew partitions) using the adjusted Rand
index: 0 = unrelated to the skew, 1 = identical. A shuffled-cluster baseline (same cluster sizes, random members)
is reported next to it, because a perfect score is not the interesting case; "clearly above the baseline" is.

    python tools/cluster_alignment.py merged/runs --out results/report/tables/cluster_alignment.csv
    python tools/cluster_alignment.py merged_xnli/runs

Reads results only. Exploratory: "dominant class/language of a client" is a crude ground truth, and rounds in
which the strategy fell back to one group are ignored. Exit status: 0 = table written, 2 = no clustering runs found.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from math import comb
from pathlib import Path

import numpy as np

_STRATEGY = re.compile(r"-(clustered|hybrid)-a")


def adjusted_rand_index(a, b) -> float:
    """ARI between two labelings of the same items (Hubert & Arabie)."""
    ia, ib = np.unique(np.asarray(a), return_inverse=True)[1], np.unique(np.asarray(b), return_inverse=True)[1]
    table = np.zeros((ia.max() + 1, ib.max() + 1), dtype=int)
    for i, j in zip(ia, ib):
        table[i, j] += 1
    pairs = sum(comb(int(x), 2) for x in table.ravel())
    rows, cols = sum(comb(int(x), 2) for x in table.sum(1)), sum(comb(int(x), 2) for x in table.sum(0))
    total = comb(int(table.sum()), 2)
    expected, maximum = rows * cols / total, (rows + cols) / 2
    return 1.0 if maximum == expected else (pairs - expected) / (maximum - expected)


def dominant_truth(report: dict) -> tuple[dict[int, object], str]:
    """client id -> its dominant attribute value (attribute-skew partitions) or dominant label (label skew)."""
    if report.get("H_over") == "attribute":
        return {c["client"]: max(c["attribute_hist"], key=c["attribute_hist"].get) for c in report["clients"]}, "attribute"
    return {c["client"]: int(np.argmax(c["label_hist"])) for c in report["clients"]}, "label"


def run_alignment(run_dir: Path, rng: np.random.Generator):
    """(mean ARI, mean shuffled-baseline ARI, number of rounds used, truth kind), or None."""
    report_file, rounds_file, summary_file = run_dir / "partition_report.json", run_dir / "rounds.jsonl", run_dir / "run_summary.json"
    if not (report_file.exists() and rounds_file.exists() and summary_file.exists()):
        return None
    truth, kind = dominant_truth(json.loads(report_file.read_text()))
    real, null = [], []
    for line in rounds_file.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("skipped") or (rec.get("strategy_info") or {}).get("cluster_status") != "ok" or not rec.get("clusters"):
            continue
        labels = [i for i, cluster in enumerate(rec["clusters"]) for _ in cluster]
        ids = [c for cluster in rec["clusters"] for c in cluster]
        target = [truth[c] for c in ids]
        real.append(adjusted_rand_index(labels, target))
        null.append(adjusted_rand_index(rng.permutation(labels), target))
    return (float(np.mean(real)), float(np.mean(null)), len(real), kind) if real else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", type=Path, help="folder holding the run directories")
    ap.add_argument("--out", type=Path, help="optional CSV to write")
    a = ap.parse_args(argv)

    rng = np.random.default_rng(0)
    groups: dict[tuple[str, float, str], list] = {}
    for d in sorted(p for p in a.runs.iterdir() if p.is_dir()):
        m = _STRATEGY.search(d.name)
        res = run_alignment(d, rng) if m else None
        if res is None:
            continue
        alpha = json.loads((d / "run_summary.json").read_text())["alpha"]
        groups.setdefault((m.group(1), float(alpha), res[3]), []).append(res)
    if not groups:
        print("no clustered/hybrid runs with clustering rounds found")
        return 2

    rows = [[s, f"{al:g}", kind, len(v), f"{np.mean([x[0] for x in v]):+.2f}", f"{np.mean([x[1] for x in v]):+.2f}"]
            for (s, al, kind), v in sorted(groups.items())]
    header = ["strategy", "alpha", "skew type", "runs", "mean ARI vs skew", "shuffled baseline"]
    print("  ".join(f"{h:<18}" for h in header))
    for r in rows:
        print("  ".join(f"{c:<18}" for c in r))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
