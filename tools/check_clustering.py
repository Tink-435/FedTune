#!/usr/bin/env python3
"""Check whether the clustering strategies actually cluster.

`clustered` and `hybrid` fall back to a single group (== FedAvg / diversity-only weights) in rounds
where the client updates are too similar to separate or every client ends up alone. The fallback is
logged per round in ``strategy_info.cluster_status`` of each run's ``rounds.jsonl``. This tool counts
those statuses so you can tell, BEFORE spending hours on the grid, whether clustering is doing anything.

Usage (from the repository root):

    python tools/check_clustering.py tuning_in/clustered/runs tuning_in/hybrid/runs
    python tools/check_clustering.py merged/runs --min-share 0.5

Each path may be a ``runs`` folder, a single run folder, or a ``rounds.jsonl`` file.
Runs of other strategies (fedavg, diversity) are ignored.

Exit status: 0 = clustering active in at least ``--min-share`` of the eligible rounds,
             1 = below that share, 2 = no clustering runs found.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# cluster_status values written by fedtune/coordinator/strategies.py
ACTIVE = frozenset({"ok", "fixed_J"})                                      # real clustering happened
DEGENERATE = frozenset({"degenerate_no_spread", "degenerate_all_singletons"})  # fell back to one group
NEUTRAL = frozenset({"warmup", "too_few_clients"})                         # nothing to cluster this round

_STRATEGY_IN_NAME = re.compile(r"-(clustered|hybrid)-a")                   # run folder: <name>-<strategy>-a<alpha>-...


@dataclass
class RunStats:
    """Per-run tally of the clustering status over its (non-skipped) rounds."""
    name: str
    strategy: str
    statuses: Counter = field(default_factory=Counter)
    clusters_when_active: list[int] = field(default_factory=list)
    similarity_std: list[float] = field(default_factory=list)

    @property
    def active(self) -> int:
        return sum(n for s, n in self.statuses.items() if s in ACTIVE)

    @property
    def degenerate(self) -> int:
        return sum(n for s, n in self.statuses.items() if s in DEGENERATE)

    @property
    def eligible(self) -> int:
        """Rounds where clustering was actually attempted (warm-up and too-few-client rounds excluded)."""
        return self.active + self.degenerate

    @property
    def share(self) -> float | None:
        return self.active / self.eligible if self.eligible else None


def find_rounds_files(paths: list[Path]) -> list[Path]:
    """Expand the given files/folders into a sorted, de-duplicated list of rounds.jsonl files."""
    found: set[Path] = set()
    for p in paths:
        if p.is_file() and p.name == "rounds.jsonl":
            found.add(p)
        elif p.is_dir():
            found.update(p.rglob("rounds.jsonl"))
    return sorted(found)


def read_run(rounds_file: Path) -> RunStats | None:
    """Tally one run; None if it is not a clustered/hybrid run or has no clustering information."""
    run_name = rounds_file.parent.name
    m = _STRATEGY_IN_NAME.search(run_name)
    if not m:
        return None
    stats = RunStats(name=run_name, strategy=m.group(1))
    with open(rounds_file) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("skipped"):                                         # round dropped: nothing was aggregated
                continue
            info = rec.get("strategy_info") or {}
            status = info.get("cluster_status")
            if status is None:
                continue
            stats.statuses[status] += 1
            if status in ACTIVE and "num_clusters" in info:
                stats.clusters_when_active.append(int(info["num_clusters"]))
            sim = info.get("similarity")
            if isinstance(sim, dict) and "std" in sim:
                stats.similarity_std.append(float(sim["std"]))
    return stats if stats.statuses else None


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _fmt(x: float | None, spec: str) -> str:
    return "-" if x is None else format(x, spec)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", type=Path, help="runs folder(s), run folder(s) or rounds.jsonl file(s)")
    ap.add_argument("--min-share", type=float, default=0.5,
                    help="minimum fraction of eligible rounds with real clustering (default 0.5; a rule of thumb)")
    a = ap.parse_args(argv)

    runs = [r for r in (read_run(f) for f in find_rounds_files(a.paths)) if r is not None]
    if not runs:
        print("no clustered/hybrid runs with clustering information found under the given paths")
        return 2

    print(f"{'run':<48} {'strat':<9} {'eligible':>8} {'active':>6} {'fallback':>8} {'share':>6} {'clusters':>8} {'sim-std':>7}")
    for r in runs:
        print(f"{r.name[-48:]:<48} {r.strategy:<9} {r.eligible:>8} {r.active:>6} {r.degenerate:>8} "
              f"{_fmt(r.share, '.0%'):>6} {_fmt(_mean(r.clusters_when_active), '.1f'):>8} "
              f"{_fmt(_mean(r.similarity_std), '.3f'):>7}")

    total = Counter()
    for r in runs:
        total.update(r.statuses)
    active = sum(n for s, n in total.items() if s in ACTIVE)
    fallback = sum(n for s, n in total.items() if s in DEGENERATE)
    unknown = {s: n for s, n in total.items() if s not in ACTIVE | DEGENERATE | NEUTRAL}
    eligible = active + fallback
    share = active / eligible if eligible else 0.0

    print(f"\nstatus counts over all runs: {dict(total)}")
    if unknown:
        print(f"note: unrecognised statuses {unknown} were left out of the share")
    print(f"POOLED: clustering active in {active}/{eligible} eligible rounds ({share:.0%}); "
          f"threshold {a.min_share:.0%}")
    if share >= a.min_share:
        print("VERDICT: clustering is doing real work; the grid will not just repeat FedAvg/diversity.")
        return 0
    print("VERDICT: clustering mostly falls back to one group, so clustered ~ FedAvg and hybrid ~ diversity-only.\n"
          "         Do not start the grid yet; decide how to proceed first (see tools/README.md).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
