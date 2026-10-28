#!/usr/bin/env python3
"""Summarise the final grid after every teammate's runs/ have been copied into ONE folder.

Unlike `fedtune grid` this never trains anything: it only reads finished runs and writes
grid_summary.csv, paired_vs_fedavg.csv and missing_cells.json, so a missing cell can never
silently start a multi-minute training job on your laptop.

    # all teammates' run folders copied into  merged/runs/<run_id>/...
    python tools/merge_grid.py -c tuned_final/tuned.yaml --runs merged/runs

Outputs go to  merged/grids/<experiment name>/  (next to the runs folder).
If cells are missing, the script lists them and exits with status 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fedtune.config import load_config
from fedtune import experiment


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="the shared tuned.yaml")
    ap.add_argument("--runs", required=True, help="folder holding every run directory")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, [f"experiment.output_dir={Path(a.runs)}"])
    gdir = experiment.summarise_grid(cfg)
    missing = json.loads((gdir / "missing_cells.json").read_text())
    total = len(experiment.grid_cells(cfg))
    print(f"{total - len(missing)} / {total} cells complete; summary in {gdir}")
    if missing:
        print("missing cells (run these shards again, or these cells):")
        for m in missing:
            print("  ", m)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
