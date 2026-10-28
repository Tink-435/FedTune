#!/usr/bin/env python3
"""Merge per-strategy `fedtune tune` outputs into ONE frozen tuned.yaml.

Each teammate tunes one strategy and sends back their `tunes/agnews/` folder
(best_params.json + tuning_results.csv). Put them in separate folders, e.g.

    tuning_in/fedavg/best_params.json     tuning_in/diversity/best_params.json
    tuning_in/clustered/best_params.json  tuning_in/hybrid/best_params.json

Run from the fedtune repo root (the folder holding configs/ and pyproject.toml):

    python tools/merge_tuning.py tuning_in/*/ --config configs/tune.yaml --out tuned_final

A strategy's `tuned.yaml` (written by `fedtune tune` next to best_params.json) is accepted too, for
when best_params.json was not kept: pass the .yaml file itself or its folder. Its winner is read from
strategy_overrides, and it is rejected if its other settings differ from --config.

Writes  tuned_final/tuned.yaml  (+ tuning_results_all.csv, best_params_all.json).
The result is built exactly like `fedtune tune` builds its own tuned.yaml
(same fields, same freezing), then re-loaded through the config validator.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import yaml

from fedtune.config import load_config


_SAME = ("data", "model", "train", "dropout", "aggregation", "evaluation", "grid")


def load_input(p: Path, cfg: dict):
    """-> (tuning seed, selection rule, {strategy: {params, score, trial, run_id}}, source path)"""
    if p.is_dir():
        js = sorted(p.rglob("best_params.json"))
        ys = sorted(p.rglob("tuned.yaml"))
        if len(js) == 1:
            p = js[0]
        elif not js and len(ys) == 1:
            p = ys[0]
        else:
            sys.exit(f"{p}: expected one best_params.json (or, failing that, one tuned.yaml); "
                     f"found {len(js)} and {len(ys)}")
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        return d["tuning_seed"], d["select"], d["best"], p
    # tuned.yaml written by `fedtune tune`: the winners are frozen in strategy_overrides
    y = load_config(p)
    for k in _SAME:
        if y[k] != cfg[k]:
            sys.exit(f"{p}: section '{k}' differs from the --config file; "
                     "it was tuned with a different configuration. Re-run that strategy with the shared code/config.")
    for k in ("trials", "alpha", "dropout_rate", "select_metric", "select_last", "seed"):
        if y["tune"][k] != cfg["tune"][k]:
            sys.exit(f"{p}: tune.{k} is {y['tune'][k]!r}, the shared config has {cfg['tune'][k]!r}")
    t = y["tune"]
    best = {s: {"params": prm, "score": None, "trial": None, "run_id": None}
            for s, prm in y["strategy_overrides"].items()
            if s in y["experiment"]["strategy"]}
    return t["seed"], f"mean val {t['select_metric']} over last {t['select_last']} rounds", best, p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="folders (or best_params.json files), one per tuned strategy")
    ap.add_argument("--config", default="configs/tune.yaml", help="the SAME config the tuning used")
    ap.add_argument("--out", default="tuned_final")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    want = list(cfg["experiment"]["strategy"])

    best, seeds, selects = {}, set(), set()
    for inp in a.inputs:
        seed, select, found, bp = load_input(Path(inp), cfg)
        seeds.add(seed)
        selects.add(select)
        for strat, b in found.items():
            if strat in best:
                sys.exit(f"strategy {strat!r} appears in more than one input ({bp}); give each strategy once")
            best[strat] = b
            sc = f"{b['score']:.4f}" if b["score"] is not None else "n/a (read from tuned.yaml)"
            print(f"{strat:<10} score={sc}  trial={b['trial']}  params={b['params'] or '(base settings)'}")

    if seeds != {cfg["tune"]["seed"]} or len(selects) != 1:
        sys.exit(f"inputs disagree with the config: tuning seeds {sorted(seeds)} (config {cfg['tune']['seed']}), "
                 f"selection rules {sorted(selects)}. Re-run the odd one out.")
    missing = [s for s in want if s not in best]
    if missing:
        sys.exit(f"no tuning result for {missing}; every strategy in the config needs one")
    extra = [s for s in best if s not in want]
    if extra:
        sys.exit(f"tuning results for {extra} are not in {a.config}'s strategy list")

    # --- same construction as experiment.run_tuning -------------------------------------------
    tuned = copy.deepcopy(cfg)
    tuned["strategy_overrides"] = {s: {**cfg["strategy_overrides"].get(s, {}), **best[s]["params"]} for s in want}
    tuned["evaluation"]["final_test"] = True
    tuned["experiment"]["name"] = f"{cfg['experiment']['name']}-final"
    tuned["tune"]["space"] = {}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    header = (f"# Frozen by merge_tuning.py from per-strategy `fedtune tune` runs (tuning seed {cfg['tune']['seed']}, "
              "selected on global validation, never on test).\n"
              "# Every teammate must run the grid with THIS file, byte for byte.\n")
    (out / "tuned.yaml").write_text(header + yaml.safe_dump(tuned, sort_keys=False))
    (out / "best_params_all.json").write_text(json.dumps(
        {"tuning_seed": cfg["tune"]["seed"], "select": selects.pop(), "best": best}, indent=2))

    # concatenate the per-strategy trial tables for the report
    rows = []
    for inp in a.inputs:
        for csv_path in sorted(Path(inp).rglob("tuning_results.csv")) if Path(inp).is_dir() else []:
            with open(csv_path, newline="") as f:
                rows += list(csv.DictReader(f))
    if rows:
        with open(out / "tuning_results_all.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    load_config(out / "tuned.yaml")       # must validate, exactly as `fedtune grid` will load it
    print(f"\nwrote {out / 'tuned.yaml'} (validated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
