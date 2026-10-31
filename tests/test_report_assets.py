"""Tests for tools/report_assets.py (a plain script, loaded by file path)."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    spec = importlib.util.spec_from_file_location("report_assets", ROOT / "tools" / "report_assets.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["report_assets"] = module                  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


def make_run(runs: Path, strategy: str, alpha: float, dropout: float, seed: int, acc: float, curve: list[float | None]) -> None:
    d = runs / f"x-final-{strategy}-a{alpha:g}-p{dropout:g}-s{seed}-{seed:08x}"
    d.mkdir(parents=True)
    (d / "run_summary.json").write_text(json.dumps({
        "status": "completed", "strategy": strategy, "alpha": alpha, "dropout_rate": dropout, "seed": seed,
        "final_test_metrics": {"accuracy": acc, "macro_f1": acc - 0.01}, "client_dispersion": {"worst_client_accuracy": acc - 0.1},
        "total_time_s": 60.0, "cumulative_bytes": {"up": 1000, "down": 1000}}))
    (d / "rounds.jsonl").write_text("".join(json.dumps({"round": i + 1, "val_metrics": None if v is None else {"accuracy": v}}) + "\n"
                                           for i, v in enumerate(curve)))


def build(runs: Path) -> None:
    for alpha in (0.1, 0.5):
        for dropout in (0.0, 0.2):
            for seed in (1, 2, 3):
                base = 0.80 + 0.02 * seed
                make_run(runs, "fedavg", alpha, dropout, seed, base, [0.5, 0.7, 0.8, 0.7, 0.8, 0.7, 0.8, 0.7])
                make_run(runs, "diversity", alpha, dropout, seed, base + 0.03, [0.5, 0.7, 0.8, 0.8, 0.81, 0.8, 0.81, 0.8])


def test_paired_diff_matches_hand_computation(tmp_path):
    tool = load_tool()
    build(tmp_path)
    runs = tool.load_runs(tmp_path)
    pairs = {(r.alpha, r.dropout, r.seed) for r in runs}
    res = tool.paired_diff(runs, "diversity", "accuracy", pairs)
    assert res["n"] == 12 and abs(res["mean"] - 0.03) < 1e-9
    assert res["wins"] == 12 and res["losses"] == 0
    assert tool.paired_diff(runs, "diversity", "oscillation", pairs)["mean"] < 0          # smoother curve: negative difference


def test_skipped_rounds_carry_the_previous_validation_value(tmp_path):
    tool = load_tool()
    make_run(tmp_path, "fedavg", 0.1, 0.5, 1, 0.8, [0.6, None, 0.7])
    (run,) = tool.load_runs(tmp_path)
    assert list(run.val) == [0.6, 0.6, 0.7]


def test_main_writes_every_table_and_figure(tmp_path):
    tool = load_tool()
    build(tmp_path / "runs")
    out = tmp_path / "report"
    assert tool.main([str(tmp_path / "runs"), "--out", str(out)]) == 0
    tables = {p.stem for p in (out / "tables").glob("*.csv")}
    assert {"accuracy_by_cell", "paired_vs_fedavg", "stability", "collapse_counts", "collapsed_runs", "run_cost"} <= tables
    assert len(list((out / "figures").glob("*.png"))) == 4
    with open(out / "tables" / "paired_vs_fedavg.csv") as f:
        rows = list(csv.DictReader(f))
    acc_all = next(r for r in rows if r["metric"] == "final test accuracy" and r["subset"] == "all pairs")
    assert acc_all["strategy"] == "diversity" and acc_all["diff (pp)"] == "+3.00" and acc_all["wins/losses"] == "12/0"


def test_main_reports_when_nothing_is_found(tmp_path):
    assert load_tool().main([str(tmp_path), "--out", str(tmp_path / "o")]) == 2
