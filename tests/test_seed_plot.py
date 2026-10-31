"""Tests for tools/seed_plot.py (a plain script, loaded by file path)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    spec = importlib.util.spec_from_file_location("seed_plot", ROOT / "tools" / "seed_plot.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["seed_plot"] = module                 # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


def make_run(runs: Path, strategy: str, alpha: str, dropout: str, seed: int, acc, status="completed") -> None:
    d = runs / f"agnews-final-{strategy}-a{alpha}-p{dropout}-s{seed}-{seed:08x}"
    d.mkdir(parents=True)
    summary = {"status": status}
    if acc is not None:
        summary["final_test_metrics"] = {"accuracy": acc}
    (d / "run_summary.json").write_text(json.dumps(summary))


def test_reads_only_completed_runs_with_the_metric(tmp_path):
    tool = load_tool()
    make_run(tmp_path, "fedavg", "0.1", "0", 1, 0.87)
    make_run(tmp_path, "fedavg", "0.1", "0.2", 2, 0.75)
    make_run(tmp_path, "hybrid", "5.0", "0.5", 3, 0.91)
    make_run(tmp_path, "fedavg", "0.1", "0", 3, None)                 # no final test metric: skipped
    make_run(tmp_path, "diversity", "0.5", "0", 1, 0.9, "failed")     # not completed: skipped
    (tmp_path / "not-a-run").mkdir()                                   # unrelated folder: ignored
    pts = tool.read_points(tmp_path)
    assert sorted((p.strategy, p.alpha, p.dropout, p.seed) for p in pts) == [
        ("fedavg", 0.1, 0.0, 1), ("fedavg", 0.1, 0.2, 2), ("hybrid", 5.0, 0.5, 3)]


def test_main_writes_a_figure_with_a_dot_and_a_mean_per_group(tmp_path):
    tool = load_tool()
    runs = tmp_path / "runs"
    for s in (1, 2, 3):
        make_run(runs, "fedavg", "0.1", "0", s, 0.8 + 0.01 * s)
        make_run(runs, "diversity", "0.1", "0", s, 0.88 + 0.001 * s)
    out = tmp_path / "figs" / "p.html"
    assert tool.main([str(runs), "--out", str(out)]) == 0
    assert out.exists() and "per-seed" in out.read_text().lower()
    fig = tool.build_figure(tool.read_points(runs))
    dots = [t for t in fig.data if t.mode == "markers" and t.marker.symbol != "line-ew"]
    assert sorted(len(t.x) for t in dots) == [3, 3]                    # three seeds per strategy


def test_main_reports_when_nothing_is_found(tmp_path):
    tool = load_tool()
    assert tool.main([str(tmp_path), "--out", str(tmp_path / "x.html")]) == 2
