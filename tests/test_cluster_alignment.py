"""Tests for tools/cluster_alignment.py (a plain script, loaded by file path)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    spec = importlib.util.spec_from_file_location("cluster_alignment", ROOT / "tools" / "cluster_alignment.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cluster_alignment"] = module
    spec.loader.exec_module(module)
    return module


def make_run(root: Path, clusters: list[list[int]]) -> None:
    d = root / "x-final-clustered-a0.1-p0-s1-00000001"
    d.mkdir(parents=True)
    langs = ["de", "de", "de", "fr", "fr", "fr"]                       # 6 clients, two language groups
    (d / "partition_report.json").write_text(json.dumps({"H_over": "attribute", "clients": [
        {"client": i, "attribute_hist": {l: (90 if l == langs[i] else 5) for l in ("de", "fr")}, "label_hist": [1, 1, 1]} for i in range(6)]}))
    (d / "run_summary.json").write_text(json.dumps({"alpha": 0.1}))
    rec = {"round": 2, "skipped": False, "strategy_info": {"cluster_status": "ok"}, "clusters": clusters}
    skipped = {"round": 3, "skipped": True, "strategy_info": {}, "clusters": []}
    (d / "rounds.jsonl").write_text(json.dumps(rec) + "\n" + json.dumps(skipped) + "\n")


def test_ari_basics():
    tool = load_tool()
    assert tool.adjusted_rand_index([0, 0, 1, 1], ["a", "a", "b", "b"]) == 1.0           # same partition, different names
    assert abs(tool.adjusted_rand_index([0, 1, 0, 1], ["a", "a", "b", "b"]) + 0.5) < 1e-9  # worse than chance


def test_perfect_clusters_score_one_and_beat_the_shuffled_baseline(tmp_path, capsys):
    tool = load_tool()
    make_run(tmp_path, [[0, 1, 2], [3, 4, 5]])
    res = tool.run_alignment(next(tmp_path.iterdir()), np.random.default_rng(0))
    assert res[0] == 1.0 and res[1] < 0.5 and res[2] == 1 and res[3] == "attribute"      # skipped round ignored
    assert tool.main([str(tmp_path), "--out", str(tmp_path / "o" / "a.csv")]) == 0
    assert (tmp_path / "o" / "a.csv").exists() and "+1.00" in capsys.readouterr().out


def test_nothing_to_report(tmp_path):
    assert load_tool().main([str(tmp_path)]) == 2
