"""Tests for the helper scripts in tools/ (they are plain scripts, so they are loaded by file path)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from fedtune.config import load_config

ROOT = Path(__file__).resolve().parents[1]
TUNE_CFG = ROOT / "configs" / "tune.yaml"
SELECT = "mean val accuracy over last 3 rounds"


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module                      # dataclasses need the module to be registered
    spec.loader.exec_module(module)
    return module


def write_run(runs: Path, name: str, records: list[dict]) -> None:
    run = runs / name
    run.mkdir(parents=True)
    (run / "rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def rnd(i: int, status: str | None = None, clusters: int | None = None, skipped: bool = False) -> dict:
    info = {} if status is None else {"cluster_status": status, "similarity": {"std": 0.2}}
    if clusters is not None:
        info["num_clusters"] = clusters
    return {"round": i, "skipped": skipped, "strategy_info": info}


# ---------------------------------------------------------------- check_clustering

def test_check_clustering_counts_only_eligible_rounds(tmp_path, capsys):
    tool = load_tool("check_clustering")
    write_run(tmp_path, "agnews-clustered-a0.5-p0.2-s1-aaaa1111", [
        rnd(1, "warmup"),                       # neutral: excluded from the share
        rnd(2, "ok", clusters=3),
        rnd(3, "ok", clusters=2),
        rnd(4, "degenerate_no_spread"),
        rnd(5, skipped=True),                   # dropped round: ignored
    ])
    assert tool.main([str(tmp_path), "--min-share", "0.5"]) == 0      # 2 of 3 eligible rounds are active
    assert "active in 2/3" in capsys.readouterr().out
    assert tool.main([str(tmp_path), "--min-share", "0.9"]) == 1


def test_check_clustering_flags_all_fallback(tmp_path):
    tool = load_tool("check_clustering")
    write_run(tmp_path, "agnews-hybrid-a0.5-p0.2-s1-bbbb2222",
              [rnd(i, "degenerate_all_singletons") for i in range(1, 6)])
    assert tool.main([str(tmp_path)]) == 1


def test_check_clustering_ignores_other_strategies(tmp_path):
    tool = load_tool("check_clustering")
    write_run(tmp_path, "agnews-fedavg-a0.5-p0.2-s1-cccc3333", [rnd(1), rnd(2)])
    assert tool.main([str(tmp_path)]) == 2


# ---------------------------------------------------------------- merge_tuning

def best_json(folder: Path, strategy: str, params: dict) -> Path:
    folder.mkdir(parents=True)
    (folder / "best_params.json").write_text(json.dumps({
        "tuning_seed": 1000, "select": SELECT,
        "best": {strategy: {"params": params, "score": 0.9, "trial": 1, "run_id": "x"}}}))
    return folder


def fedavg_tuned_yaml(folder: Path, mutate=lambda cfg: None) -> Path:
    """A `tuned.yaml` as `fedtune tune` writes it when only fedavg was tuned."""
    cfg = load_config(TUNE_CFG)
    cfg["experiment"]["strategy"] = ["fedavg"]
    cfg["strategy_overrides"] = {"fedavg": {"train.lr": 0.002}}
    cfg["tune"]["space"] = {}
    mutate(cfg)
    folder.mkdir(parents=True)
    (folder / "tuned.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return folder


def three_other_strategies(root: Path) -> list[Path]:
    return [
        best_json(root / "diversity", "diversity", {"train.lr": 0.001, "aggregation.diversity.lam": 0.25}),
        best_json(root / "clustered", "clustered", {"train.lr": 0.001}),
        best_json(root / "hybrid", "hybrid", {"train.lr": 0.0005}),
    ]


def test_merge_tuning_accepts_json_and_yaml_inputs(tmp_path):
    tool = load_tool("merge_tuning")
    inputs = [fedavg_tuned_yaml(tmp_path / "fedavg")] + three_other_strategies(tmp_path)
    out = tmp_path / "out"
    assert tool.main([*map(str, inputs), "--config", str(TUNE_CFG), "--out", str(out)]) == 0
    merged = load_config(out / "tuned.yaml")                          # must validate like `fedtune grid` loads it
    assert merged["strategy_overrides"]["fedavg"] == {"train.lr": 0.002}
    assert merged["strategy_overrides"]["hybrid"] == {"train.lr": 0.0005}
    assert merged["experiment"]["name"] == "agnews-final" and merged["evaluation"]["final_test"] is True


def test_merge_tuning_rejects_yaml_from_a_different_config(tmp_path):
    tool = load_tool("merge_tuning")
    bad = fedavg_tuned_yaml(tmp_path / "fedavg", lambda cfg: cfg["data"].update(max_length=128))
    with pytest.raises(SystemExit, match="differs"):
        tool.main([str(bad), *map(str, three_other_strategies(tmp_path)),
                   "--config", str(TUNE_CFG), "--out", str(tmp_path / "out")])


def test_merge_tuning_requires_every_strategy_exactly_once(tmp_path):
    tool = load_tool("merge_tuning")
    only_two = [fedavg_tuned_yaml(tmp_path / "fedavg"), best_json(tmp_path / "diversity", "diversity", {})]
    with pytest.raises(SystemExit, match="no tuning result"):
        tool.main([*map(str, only_two), "--config", str(TUNE_CFG), "--out", str(tmp_path / "out")])
    twice = [fedavg_tuned_yaml(tmp_path / "a"), fedavg_tuned_yaml(tmp_path / "b")]
    with pytest.raises(SystemExit, match="more than one input"):
        tool.main([*map(str, twice), "--config", str(TUNE_CFG), "--out", str(tmp_path / "out2")])


# ---------------------------------------------------------------- fingerprint

def test_fingerprint_depends_only_on_code_and_configs(tmp_path):
    tool = load_tool("fingerprint")
    (tmp_path / "fedtune").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "fedtune" / "a.py").write_text("x = 1\n")
    (tmp_path / "configs" / "c.yaml").write_text("a: 1\n")
    base = tool.fingerprint(tmp_path)
    assert len(base) == 16

    for extra in ("tools/t.py", "notebooks/n.ipynb", "results/r.csv", "README.md"):   # outside the hashed set
        (tmp_path / extra).parent.mkdir(exist_ok=True)
        (tmp_path / extra).write_text("anything\n")
    assert tool.fingerprint(tmp_path) == base

    (tmp_path / "fedtune" / "a.py").write_text("x = 2\n")                              # inside the hashed set
    assert tool.fingerprint(tmp_path) != base


def test_fingerprint_rejects_a_folder_that_is_not_a_checkout(tmp_path):
    with pytest.raises(SystemExit, match="FedTune checkout"):
        load_tool("fingerprint").fingerprint(tmp_path)
