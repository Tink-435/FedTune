"""Integration tests (NFR 10, Design 2.3.5): smoke run, failure handling, resume, plug-in, grid, export, dashboard."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from fedtune import experiment
from fedtune.config import load_config
from fedtune.coordinator.strategies import REGISTRY

ROOT = Path(__file__).resolve().parents[1]


def tiny_cfg(tmp_path, **over):
    sets = [f"experiment.output_dir={tmp_path / 'runs'}", "experiment.rounds=3", "experiment.num_clients=4",
            "experiment.checkpoint_every=1", "data.max_examples=1200", "dropout.rate=0.25",
            "dropout.min_participants=2", "aggregation.diversity.codebook_size=8",
            "aggregation.diversity.embed_dim=16", "train.lr=1e-2"]
    sets += [f"{k}={v}" for k, v in over.items()]
    return load_config(ROOT / "configs" / "smoke.yaml", sets)


def rounds(cfg, strat):
    return [json.loads(l) for l in (Path(cfg["experiment"]["output_dir"]) / experiment.run_id(cfg, strat) / "rounds.jsonl").read_text().splitlines()]


def test_all_strategies_share_participation_and_finish(tmp_path):
    cfg = tiny_cfg(tmp_path)
    rows = experiment.compare(cfg)
    assert [r["status"] for r in rows] == ["completed"] * 4
    parts = {s: [r["participating_clients"] for r in rounds(cfg, s)] for s in cfg["experiment"]["strategy"]}
    assert len({json.dumps(p) for p in parts.values()}) == 1                   # FR 17
    s = rows[0]
    assert s["adapter_bytes"] / s["full_model_bytes"] < 0.05                     # NFR 3
    run = Path(cfg["experiment"]["output_dir"]) / rows[1]["run_id"]
    assert (run / "summary_log.jsonl").exists()                                 # FR 12 (diversity)
    for f in ("config.resolved.yaml", "partition_report.json", "participation.npy", "run_summary.json"):
        assert (run / f).exists()
    clu = [r for r in rounds(cfg, "clustered") if not r["skipped"] and r["round"] > 1]
    assert all("cluster_status" in r["strategy_info"] for r in clu)                 # degeneracy is logged
    from fedtune.dashboard import app as dash_app
    runs = Path(cfg["experiment"]["output_dir"])
    assert dash_app.fig_similarity(runs, rows[2]["run_id"]).data                   # similarity diagnostics view
    assert len(dash_app.comparison_rows(runs, [r["run_id"] for r in rows])) == 4


def test_nan_update_excluded_and_run_continues(tmp_path):
    cfg = tiny_cfg(tmp_path, **{"experiment.strategy": "[fedavg]", "dropout.rate": 0.0})

    def hook(clients):
        clients[1].fault_rounds = {2}

    s = experiment.run_single(cfg, "fedavg", fault_hook=hook)
    rec = rounds(cfg, "fedavg")[1]
    assert rec["excluded_clients"] == [{"client": 1, "reason": "non-finite"}]
    assert "1" not in rec["aggregation_weights"] and s["status"] == "completed"
    assert np.isfinite(s["final_test_metrics"]["accuracy"])


def test_too_few_participants_skips_round(tmp_path):
    cfg = tiny_cfg(tmp_path, **{"experiment.strategy": "[fedavg]", "dropout.rate": 1.0, "dropout.min_participants": 1})
    s = experiment.run_single(cfg, "fedavg")
    assert s["rounds_skipped"] == 3 and all(r["skipped"] for r in rounds(cfg, "fedavg"))


def test_killed_run_resumes_identically(tmp_path):
    ref_cfg = tiny_cfg(tmp_path / "ref", **{"experiment.strategy": "[clustered]", "experiment.rounds": 4})
    experiment.run_single(ref_cfg, "clustered")
    cfg = tiny_cfg(tmp_path / "crash", **{"experiment.strategy": "[clustered]", "experiment.rounds": 4,
                                          "experiment.checkpoint_every": 2})

    def crash(clients):
        def boom(b):
            raise KeyboardInterrupt("simulated kill")
        for c in clients.values():
            orig = c.train
            c.train = (lambda o: (lambda b: boom(b) if b.round == 4 else o(b)))(orig)

    with pytest.raises(KeyboardInterrupt):
        experiment.run_single(cfg, "clustered", fault_hook=crash)
    run_dir = Path(cfg["experiment"]["output_dir"]) / experiment.run_id(cfg, "clustered")
    assert json.loads((run_dir / "run_summary.json").read_text())["status"] == "failed"   # FR 34
    experiment.run_single(cfg, "clustered", resume=True)
    a, b = rounds(ref_cfg, "clustered"), rounds(cfg, "clustered")
    assert [r["round"] for r in b] == [1, 2, 3, 4]                                         # no round twice
    assert [r["participating_clients"] for r in a] == [r["participating_clients"] for r in b]
    for x, y in zip(a, b):
        assert x["skipped"] == y["skipped"]
        assert abs(x["val_metrics"].get("accuracy", 0) - y["val_metrics"].get("accuracy", 0)) <= 0.005   # FR 3 tolerance


def test_custom_strategy_plugin(tmp_path):
    sys.path.insert(0, str(ROOT / "examples"))
    import custom_strategy  # noqa: F401  registers "uniform"
    assert "uniform" in REGISTRY
    cfg = tiny_cfg(tmp_path, **{"experiment.strategy": "[fedavg,uniform]"})
    rows = experiment.compare(cfg)
    assert rows[1]["strategy"] == "uniform" and rows[1]["status"] == "completed"
    w = [r for r in rounds(cfg, "uniform") if not r["skipped"]][0]["aggregation_weights"]
    assert len(set(w.values())) == 1


def test_product_merge_and_stale_updates(tmp_path):
    cfg = tiny_cfg(tmp_path, **{"experiment.strategy": "[hybrid]", "aggregation.lora_method": "product",
                                "dropout.stale_updates": True, "dropout.rate": 0.4, "experiment.rounds": 4})
    s = experiment.run_single(cfg, "hybrid")
    recs = [r for r in rounds(cfg, "hybrid") if not r["skipped"]]
    assert s["status"] == "completed" and all(r["merge_residual"] is not None for r in recs)


def test_grid_export_and_dashboard(tmp_path):
    cfg = tiny_cfg(tmp_path, **{"experiment.strategy": "[fedavg]", "experiment.rounds": 2,
                                "grid.alpha": "[0.1,5.0]", "grid.dropout_rate": "[0.0,0.5]", "grid.seeds": "[1,2]"})
    gdir = experiment.run_grid(cfg)
    rows = (gdir / "grid_summary.csv").read_text().splitlines()
    assert len(rows) == 1 + 4 and json.loads((gdir / "missing_cells.json").read_text()) == []
    assert (gdir / "paired_vs_fedavg.csv").exists()
    runs = Path(cfg["experiment"]["output_dir"])
    out = experiment.export_run(sorted(p for p in runs.iterdir() if p.is_dir())[0], tmp_path / "exports")
    assert (out / "rounds.csv").exists() and (out / "config.resolved.yaml").exists()
    from fedtune.dashboard import app as dash_app
    dash_app.build_app(runs)                                                 # dashboard loads artifacts
    assert len(dash_app.fig_participation(runs, dash_app.list_runs(runs)[0]).data) == 5
    assert dash_app.fig_heatmap(str(gdir / "grid_summary.csv"), "fedavg", "accuracy").data
    dash_app.fig_heatmap(str(gdir / "grid_summary.csv"), "fedavg", "accuracy", "paired")   # no pairs: empty, no crash
    dash_app.fig_paired(str(gdir / "grid_summary.csv"), "accuracy")


def test_test_set_held_out_until_the_end(tmp_path):
    cfg = tiny_cfg(tmp_path, **{"experiment.strategy": "[fedavg]"})
    s = experiment.run_single(cfg, "fedavg")
    recs = rounds(cfg, "fedavg")
    assert all("test_metrics" not in r for r in recs) and all(r.get("val_metrics") for r in recs if not r["skipped"])
    assert s["final_test_metrics"] and s["final_val_metrics"]
    assert "statistical" in s["reproducibility"]["training"] or "bitwise" in s["reproducibility"]["training"]


def test_changed_config_never_reuses_stale_summary(tmp_path):
    cfg = tiny_cfg(tmp_path, **{"experiment.strategy": "[fedavg]", "experiment.rounds": 2})
    a = experiment.run_single(cfg, "fedavg", resume=True)
    again = experiment.run_single(cfg, "fedavg", resume=True)               # same config: reused
    assert again["total_time_s"] == a["total_time_s"]
    cfg2 = tiny_cfg(tmp_path, **{"experiment.strategy": "[fedavg]", "experiment.rounds": 2, "train.lr": 5e-3})
    assert experiment.run_id(cfg2, "fedavg") != experiment.run_id(cfg, "fedavg")
    b = experiment.run_single(cfg2, "fedavg", resume=True)                  # changed lr: a fresh run
    assert b["config_hash"] != a["config_hash"] and b["run_id"] != a["run_id"]
    # a summary whose hash does not match its directory (e.g. copied by hand) is not trusted either
    run_dir = Path(cfg["experiment"]["output_dir"]) / a["run_id"]
    bad = json.loads((run_dir / "run_summary.json").read_text()); bad["config_hash"] = "deadbeef"
    (run_dir / "run_summary.json").write_text(json.dumps(bad))
    c = experiment.run_single(cfg, "fedavg", resume=True)
    assert c["config_hash"] == a["config_hash"] and c["total_time_s"] != a["total_time_s"]


def test_tuning_selects_on_validation_and_freezes_config(tmp_path):
    space = {"fedavg": {"train.lr": [1e-3, 1e-2]}, "clustered": {"aggregation.clustered.rule": ["equal", "sqrt"]}}
    cfg = tiny_cfg(tmp_path, **{"experiment.strategy": "[fedavg,clustered]", "experiment.rounds": 2,
                                "experiment.name": "tn", "tune.space": json.dumps(space), "tune.trials": 2,
                                "tune.seed": 77, "grid.seeds": "[1]"})
    out = experiment.run_tuning(cfg)
    res = (out / "tuning_results.csv").read_text().splitlines()
    assert len(res) == 1 + 4
    for line in res[1:]:                                                    # the test set was never evaluated
        rid = line.rsplit(",", 1)[1]
        s = json.loads((Path(cfg["experiment"]["output_dir"]) / rid / "run_summary.json").read_text())
        assert s["final_test_metrics"] is None and s["seed"] == 77
    tuned = load_config(out / "tuned.yaml")
    best = json.loads((out / "best_params.json").read_text())["best"]
    assert tuned["strategy_overrides"]["fedavg"]["train.lr"] == best["fedavg"]["params"]["train.lr"]
    assert tuned["evaluation"]["final_test"] is True and tuned["tune"]["space"] == {}
