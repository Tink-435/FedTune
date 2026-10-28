"""Unit tests (NFR 10, Design 2.3.5): partitioning, schedule, boundary, linear algebra, weights, config."""
import ast
from pathlib import Path

import numpy as np
import pytest
import torch

import copy
import warnings

from fedtune.config import ConfigError, ConfigWarning, DEFAULTS, load_config
from fedtune.coordinator import linalg
from fedtune.coordinator.participation import build_schedule
from fedtune.coordinator.strategies import (REGISTRY, ClusterTracker, RoundContext, cap, cluster_clients,
                                            clustered_groups, diversity_weights)
from fedtune.data.loader import load_dataset_from_config
from fedtune.data.partition import partition
from fedtune.protocol import BoundaryViolation, ClientUpdate, GlobalBroadcast
from fedtune.rng import RngService

PKG = Path(__file__).resolve().parents[1] / "fedtune"


# --------------------------------------------------------------------------- config (FR 1-2)
def test_config_rejects_bad_value_naming_key():
    with pytest.raises(ConfigError, match=r"dropout\.rate.*\[0\.0, 1\.0\]"):
        load_config(None, ["dropout.rate=1.4"])
    with pytest.raises(ConfigError, match="unknown configuration key"):
        load_config(None, ["dropout.rat=0.1"])
    with pytest.raises(ConfigError, match="experiment.strategy"):
        load_config(None, ["experiment.strategy=[nope]"])


def test_config_cli_override_wins():
    cfg = load_config(None, ["dropout.rate=0.3", "train.lr=1e-3"])
    assert cfg["dropout"]["rate"] == 0.3 and cfg["train"]["lr"] == 1e-3


# --------------------------------------------------------------------------- partitioning (FR 3, 5-7, UC2)
AG = load_dataset_from_config({**DEFAULTS["data"], "dataset": "ag_news", "max_examples": 1500}, seed=1)


def test_bundled_datasets_load():
    for name, n, k in (("ag_news", 127_600, 4), ("mnli", 412_349, 3), ("xnli", 112_500, 3)):
        d = load_dataset_from_config({**DEFAULTS["data"], "dataset": name, "max_examples": 10**9}, seed=0)
        assert len(d) == n and d.num_classes == k
    x = load_dataset_from_config({**DEFAULTS["data"], "dataset": "xnli", "languages": ["en", "hi"], "max_examples": 10**9}, 0)
    assert set(x.attributes) == {"en", "hi"} and isinstance(x.texts[0], tuple)


def _part(alpha, seed=0, mode="dirichlet"):
    data = AG
    pcfg = {**DEFAULTS["data"]["partition"], "mode": mode, "alpha": alpha}
    return partition(data, 6, pcfg, DEFAULTS["data"], RngService(seed).fresh("partition"))


def test_partition_deterministic():
    a, b = _part(0.5), _part(0.5)
    assert all(x.texts == y.texts for x, y in zip(a.train, b.train))
    assert a.test.texts == b.test.texts


def test_partition_H_falls_as_alpha_grows():
    H = [np.mean([_part(a, seed=s).H for s in range(3)]) for a in (0.1, 0.5, 5.0)]
    assert H[0] > H[1] > H[2] and 0 <= H[2] <= np.log(2)


def test_partition_disjoint_and_min_size():
    p = _part(0.1)
    sets = [set(map(str, d.texts)) for d in p.train]
    assert min(len(d) for d in p.train) + min(len(v) for v in p.val) >= 10
    total = (sum(len(d) for d in p.train) + sum(len(v) for v in p.val) + len(p.test) + len(p.reference)
             + len(p.global_val))
    assert total == 1500


def test_global_validation_set_disjoint_and_stratified():
    p = _part(0.1)
    gv = set(map(str, p.global_val.texts))
    assert len(p.global_val) == round(0.05 * 1500) or abs(len(p.global_val) - 75) <= 4
    assert not gv & set(map(str, p.test.texts)) and not gv & set(map(str, p.reference.texts))
    for d in p.train + p.val:
        assert not gv & set(map(str, d.texts))
    assert len(set(p.global_val.labels.tolist())) == 4          # follows the pooled label distribution


@pytest.mark.parametrize("mode", ["domain", "quantity", "iid"])
def test_other_partition_modes(mode):
    p = _part(0.2, mode=mode)
    assert len(p.train) == 6 and p.report["mode"] == mode


# --------------------------------------------------------------------------- schedule (FR 17, 19, 23)
def test_schedule_nested_across_dropout_rates():
    P1 = build_schedule(10, 20, 0.2, RngService(7).fresh("participation"))
    P2 = build_schedule(10, 20, 0.5, RngService(7).fresh("participation"))
    assert np.all(P1[P2])                    # present at p2 => present at p1
    assert np.array_equal(P1, build_schedule(10, 20, 0.2, RngService(7).fresh("participation")))


def test_rng_streams_independent():
    r = RngService(3)
    a = r.fresh("participation").random(5)
    r.stream("something-new").random(100)   # adding a consumer never shifts another stream
    assert np.array_equal(a, r.fresh("participation").random(5))


# --------------------------------------------------------------------------- boundary (NFR 6, R1)
def test_coordinator_imports_no_client_or_data_code():
    banned = ("fedtune.client", "fedtune.data", "fedtune.evaluation", "fedtune.experiment")
    for f in (PKG / "coordinator").glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.ImportFrom):
                if node.level:                                  # resolve relative to fedtune.coordinator
                    base = ["fedtune", "coordinator"][: 2 - (node.level - 1)]
                    mod = ".".join(base + ([node.module] if node.module else []))
                else:
                    mod = node.module or ""
                names = [mod] + [f"{mod}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                continue
            assert not any(n.startswith(banned) for n in names), f"{f.name} imports {names}"


def _adapter(r=2, d=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"m0": {"A": torch.randn(r, d, generator=g), "B": torch.randn(d, r, generator=g)},
            "m1": {"A": torch.randn(r, d, generator=g), "B": torch.randn(d, r, generator=g)}}


def test_client_update_rejects_text_and_labels():
    ok = dict(client_id=0, round=1, adapter=_adapter(), head={"w": torch.zeros(2)}, n_examples=5,
              local_loss=0.1, train_time_s=0.0)
    ClientUpdate(**ok)
    with pytest.raises(BoundaryViolation):
        ClientUpdate(**{**ok, "summary_y": ("some raw text",)})
    with pytest.raises(BoundaryViolation):
        ClientUpdate(**{**ok, "head": {"labels": torch.tensor([0, 1, 1])}})     # integer per-example labels
    with pytest.raises(TypeError):
        ClientUpdate(**ok, texts=["hello"])                                     # no such field
    with pytest.raises(BoundaryViolation):
        GlobalBroadcast(round=1, adapter=_adapter(), head={}, hparams={"note": "x"}, client_seed=1)
    u = ClientUpdate(**ok)
    with pytest.raises(Exception):
        u.adapter["m0"] = None                                                  # immutable


# --------------------------------------------------------------------------- linear algebra (Alg. 4, 9)
def test_trace_identity_matches_explicit():
    g = _adapter(seed=99)
    ads = [_adapter(seed=s) for s in range(4)]
    S = linalg.update_similarity(ads, g)
    for i in range(4):
        for j in range(4):
            ip = linalg.explicit_inner(ads[i], ads[j], g)
            ni, nj = linalg.explicit_inner(ads[i], ads[i], g), linalg.explicit_inner(ads[j], ads[j], g)
            assert abs(S[i, j] - ip / np.sqrt(ni * nj)) < 1e-5


def test_product_merge_exact_when_rank_fits():
    base = _adapter(r=2, seed=1)
    ads = [base, {m: {"A": 2 * ab["A"], "B": ab["B"]} for m, ab in base.items()}]   # same column space
    w = np.array([0.3, 0.7])
    out, _, eps = linalg.merge(ads, [{}, {}], w, "product")
    for m in base:
        target = sum(wi * a[m]["B"].double() @ a[m]["A"].double() for wi, a in zip(w, ads))
        assert torch.allclose(out[m]["B"].double() @ out[m]["A"].double(), target, atol=1e-5)
    assert eps < 1e-6


def test_average_merge_is_factorwise():
    ads = [_adapter(seed=s) for s in range(3)]
    out, head, eps = linalg.merge(ads, [{"h": torch.ones(2) * i} for i in range(3)], np.array([0.2, 0.3, 0.5]))
    assert eps is None and torch.allclose(head["h"], torch.full((2,), 1.3))
    assert torch.allclose(out["m0"]["A"], 0.2 * ads[0]["m0"]["A"] + 0.3 * ads[1]["m0"]["A"] + 0.5 * ads[2]["m0"]["A"], atol=1e-6)


# --------------------------------------------------------------------------- weights (Alg. 5-8)
def test_cap_bounds_and_normalises():
    w = cap(np.array([10, 1, 1, 1.0]), 0.4)
    assert abs(w.sum() - 1) < 1e-9 and w.max() <= 0.4 + 1e-9 and (w >= 0).all()


def _summaries(m, Q=8, C=4, seed=0):
    g = np.random.default_rng(seed)
    hc = g.dirichlet(np.ones(Q), m)
    hy = g.dirichlet(np.ones(C), m)
    return hc, hy


def test_diversity_lambda_zero_is_fedavg_and_weights_valid():
    n = np.array([100, 50, 30, 20.0])
    hc, hy = _summaries(4)
    w0, _ = diversity_weights(n, hc, hy, 8, 4, 0.0, 0.5, 1e-3, 0.4)
    assert np.allclose(w0, n / n.sum())
    w, s = diversity_weights(n, hc, hy, 8, 4, 0.5, 0.5, 1e-3, 0.4)
    assert abs(w.sum() - 1) < 1e-9 and w.max() <= 0.4 + 1e-9 and (s > 0).all()


def test_diversity_favours_unique_content():
    n = np.ones(4) * 50
    hy = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1.0]])
    w, _ = diversity_weights(n, hy, hy, 4, 4, 0.5, 0.5, 1e-3, 1.0)
    assert w[3] > w[0]


def _ctx(m, r=3, S=None):
    hc, hy = _summaries(m)
    return RoundContext(round=r, client_ids=list(range(m)), n=np.linspace(20, 60, m), summaries_c=list(hc),
                        summaries_y=list(hy), similarity=lambda: S, num_classes=4, codebook_size=8)


def test_clustered_and_hybrid_recover_two_groups():
    S = np.full((6, 6), -0.5)
    S[:3, :3] = 0.95; S[3:, 3:] = 0.95
    np.fill_diagonal(S, 1)
    assert len(set(cluster_clients(S, 0.6, None))) == 2
    cfg = DEFAULTS["aggregation"]
    for name in ("clustered", "hybrid"):
        w, clusters = REGISTRY[name](cfg).compute_weights(_ctx(6, S=S))
        assert abs(w.sum() - 1) < 1e-9 and sorted(map(sorted, clusters)) == [[0, 1, 2], [3, 4, 5]]


def test_cluster_warmup_single_cluster():
    w, clusters = REGISTRY["clustered"](DEFAULTS["aggregation"]).compute_weights(_ctx(5, r=1, S=np.eye(5)))
    assert clusters == [[0, 1, 2, 3, 4]]


def test_cluster_label_alignment():
    assert ClusterTracker.align([[3, 4], [0, 1, 2]], [[0, 1], [3, 4, 5]]) == [[0, 1, 2], [3, 4]]


# --------------------------------------------------------------------------- clustering rules and degeneracy
def test_clustered_volume_rule_is_exactly_fedavg_and_warns():
    S = np.full((6, 6), -0.5)
    S[:3, :3] = 0.95; S[3:, 3:] = 0.95
    np.fill_diagonal(S, 1)
    cfg = copy.deepcopy(DEFAULTS["aggregation"])
    ctx = _ctx(6, S=S)
    fed, _ = REGISTRY["fedavg"](cfg).compute_weights(ctx)
    cfg["clustered"]["rule"] = "volume"
    vol, _ = REGISTRY["clustered"](cfg).compute_weights(ctx)
    assert np.allclose(vol, fed)                                       # the documented no-op
    for rule in ("equal", "sqrt"):
        cfg["clustered"]["rule"] = rule
        w, _ = REGISTRY["clustered"](cfg).compute_weights(ctx)
        assert abs(w.sum() - 1) < 1e-9 and not np.allclose(w, fed)
    with pytest.warns(ConfigWarning, match="FedAvg"):
        load_config(None, ["experiment.strategy=[clustered]", "aggregation.clustered.rule=volume"])
    assert DEFAULTS["aggregation"]["clustered"]["rule"] != "volume"


def _near_orthogonal(m, seed=0, spread=0.01):
    g = np.random.default_rng(seed)
    S = g.normal(0, spread, (m, m))
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1)
    return S


def test_absolute_tau_with_near_zero_cosines_falls_back_and_is_flagged():
    cc = {**DEFAULTS["aggregation"]["clustered"], "tau_mode": "absolute", "tau": 0.6, "min_spread": 0.0}
    S = _near_orthogonal(6)
    assert len(set(cluster_clients(S, 0.6, None))) == 6                # raw cut: every client a singleton
    info = {}
    groups, _ = clustered_groups(_ctx(6, S=S), cc, info)
    assert len(groups) == 1 and info["cluster_status"] == "degenerate_all_singletons"
    assert abs(info["similarity"]["mean"]) < 0.05


def test_quantile_cut_finds_structure_at_small_cosine_scale():
    S = _near_orthogonal(6, spread=0.002)
    S[:3, :3] += 0.08; S[3:, 3:] += 0.08                               # weak but real two-group structure
    np.fill_diagonal(S, 1)
    cc = {**DEFAULTS["aggregation"]["clustered"], "tau_mode": "quantile", "tau_quantile": 0.3}
    info = {}
    groups, cut = clustered_groups(_ctx(6, S=S), cc, info)
    assert sorted(map(sorted, (g.tolist() for g in groups))) == [[0, 1, 2], [3, 4, 5]]
    assert info["cluster_status"] == "ok" and cut < 1.0
    info = {}
    groups, _ = clustered_groups(_ctx(6, S=np.eye(6)), cc, info)           # no spread at all
    assert len(groups) == 1 and info["cluster_status"] == "degenerate_no_spread"


def test_hybrid_damping_never_silently_uniform():
    cfg = copy.deepcopy(DEFAULTS["aggregation"])
    cfg["clustered"].update(num_clusters=3)                            # force 3 singleton-ish groups
    h = REGISTRY["hybrid"](cfg)
    S = _near_orthogonal(3)
    h.compute_weights(_ctx(3, S=S))
    assert h.round_info["damping"].startswith("inert")                 # all flagged: reported, not hidden
    S = np.full((6, 6), -0.5); S[:4, :4] = 0.95; np.fill_diagonal(S, 1)  # one cohesive group + 2 singletons
    cfg["clustered"].update(num_clusters=None, tau_mode="absolute", tau=0.6)
    h = REGISTRY["hybrid"](cfg)
    h.compute_weights(_ctx(6, S=S))
    assert h.round_info["damping"] == "applied" and len(h.round_info["damped_clusters"]) == 2


# --------------------------------------------------------------------------- run identity / overrides / tuning config
def test_run_id_tracks_semantic_config_only():
    from fedtune.experiment import run_id
    base = load_config(None, ["experiment.strategy=[fedavg,diversity]"])
    rid = lambda c, s="fedavg": run_id(c, s)
    lr = copy.deepcopy(base); lr["train"]["lr"] = 1e-3
    kap = copy.deepcopy(base); kap["aggregation"]["diversity"]["kappa"] = 0.9
    out = copy.deepcopy(base); out["experiment"]["output_dir"] = "elsewhere"; out["experiment"]["checkpoint_every"] = 7
    assert rid(lr) != rid(base)                                        # changed lr -> new run directory
    assert rid(kap, "diversity") != rid(base, "diversity")             # changed kappa re-runs diversity ...
    assert rid(kap) == rid(base)                                       # ... but not the FedAvg baseline
    assert rid(out) == rid(base)                                       # where / how often to checkpoint: same run
    so = copy.deepcopy(base); so["strategy_overrides"] = {"diversity": {"train.lr": 1e-3}}
    assert rid(so, "diversity") == rid(lr, "diversity") and rid(so) == rid(base)


def test_strategy_overrides_and_tune_validation():
    with pytest.raises(ConfigError, match="cannot be set per strategy"):
        load_config(None, ['strategy_overrides={fedavg: {dropout.rate: 0.5}}'])
    with pytest.raises(ConfigError, match="out of range"):
        load_config(None, ['strategy_overrides={fedavg: {train.lr: 5.0}}'])
    with pytest.raises(ConfigError, match="separate seed"):
        load_config(None, ["grid.seeds=[1,2,3]", "tune.seed=2"])
    with pytest.raises(ConfigError, match="non-empty list"):
        load_config(None, ['tune.space={fedavg: {train.lr: 0.1}}'])
    cfg = load_config(None, ['strategy_overrides={fedavg: {train.lr: 1e-3}}'])
    assert cfg["strategy_overrides"]["fedavg"]["train.lr"] == 1e-3


def test_tuning_trials_equal_budget_and_deterministic():
    from fedtune.experiment import tuning_trials
    space = {"fedavg": {"train.lr": [1e-4, 2e-4, 5e-4]},
             "diversity": {"train.lr": [1e-4, 2e-4, 5e-4], "aggregation.diversity.lam": [0.25, 0.5, 0.75]}}
    cfg = load_config(None, ["experiment.strategy=[fedavg,diversity]", f"tune.space={space}", "tune.trials=6"])
    a, b = tuning_trials(cfg, "diversity"), tuning_trials(cfg, "diversity")
    assert len(a) == 6 and a == b and len({str(sorted(x.items())) for x in a}) == 6
    assert len(tuning_trials(cfg, "fedavg")) == 3                      # full product when it fits the budget


# --------------------------------------------------------------------------- paired statistics
def test_paired_stats_match_scipy_ttest_rel():
    from scipy import stats
    from fedtune.experiment import paired_stats
    a, b = np.array([0.81, 0.84, 0.80, 0.83]), np.array([0.80, 0.82, 0.80, 0.81])
    out = paired_stats((a - b).tolist())
    ref = stats.ttest_rel(a, b)
    assert abs(out["t"] - ref.statistic) < 1e-9 and abs(out["p_value"] - ref.pvalue) < 1e-9
    assert out["ci95_low"] < out["mean_diff"] < out["ci95_high"]
    assert paired_stats([0.003, -0.002, 0.004])["verdict"].startswith("inconclusive")
