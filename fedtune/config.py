"""ConfigLoader & Validator (FR 1-2, UC1).

Parses YAML, applies ``--set key.path=value`` CLI overrides, validates every
leaf against a typed schema with ranges, and writes the resolved config with a
content hash so a run directory is self-describing (FR 3, NFR 5).
"""
from __future__ import annotations

import copy
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised with the offending key and allowed range (FR 2)."""


class ConfigWarning(UserWarning):
    """A valid configuration that probably does not do what was intended."""


# ---------------------------------------------------------------------------
# Defaults: every key the system understands. Unknown keys are rejected.
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    "experiment": {
        "name": "run",
        "seed": 42,
        "rounds": 20,
        "num_clients": 10,
        "strategy": ["fedavg"],
        "output_dir": "runs",
        "checkpoint_every": 5,
        "eval_every": 1,
    },
    "data": {
        "task": "classification",
        "dataset": "ag_news",            # ag_news | mnli | xnli (bundled in data/) | path to .csv/.jsonl
        "languages": None,               # XNLI only: subset of the 15 languages, e.g. [en, fr, hi]
        "text_fields": ["text"],         # one field, or two for sentence pairs
        "label_field": "label",
        "attribute_field": None,         # genre / language column for domain/language skew
        "max_examples": 6000,
        "max_length": 64,
        "test_frac": 0.10,
        "reference_frac": 0.02,
        "val_frac": 0.10,                # per-client local validation V_k
        "global_val_frac": 0.05,         # coordinator-held global validation set (model selection / tuning)
        "partition": {
            "mode": "dirichlet",         # dirichlet | domain | language | quantity | iid
            "alpha": 0.5,                # grid key: alpha, eta or beta depending on mode
            "min_size": 10,
            "max_tries": 20,
            "quantity_beta": None,       # combine quantity skew with any mode
        },
    },
    "model": {
        "base": "tiny-random",           # tiny-random (offline) or HF id <= ~1B params
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": None},
        "deterministic": False,
        "device": "auto",                # auto (MPS > CUDA > CPU) | cpu | cuda | mps
    },
    "train": {
        "local_epochs": 1,
        "max_steps": None,
        "lr": 2.0e-4,
        "batch_size": 16,
        "grad_clip": 1.0,
        "weight_decay": 0.0,
    },
    "dropout": {"rate": 0.2, "min_participants": 3, "stale_updates": False,
                "stale_max_age": 2, "stale_discount": 0.5},
    "aggregation": {
        "lora_method": "adapter_average",   # adapter_average | product
        "diversity": {"lam": 0.5, "kappa": 0.5, "eps": 1.0e-3, "w_max": 0.4,
                      "codebook_size": 32, "embed_dim": 64},
        # tau_mode: absolute -> cut the dendrogram at distance tau (1 - cosine);
        #           quantile -> cut at the tau_quantile of this round's pairwise distances (scale-free,
        #           robust when LoRA update cosines all sit near 0).
        # rule: equal (each cluster weight 1/|clusters|) | sqrt (omega_G ~ sqrt(N_G)) | volume (= FedAvg, ablation)
        "clustered": {"tau": 0.6, "tau_mode": "quantile", "tau_quantile": 0.25, "min_spread": 0.01,
                      "num_clusters": None, "warmup_rounds": 1, "rule": "equal"},
        "hybrid": {"rho": 0.5, "omega_max": 0.5},
    },
    "evaluation": {
        "test_every_round": False,       # False: test set T is touched once, after the last round
        "final_test": True,              # False during tuning: T is never evaluated
    },
    # per-strategy overrides of train.* / aggregation.* keys, e.g. {diversity: {train.lr: 1e-4}}
    # (written by `fedtune tune`; partitions and participation stay identical across strategies)
    "strategy_overrides": {},
    "grid": {"alpha": None, "dropout_rate": None, "seeds": None},
    "tune": {
        "seed": 1000,                    # tuning seed: must not be one of grid.seeds
        "trials": 6,                     # trials per strategy (same budget for every strategy)
        "alpha": None,                   # partition skew used for tuning (default: data.partition.alpha)
        "dropout_rate": None,            # default: dropout.rate
        "select_metric": "accuracy",     # on the global validation set, never on test
        "select_last": 3,                # score = mean over the last k completed rounds
        "space": {},                     # {strategy: {dotted.key: [values, ...]}}
    },
}

FREEFORM = {"strategy_overrides", "tune.space"}          # user-defined keys below these paths
OVERRIDABLE_PREFIXES = ("train.", "aggregation.")       # never data/dropout/experiment (FR 17)

STRATEGIES = {"fedavg", "diversity", "clustered", "hybrid"}

# key -> (type(s), lo, hi, allowed values)  (None = unbounded / no choice set)
SCHEMA: dict[str, tuple] = {
    "experiment.name": (str, None, None, None),
    "experiment.seed": (int, 0, 2**32 - 1, None),
    "experiment.rounds": (int, 1, 100, None),
    "experiment.num_clients": (int, 1, 50, None),
    "experiment.output_dir": (str, None, None, None),
    "experiment.checkpoint_every": (int, 1, 100, None),
    "experiment.eval_every": (int, 1, 100, None),
    "data.task": (str, None, None, {"classification"}),
    "data.dataset": (str, None, None, None),
    "data.label_field": (str, None, None, None),
    "data.max_examples": (int, 50, 10_000_000, None),
    "data.max_length": (int, 8, 1024, None),
    "data.test_frac": (float, 0.01, 0.5, None),
    "data.reference_frac": (float, 0.0, 0.2, None),
    "data.val_frac": (float, 0.0, 0.5, None),
    "data.global_val_frac": (float, 0.0, 0.5, None),
    "data.partition.mode": (str, None, None, {"dirichlet", "domain", "language", "quantity", "iid"}),
    "data.partition.alpha": (float, 0.0, 1000.0, None),
    "data.partition.min_size": (int, 1, 100_000, None),
    "data.partition.max_tries": (int, 1, 1000, None),
    "model.base": (str, None, None, None),
    "model.lora.rank": (int, 1, 256, None),
    "model.lora.alpha": (float, 0.0, 1024.0, None),
    "model.lora.dropout": (float, 0.0, 0.9, None),
    "model.deterministic": (bool, None, None, None),
    "model.device": (str, None, None, {"auto", "cpu", "cuda", "mps"}),
    "train.local_epochs": (int, 1, 50, None),
    "train.lr": (float, 1e-7, 1.0, None),
    "train.batch_size": (int, 1, 1024, None),
    "train.grad_clip": (float, 0.0, 1e6, None),
    "train.weight_decay": (float, 0.0, 1.0, None),
    "dropout.rate": (float, 0.0, 1.0, None),
    "dropout.min_participants": (int, 1, 50, None),
    "dropout.stale_updates": (bool, None, None, None),
    "dropout.stale_max_age": (int, 1, 100, None),
    "dropout.stale_discount": (float, 0.0, 1.0, None),
    "aggregation.lora_method": (str, None, None, {"adapter_average", "product"}),
    "aggregation.diversity.lam": (float, 0.0, 1.0, None),
    "aggregation.diversity.kappa": (float, 0.0, 1.0, None),
    "aggregation.diversity.eps": (float, 0.0, 1.0, None),
    "aggregation.diversity.w_max": (float, 0.0, 1.0, None),
    "aggregation.diversity.codebook_size": (int, 2, 1024, None),
    "aggregation.diversity.embed_dim": (int, 2, 4096, None),
    "aggregation.clustered.tau": (float, 0.0, 2.0, None),
    "aggregation.clustered.tau_mode": (str, None, None, {"absolute", "quantile"}),
    "aggregation.clustered.tau_quantile": (float, 0.0, 1.0, None),
    "aggregation.clustered.min_spread": (float, 0.0, 1.0, None),
    "aggregation.clustered.warmup_rounds": (int, 0, 100, None),
    "aggregation.clustered.rule": (str, None, None, {"volume", "equal", "sqrt"}),
    "aggregation.hybrid.rho": (float, 0.0, 1.0, None),
    "aggregation.hybrid.omega_max": (float, 0.0, 1.0, None),
    "evaluation.test_every_round": (bool, None, None, None),
    "evaluation.final_test": (bool, None, None, None),
    "tune.seed": (int, 0, 2**32 - 1, None),
    "tune.trials": (int, 1, 1000, None),
    "tune.select_metric": (str, None, None, {"accuracy", "macro_f1"}),
    "tune.select_last": (int, 1, 100, None),
}


def _deep_merge(base: dict, override: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        key = f"{path}{k}"
        if key in FREEFORM:
            if not isinstance(v, dict):
                raise ConfigError(f"'{key}' must be a mapping")
            out[k] = copy.deepcopy(v)
            continue
        if k not in out:
            raise ConfigError(f"unknown configuration key '{key}'")
        if isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v, key + ".")
        else:
            out[k] = v
    return out


def _get(cfg: dict, dotted: str) -> Any:
    cur = cfg
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def _set(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            raise ConfigError(f"unknown configuration key '{dotted}'")
        cur = cur[p]
    if parts[-1] not in cur:
        raise ConfigError(f"unknown configuration key '{dotted}'")
    cur[parts[-1]] = value


def parse_overrides(pairs: list[str] | None) -> list[tuple[str, Any]]:
    """``["dropout.rate=0.3", "experiment.strategy=[fedavg,clustered]"]``."""
    out = []
    for p in pairs or []:
        if "=" not in p:
            raise ConfigError(f"override '{p}' must look like key.path=value")
        k, v = p.split("=", 1)
        out.append((k.strip(), yaml.safe_load(v)))
    return out


def validate(cfg: dict) -> dict:
    for key, (typ, lo, hi, choices) in SCHEMA.items():
        val = _get(cfg, key)
        if typ is float and isinstance(val, str):        # YAML 1.1 reads "1e-2" as a string
            try:
                val = float(val)
                _set(cfg, key, val)
            except ValueError:
                pass
        if typ is float and isinstance(val, int) and not isinstance(val, bool):
            val = float(val)
            _set(cfg, key, val)
        if typ is int and isinstance(val, bool) or not isinstance(val, typ):
            raise ConfigError(f"'{key}' must be of type {typ.__name__}, got {val!r}")
        if choices is not None and val not in choices:
            raise ConfigError(f"'{key}'={val!r} not allowed; choose one of {sorted(choices)}")
        if lo is not None and not (lo <= val <= hi):
            raise ConfigError(f"'{key}'={val!r} out of range; allowed [{lo}, {hi}]")
    strat = cfg["experiment"]["strategy"]
    if isinstance(strat, str):
        strat = [strat]
        cfg["experiment"]["strategy"] = strat
    if not strat:
        raise ConfigError("'experiment.strategy' must list at least one strategy")
    from .coordinator.strategies import REGISTRY  # registered plug-ins count too (FR 18)
    for s in strat:
        if s not in REGISTRY:
            raise ConfigError(f"'experiment.strategy' entry {s!r} unknown; registered: {sorted(REGISTRY)}")
    if cfg["dropout"]["min_participants"] > cfg["experiment"]["num_clients"]:
        raise ConfigError("'dropout.min_participants' must be <= experiment.num_clients "
                          f"({cfg['experiment']['num_clients']})")
    tf = cfg["data"]["text_fields"]
    if not isinstance(tf, list) or not 1 <= len(tf) <= 2:
        raise ConfigError("'data.text_fields' must list one field or two (sentence pair)")
    for gk, lo, hi in (("alpha", 0.0, 1000.0), ("dropout_rate", 0.0, 1.0)):
        vals = cfg["grid"][gk]
        if vals is not None:
            if not isinstance(vals, list) or not all(isinstance(v, (int, float)) and lo <= v <= hi for v in vals):
                raise ConfigError(f"'grid.{gk}' must be a list of numbers in [{lo}, {hi}]")
    _validate_overrides(cfg, REGISTRY)
    _validate_tune(cfg, REGISTRY)
    if "clustered" in strat and cfg["aggregation"]["clustered"]["rule"] == "volume":
        warnings.warn("aggregation.clustered.rule=volume gives exactly the FedAvg weights "
                      "(omega_G * n_k/N_G = n_k/N); use 'equal' or 'sqrt' unless this is a deliberate ablation",
                      ConfigWarning, stacklevel=2)
    return cfg


def _check_override_key(where: str, key: str) -> None:
    if not isinstance(key, str) or not key.startswith(OVERRIDABLE_PREFIXES):
        raise ConfigError(f"'{where}': key {key!r} cannot be set per strategy; only "
                          f"{', '.join(p + '*' for p in OVERRIDABLE_PREFIXES)} keys (partitions and "
                          "participation must stay identical across strategies, FR 17)")
    if key not in SCHEMA:
        raise ConfigError(f"'{where}': unknown configuration key {key!r}")


def _validate_overrides(cfg: dict, registry) -> None:
    so = cfg["strategy_overrides"]
    if not isinstance(so, dict):
        raise ConfigError("'strategy_overrides' must map strategy -> {key: value}")
    for strat, kv in so.items():
        if strat not in registry:
            raise ConfigError(f"'strategy_overrides.{strat}': unknown strategy; registered: {sorted(registry)}")
        if not isinstance(kv, dict):
            raise ConfigError(f"'strategy_overrides.{strat}' must map dotted keys to values")
        for k in kv:
            _check_override_key(f"strategy_overrides.{strat}", k)
        trial = copy.deepcopy(cfg)
        trial["strategy_overrides"] = {}
        for k, v in kv.items():
            _set(trial, k, v)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConfigWarning)
                validate(trial)
        except ConfigError as e:
            raise ConfigError(f"'strategy_overrides.{strat}': {e}") from None
        for k in kv:                                   # keep the coerced value (YAML reads 1e-3 as a string)
            kv[k] = _get(trial, k)


def _validate_tune(cfg: dict, registry) -> None:
    t = cfg["tune"]
    for k, lo, hi in (("alpha", 0.0, 1000.0), ("dropout_rate", 0.0, 1.0)):
        v = t[k]
        if v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool) or not lo <= v <= hi):
            raise ConfigError(f"'tune.{k}' must be null or a number in [{lo}, {hi}]")
    if cfg["grid"]["seeds"] and t["seed"] in cfg["grid"]["seeds"]:
        raise ConfigError(f"'tune.seed'={t['seed']} is also a grid seed; tuning must use a separate seed")
    for strat, space in t["space"].items():
        if strat not in registry:
            raise ConfigError(f"'tune.space.{strat}': unknown strategy; registered: {sorted(registry)}")
        if not isinstance(space, dict):
            raise ConfigError(f"'tune.space.{strat}' must map dotted keys to lists of values")
        for k, vals in space.items():
            _check_override_key(f"tune.space.{strat}", k)
            if not isinstance(vals, list) or not vals:
                raise ConfigError(f"'tune.space.{strat}.{k}' must be a non-empty list of values")


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    user = {}
    if path:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
    cfg = _deep_merge(DEFAULTS, user)
    for k, v in parse_overrides(overrides):
        _set(cfg, k, v)
    return validate(cfg)


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# keys that never change what a run computes (where it is written, how often it checkpoints,
# which other cells a grid/tune sweep contains)
_NON_SEMANTIC = ("experiment.output_dir", "experiment.checkpoint_every", "experiment.name")


def effective_config(cfg: dict, strategy: str) -> dict:
    """The configuration one (strategy, seed, alpha, p) run actually uses: strategy_overrides applied."""
    c = copy.deepcopy(cfg)
    c["experiment"]["strategy"] = [strategy]
    for k, v in (c.get("strategy_overrides") or {}).get(strategy, {}).items():
        _set(c, k, v)
    c["strategy_overrides"] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConfigWarning)
        return validate(c)


def run_hash(eff: dict, strategy: str) -> str:
    """Hash of everything that can change a run's numbers.

    ``eff`` is an effective config. Grid/tune sweep definitions and non-semantic keys are
    dropped, and only the aggregation blocks the strategy reads are kept, so changing
    diversity.kappa re-runs diversity and hybrid but not the FedAvg baseline.
    """
    from .coordinator.strategies import REGISTRY
    c = copy.deepcopy(eff)
    for k in ("grid", "tune", "strategy_overrides"):
        c.pop(k, None)
    for dotted in _NON_SEMANTIC:
        head, last = dotted.rsplit(".", 1)
        _get(c, head).pop(last, None)
    sections = getattr(REGISTRY.get(strategy), "config_sections", None)
    if sections is not None:
        c["aggregation"] = {k: v for k, v in c["aggregation"].items() if k == "lora_method" or k in sections}
    c["_strategy"] = strategy
    return config_hash(c)


def dump_resolved(cfg: dict, path: Path) -> str:
    h = config_hash(cfg)
    with open(path, "w") as f:
        yaml.safe_dump({**cfg, "config_hash": h}, f, sort_keys=False)
    return h


def describe_options() -> str:
    """Text for ``--help``: every option, its default and allowed range."""
    lines = []
    for key, (typ, lo, hi, choices) in SCHEMA.items():
        rng = f"{sorted(choices)}" if choices else (f"[{lo}, {hi}]" if lo is not None else "any")
        lines.append(f"  {key:<42} default={_get(DEFAULTS, key)!r:<20} allowed={rng}")
    lines.append(f"  {'experiment.strategy':<42} default=['fedavg']           allowed=subset of {sorted(STRATEGIES)} + plug-ins")
    lines.append(f"  {'strategy_overrides':<42} default={{}}                   allowed={{strategy: {{train.*|aggregation.*: value}}}}")
    lines.append(f"  {'tune.alpha / tune.dropout_rate':<42} default=None                 allowed=null or number")
    lines.append(f"  {'tune.space':<42} default={{}}                   allowed={{strategy: {{train.*|aggregation.*: [values]}}}}")
    return "\n".join(lines)
