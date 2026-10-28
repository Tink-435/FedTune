"""Command line (Section 4.6): validate, run, grid, resume, dashboard, export.

    fedtune validate -c configs/smoke.yaml
    fedtune run      -c configs/smoke.yaml --set dropout.rate=0.3
    fedtune tune     -c configs/tune.yaml [--grid]       (tune on a separate seed, then freeze)
    fedtune grid     -c tunes/tune/tuned.yaml [--shard 0/3]
    fedtune resume   -c configs/reference.yaml
    fedtune dashboard --runs runs [--port 8050]
    fedtune export   --runs runs --out exports
    fedtune run      -c configs/smoke.yaml --plugin examples/custom_strategy.py --set "experiment.strategy=[uniform]"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, describe_options, load_config


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fedtune", formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__, epilog="configuration keys (override with --set key=value):\n" + describe_options())
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("validate", "check a configuration and print the resolved form"),
                      ("run", "run every listed strategy on identical partitions and participation"),
                      ("resume", "resume runs from their latest checkpoint"),
                      ("grid", "run the heterogeneity x dropout x strategy x seed grid"),
                      ("tune", "tune every strategy with the same trial budget on the tuning seed (global validation)")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("-c", "--config", required=True)
        s.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override any key, e.g. dropout.rate=0.3")
        s.add_argument("--plugin", nargs="*", default=[], metavar="FILE.py",
                       help="import files that @register custom strategies (FR 18, UC11)")
        s.add_argument("--cpu-deterministic", action="store_true",
                       help="run on CPU with deterministic PyTorch kernels (tightest FR 3 reproducibility)")
        if name == "grid":
            s.add_argument("--shard", default=None, help="i/n: run every n-th cell starting at i (split across machines)")
        if name == "tune":
            s.add_argument("--grid", action="store_true", help="after tuning, run the grid with the frozen tuned.yaml")
    d = sub.add_parser("dashboard", help="read-only dashboard on 127.0.0.1")
    d.add_argument("--runs", default="runs"); d.add_argument("--port", type=int, default=8050)
    e = sub.add_parser("export", help="export metrics (CSV/JSON), configs and plots")
    e.add_argument("--runs", default="runs"); e.add_argument("--out", default="exports")
    e.add_argument("--run", default=None, help="single run id (default: all)")
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    for f in getattr(args, "plugin", []):
        load_plugin(f)
    try:
        if args.cmd in ("validate", "run", "resume", "grid", "tune"):
            cfg = load_config(args.config, args.set + (["model.deterministic=true", "model.device=cpu"]
                                                        if getattr(args, "cpu_deterministic", False) else []))
        if args.cmd == "validate":
            print(json.dumps(cfg, indent=2))
            print("configuration OK", file=sys.stderr)
            return 0
    except ConfigError as e:
        print(f"invalid configuration: {e}", file=sys.stderr)
        return 2

    from . import experiment
    if args.cmd in ("run", "resume"):
        rows = experiment.compare(cfg, resume=args.cmd == "resume")
        _print_table(rows)
    elif args.cmd == "grid":
        out = experiment.run_grid(cfg, args.shard)
        print(f"grid summary written to {out / 'grid_summary.csv'}")
        print(f"paired comparison vs FedAvg written to {out / 'paired_vs_fedavg.csv'}")
    elif args.cmd == "tune":
        out = experiment.run_tuning(cfg)
        print(f"tuning results in {out}; frozen config: {out / 'tuned.yaml'}")
        if args.grid:
            tuned = load_config(out / "tuned.yaml")
            g = experiment.run_grid(tuned)
            print(f"grid summary written to {g / 'grid_summary.csv'}")
    elif args.cmd == "dashboard":
        from .dashboard.app import serve
        serve(Path(args.runs), args.port)
    elif args.cmd == "export":
        runs = [Path(args.runs) / args.run] if args.run else sorted(p for p in Path(args.runs).iterdir() if (p / "rounds.jsonl").exists())
        for r in runs:
            print("exported", experiment.export_run(r, Path(args.out)))
    return 0


def load_plugin(path: str) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'strategy':<11}{'val acc':>8}{'test acc':>9}{'macro-F1':>10}{'worst':>8}{'skipped':>9}{'MB up':>9}{'up/model':>10}{'time s':>8}{'peak MB':>9}")
    for s in rows:
        g, d = s.get("final_test_metrics") or {}, s["client_dispersion"]
        v = s.get("final_val_metrics") or {}
        frac = s["adapter_bytes"] / s["full_model_bytes"]
        worst = d["worst_client_accuracy"]
        print(f"{s['strategy']:<11}{v.get('accuracy', float('nan')):>8.3f}{g.get('accuracy', float('nan')):>9.3f}{g.get('macro_f1', float('nan')):>10.3f}"
              f"{(worst if worst is not None else float('nan')):>8.3f}{s['rounds_skipped']:>9}"
              f"{s['cumulative_bytes']['up'] / 2**20:>9.2f}{frac:>10.2%}{s['total_time_s']:>8.1f}{s['peak_memory_mb']:>9.0f}")


if __name__ == "__main__":
    raise SystemExit(main())
