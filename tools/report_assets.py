#!/usr/bin/env python3
"""Build the report tables (CSV + Markdown) and figures (PNG) from a folder of finished runs.

    python tools/report_assets.py merged/runs --out results/report
    python tools/report_assets.py merged_xnli/runs --out results/report_xnli --collapse-below 0.5

Reads results only (run_summary.json, rounds.jsonl, config.resolved.yaml); runs nothing.

Tables (tables/*.csv and tables/*.md)
  accuracy_by_cell         mean +- SD of final test accuracy per alpha x dropout x strategy
  paired_vs_fedavg         paired differences to FedAvg (same alpha, dropout, seed): mean, 95% t-interval,
                           wins/losses, exact sign test, Wilcoxon test; for all pairs and for the most skewed alpha
  stability                final accuracy, mean validation accuracy of the last 3 rounds, round-to-round
                           oscillation (SD of validation accuracy over rounds 6..R), largest one-round drop
  collapse_counts          runs whose final test accuracy is below --collapse-below, per alpha x strategy
  settings_used            learning rate and strategy hyper-parameters as actually run
  run_cost                 wall-clock minutes per run and upload bytes relative to FedAvg

Figures (figures/*.png): accuracy vs dropout, learning curves in the most skewed setting, paired
differences with intervals, and oscillation. Colour-blind-safe palette; each strategy has its own marker.

Caveats are part of the output, not footnotes: intervals use few seeds, the subset analysis by alpha is
post hoc, and the oscillation / last-3-round metrics are secondary to the final test accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                                   # no display needed
import matplotlib.pyplot as plt                         # noqa: E402
import numpy as np                                      # noqa: E402
import yaml                                             # noqa: E402
from scipy import stats                                 # noqa: E402

STRATEGIES = ("fedavg", "diversity", "clustered", "hybrid")
COLOR = {"fedavg": "#0072B2", "diversity": "#E69F00", "clustered": "#009E73", "hybrid": "#D55E00"}
MARKER = {"fedavg": "o", "diversity": "s", "clustered": "D", "hybrid": "^"}
OSC_FROM_ROUND = 6          # oscillation is measured from round 6 on, after the initial climb


@dataclass
class Run:
    strategy: str
    alpha: float
    dropout: float
    seed: int
    accuracy: float
    macro_f1: float
    worst_client: float
    minutes: float
    upload_bytes: float
    val: np.ndarray                                     # global validation accuracy per round (skipped rounds carried forward)
    settings: dict = field(default_factory=dict)

    @property
    def last3(self) -> float:
        return float(np.nanmean(self.val[-3:]))

    @property
    def oscillation(self) -> float:
        return float(np.nanstd(self.val[OSC_FROM_ROUND - 1:]))

    @property
    def largest_drop(self) -> float:
        return float(np.nanmax(-np.diff(self.val[OSC_FROM_ROUND - 1:])))


# ------------------------------------------------------------------------------------------ loading
def load_runs(runs_dir: Path) -> list[Run]:
    runs = []
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        sf, rf = d / "run_summary.json", d / "rounds.jsonl"
        if not sf.exists() or not rf.exists():
            continue
        s = json.loads(sf.read_text())
        if s.get("status") != "completed":
            continue
        val = []
        for line in rf.read_text().splitlines():
            if line.strip():
                v = (json.loads(line).get("val_metrics") or {}).get("accuracy")
                val.append(np.nan if v is None else float(v))
        # a skipped round leaves the global model unchanged: carry the previous validation value forward
        for i in range(1, len(val)):
            if math.isnan(val[i]):
                val[i] = val[i - 1]
        cfg_file = d / "config.resolved.yaml"
        runs.append(Run(
            strategy=s["strategy"], alpha=float(s["alpha"]), dropout=float(s["dropout_rate"]), seed=int(s["seed"]),
            accuracy=float(s["final_test_metrics"]["accuracy"]), macro_f1=float(s["final_test_metrics"]["macro_f1"]),
            worst_client=float(s["client_dispersion"]["worst_client_accuracy"]),
            minutes=float(s["total_time_s"]) / 60, upload_bytes=float(s["cumulative_bytes"]["up"]),
            val=np.array(val), settings=yaml.safe_load(cfg_file.read_text()) if cfg_file.exists() else {}))
    return runs


# ------------------------------------------------------------------------------------------ output helpers
def write_table(out: Path, name: str, header: list[str], rows: list[list], note: str = "") -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    lines = [f"| {' | '.join(header)} |", f"|{'|'.join('---' for _ in header)}|"]
    lines += [f"| {' | '.join(str(c) for c in r)} |" for r in rows]
    (out / f"{name}.md").write_text("\n".join(lines) + "\n" + (f"\n{note}\n" if note else ""))


def present(runs: list[Run]) -> list[str]:
    return [s for s in STRATEGIES if any(r.strategy == s for r in runs)]


def cell(runs: list[Run], strategy: str, alpha: float, dropout: float) -> list[Run]:
    return [r for r in runs if (r.strategy, r.alpha, r.dropout) == (strategy, alpha, dropout)]


# ------------------------------------------------------------------------------------------ tables
def accuracy_by_cell(runs, out):
    rows = []
    for a in sorted({r.alpha for r in runs}):
        for p in sorted({r.dropout for r in runs}):
            row = [f"{a:g}", f"{p:g}"]
            for s in present(runs):
                v = [r.accuracy for r in cell(runs, s, a, p)]
                row.append(f"{np.mean(v):.3f} +- {np.std(v, ddof=1):.3f}" if len(v) > 1 else (f"{v[0]:.3f}" if v else "-"))
            rows.append(row)
    write_table(out, "accuracy_by_cell", ["alpha", "dropout"] + present(runs), rows,
                "Final test accuracy, mean +- SD over seeds.")


def paired_diff(runs, strategy, metric, pairs) -> dict | None:
    base = {(r.alpha, r.dropout, r.seed): getattr(r, metric) for r in runs if r.strategy == "fedavg"}
    diffs = [getattr(r, metric) - base[(r.alpha, r.dropout, r.seed)]
             for r in runs if r.strategy == strategy and (r.alpha, r.dropout, r.seed) in base and (r.alpha, r.dropout, r.seed) in pairs]
    n = len(diffs)
    if n < 3:
        return None
    d = np.array(diffs)
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    w, l = int((d > 0).sum()), int((d < 0).sum())
    return {"n": n, "mean": m, "lo": m - half, "hi": m + half, "wins": w, "losses": l,
            "sign_p": stats.binomtest(w, w + l, 0.5).pvalue if w + l else float("nan"),
            "wilcoxon_p": stats.wilcoxon(d).pvalue if (d != 0).any() else float("nan")}


def subsets(runs) -> dict[str, set]:
    keys = {(r.alpha, r.dropout, r.seed) for r in runs}
    alphas = sorted({k[0] for k in keys})
    out = {"all pairs": keys}
    if len(alphas) > 1:
        out[f"alpha = {alphas[0]:g} (most skewed; post hoc)"] = {k for k in keys if k[0] == alphas[0]}
        out[f"alpha > {alphas[0]:g}"] = {k for k in keys if k[0] > alphas[0]}
    return out


PAIRED_METRICS = (("accuracy", "final test accuracy"), ("macro_f1", "final test macro-F1"),
                  ("worst_client", "worst-client accuracy"), ("last3", "validation accuracy, last 3 rounds"),
                  ("oscillation", "oscillation (negative = more stable)"))


def paired_table(runs, out):
    rows, results = [], {}
    for sub_name, pairs in subsets(runs).items():
        for metric, label in PAIRED_METRICS:
            for s in present(runs):
                if s == "fedavg":
                    continue
                r = paired_diff(runs, s, metric, pairs)
                if r is None:
                    continue
                results[(sub_name, metric, s)] = r
                rows.append([label, sub_name, s, r["n"], f"{100 * r['mean']:+.2f}", f"{100 * r['lo']:+.2f}", f"{100 * r['hi']:+.2f}",
                             f"{r['wins']}/{r['losses']}", f"{r['sign_p']:.3f}", f"{r['wilcoxon_p']:.3f}"])
    write_table(out, "paired_vs_fedavg",
                ["metric", "subset", "strategy", "pairs", "diff (pp)", "CI95 low", "CI95 high", "wins/losses", "sign p", "Wilcoxon p"], rows,
                "Differences to FedAvg in percentage points, paired by alpha, dropout and seed. CI95 is a t-interval over the pairs; "
                "pairs from the same seed are not independent, so intervals are optimistic. Many comparisons are shown: "
                "treat p-values as descriptive. Subsets by alpha were chosen after seeing the data.")
    return results


def stability_table(runs, out):
    rows = []
    for a in sorted({r.alpha for r in runs}):
        for s in present(runs):
            rs = [r for r in runs if r.strategy == s and r.alpha == a]
            if rs:
                rows.append([f"{a:g}", s, len(rs), f"{np.mean([r.accuracy for r in rs]):.3f}", f"{np.mean([r.last3 for r in rs]):.3f}",
                             f"{np.mean([r.oscillation for r in rs]):.3f}", f"{np.mean([r.largest_drop for r in rs]):.3f}"])
    write_table(out, "stability", ["alpha", "strategy", "runs", "final test acc", "val acc (last 3 rounds)",
                                   f"oscillation (SD of val acc, rounds {OSC_FROM_ROUND}+)", "largest one-round drop"], rows,
                "Means over all runs at that alpha. Oscillation is a secondary, post hoc metric.")


def collapse_table(runs, out, threshold):
    rows = []
    for a in sorted({r.alpha for r in runs}):
        row = [f"{a:g}"]
        for s in present(runs):
            rs = [r for r in runs if r.strategy == s and r.alpha == a]
            row.append(f"{sum(r.accuracy < threshold for r in rs)}/{len(rs)} (min {min(r.accuracy for r in rs):.3f})")
        rows.append(row)
    write_table(out, "collapse_counts", ["alpha"] + present(runs), rows,
                f"Runs with final test accuracy below {threshold:g} out of all runs at that alpha. The threshold is arbitrary.")
    low = sorted((r for r in runs if r.accuracy < threshold), key=lambda r: r.accuracy)
    write_table(out, "collapsed_runs", ["strategy", "alpha", "dropout", "seed", "final test acc", "worst-client acc"],
                [[r.strategy, f"{r.alpha:g}", f"{r.dropout:g}", r.seed, f"{r.accuracy:.4f}", f"{r.worst_client:.4f}"] for r in low])


def settings_table(runs, out):
    rows = []
    for s in present(runs):
        r = next(x for x in runs if x.strategy == s)
        cfg = r.settings
        if not cfg:
            continue
        agg = cfg.get("aggregation", {})
        info = {"lr": cfg.get("train", {}).get("lr")}
        if s in ("diversity", "hybrid"):
            info.update({k: agg["diversity"][k] for k in ("lam", "kappa", "w_max")})
        if s in ("clustered", "hybrid"):
            info.update({k: agg["clustered"][k] for k in ("tau_quantile", "rule")})
        if s == "hybrid":
            info.update({k: agg["hybrid"][k] for k in ("rho", "omega_max")})
        rows.append([s, ", ".join(f"{k}={v}" for k, v in info.items())])
    write_table(out, "settings_used", ["strategy", "settings as run"], rows,
                "Taken from config.resolved.yaml of one run per strategy. Learning rates differ across strategies because each was tuned separately.")


def cost_table(runs, out):
    base = np.mean([r.upload_bytes for r in runs if r.strategy == "fedavg"]) if any(r.strategy == "fedavg" for r in runs) else float("nan")
    rows = [[s, f"{np.mean([r.minutes for r in runs if r.strategy == s]):.1f}",
             f"{np.mean([r.upload_bytes for r in runs if r.strategy == s]) / 1e6:.1f}",
             f"{100 * (np.mean([r.upload_bytes for r in runs if r.strategy == s]) / base - 1):+.3f}%"] for s in present(runs)]
    write_table(out, "run_cost", ["strategy", "minutes per run", "upload MB per run", "upload vs FedAvg"], rows,
                "Upload bytes are averaged over all cells (dropout changes them), so compare strategies, not cells.")


# ------------------------------------------------------------------------------------------ figures
def style(ax):
    ax.grid(axis="y", color="#e6e9ef", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_accuracy(runs, out):
    alphas, drops = sorted({r.alpha for r in runs}), sorted({r.dropout for r in runs})
    fig, axes = plt.subplots(1, len(alphas), figsize=(4.2 * len(alphas), 3.8), sharey=True, squeeze=False)
    for ax, a in zip(axes[0], alphas):
        for i, s in enumerate(present(runs)):
            xs = np.arange(len(drops)) + (i - (len(present(runs)) - 1) / 2) * 0.09
            m = [np.mean([r.accuracy for r in cell(runs, s, a, p)]) for p in drops]
            sd = [np.std([r.accuracy for r in cell(runs, s, a, p)], ddof=1) for p in drops]
            ax.errorbar(xs, m, yerr=sd, color=COLOR[s], marker=MARKER[s], ms=6, lw=1.6, capsize=3, label=s)
        ax.set_xticks(range(len(drops)), [f"{p:g}" for p in drops])
        ax.set_title(f"alpha = {a:g}")
        ax.set_xlabel("dropout rate")
        style(ax)
    axes[0][0].set_ylabel("final test accuracy (mean +- SD over seeds)")
    axes[0][-1].legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "fig1_accuracy_vs_dropout.png", dpi=200)
    plt.close(fig)


def fig_curves(runs, out):
    a0, drops, strategies = min(r.alpha for r in runs), sorted({r.dropout for r in runs}), present(runs)
    def row_floor(p):                                              # lowest value in this dropout row, rounded down to 0.1
        return max(0.0, math.floor(min(float(np.nanmin(r.val)) for r in runs if r.alpha == a0 and r.dropout == p) * 10) / 10)
    fig, axes = plt.subplots(len(drops), len(strategies), figsize=(3.1 * len(strategies), 2.3 * len(drops)),
                             sharex=True, sharey="row", squeeze=False)
    for i, p in enumerate(drops):
        for j, s in enumerate(strategies):
            ax = axes[i][j]
            rs = cell(runs, s, a0, p)
            for r in rs:
                ax.plot(range(1, len(r.val) + 1), r.val, color=COLOR[s], lw=1, alpha=0.45)
            if rs:
                ax.plot(range(1, len(rs[0].val) + 1), np.mean([r.val for r in rs], axis=0), color=COLOR[s], lw=2.4)
            style(ax)
            ax.set_ylim(row_floor(p), 0.96)
            if i == 0:
                ax.set_title(s)
            if j == 0:
                ax.set_ylabel(f"dropout {p:g}\nvalidation acc.")
            if i == len(drops) - 1:
                ax.set_xlabel("round")
    fig.suptitle(f"Global validation accuracy per round at alpha = {a0:g} (thin = single seeds, bold = mean)", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig2_learning_curves_most_skewed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_paired(results, runs, out):
    names = list(subsets(runs))
    panels = [("accuracy", "final test accuracy"), ("worst_client", "worst-client accuracy"), ("oscillation", "oscillation (negative = more stable)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.6), squeeze=False)
    strategies = [s for s in present(runs) if s != "fedavg"]
    for ax, (metric, label) in zip(axes[0], panels):
        ys, ticks = [], []
        for si, sub in enumerate(names):
            for k, s in enumerate(strategies):
                r = results.get((sub, metric, s))
                if r is None:
                    continue
                y = si * (len(strategies) + 1) + k
                ax.errorbar(100 * r["mean"], y, xerr=[[100 * (r["mean"] - r["lo"])], [100 * (r["hi"] - r["mean"])]],
                            color=COLOR[s], marker=MARKER[s], ms=6, capsize=3, lw=1.6)
            ticks.append((si * (len(strategies) + 1) + (len(strategies) - 1) / 2, sub.split(" (")[0]))
        ax.axvline(0, color="#5b6576", lw=1, ls="--")
        ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks])
        ax.invert_yaxis()
        ax.set_xlabel("difference to FedAvg (percentage points)")
        ax.set_title(label, fontsize=10)
        ax.grid(axis="x", color="#e6e9ef", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [plt.Line2D([], [], color=COLOR[s], marker=MARKER[s], ls="", label=s) for s in strategies]
    fig.legend(handles=handles, loc="lower center", ncol=len(strategies), frameon=False)
    fig.text(0.99, 0.01, "Subsets by alpha were chosen after seeing the data (exploratory); bars are 95% t-intervals over pairs.",
             ha="right", va="bottom", fontsize=7.5, color="#5b6576")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "fig3_paired_differences.png", dpi=200)
    plt.close(fig)


def fig_oscillation(runs, out):
    alphas = sorted({r.alpha for r in runs})
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for s in present(runs):
        m = [np.mean([r.oscillation for r in runs if r.strategy == s and r.alpha == a]) for a in alphas]
        ax.plot(range(len(alphas)), m, color=COLOR[s], marker=MARKER[s], ms=7, lw=1.8, label=s)
    ax.set_xticks(range(len(alphas)), [f"{a:g}" for a in alphas])
    ax.set_xlabel("alpha (smaller = more skewed)")
    ax.set_ylabel(f"round-to-round oscillation\n(SD of validation acc., rounds {OSC_FROM_ROUND}+)")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig4_oscillation.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", type=Path, help="folder holding the run directories (e.g. merged/runs)")
    ap.add_argument("--out", type=Path, default=Path("results/report"))
    ap.add_argument("--collapse-below", type=float, default=0.80, help="final test accuracy below this counts as a collapse")
    a = ap.parse_args(argv)

    runs = load_runs(a.runs)
    if not runs:
        print(f"no completed runs found in {a.runs}")
        return 2
    tables, figures = a.out / "tables", a.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    accuracy_by_cell(runs, tables)
    results = paired_table(runs, tables)
    stability_table(runs, tables)
    collapse_table(runs, tables, a.collapse_below)
    settings_table(runs, tables)
    cost_table(runs, tables)
    fig_accuracy(runs, figures)
    fig_curves(runs, figures)
    fig_paired(results, runs, figures)
    fig_oscillation(runs, figures)
    print(f"{len(runs)} runs -> {len(list(tables.glob('*.csv')))} tables in {tables}, {len(list(figures.glob('*.png')))} figures in {figures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
