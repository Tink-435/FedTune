#!/usr/bin/env python3
"""Per-seed final test accuracy of every finished run, as small multiples (one panel per alpha).

The grid summary reports means and confidence intervals; this figure shows the individual seeds, so a
strategy that sometimes collapses (large seed-to-seed spread) is visible at a glance. Dots are single
runs (hover for the seed and exact value), the wide dash is the mean over seeds. Colours are the same
colour-blind-safe palette as the dashboard, and every strategy also has its own marker shape.

    python tools/seed_plot.py merged/runs --out results/figures/per_seed_accuracy.html
    python tools/seed_plot.py merged_xnli/runs --metric macro_f1 --out results/figures/xnli_f1.html

The HTML is interactive; its toolbar (camera icon) downloads a PNG. Reads results only.
Exit status: 0 = figure written, 2 = no finished runs found.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

STRATEGIES = ("fedavg", "diversity", "clustered", "hybrid")
COLOR = {"fedavg": "#0072B2", "diversity": "#E69F00", "clustered": "#009E73", "hybrid": "#D55E00"}
MARKER = {"fedavg": "circle", "diversity": "square", "clustered": "diamond", "hybrid": "triangle-up"}
INK, MUTED, GRID = "#1b2330", "#5b6576", "#e6e9ef"

# run folder name: <experiment>-<strategy>-a<alpha>-p<dropout>-s<seed>-<8 hex chars>
_RUN_NAME = re.compile(r"-(fedavg|diversity|clustered|hybrid)-a([0-9.]+)-p([0-9.]+)-s(\d+)-[0-9a-f]{8}$")


@dataclass(frozen=True)
class Point:
    strategy: str
    alpha: float
    dropout: float
    seed: int
    value: float
    run: str


def read_points(runs: Path, metric: str = "accuracy") -> list[Point]:
    """One Point per completed run whose final test metrics contain `metric`."""
    points = []
    for d in sorted(p for p in runs.iterdir() if p.is_dir()):
        m = _RUN_NAME.search(d.name)
        summary = d / "run_summary.json"
        if not m or not summary.exists():
            continue
        try:
            s = json.loads(summary.read_text())
        except json.JSONDecodeError:
            continue
        value = (s.get("final_test_metrics") or {}).get(metric)
        if s.get("status") != "completed" or value is None:
            continue
        points.append(Point(m.group(1), float(m.group(2)), float(m.group(3)), int(m.group(4)), float(value), d.name))
    return points


def build_figure(points: list[Point], metric: str = "accuracy") -> go.Figure:
    alphas = sorted({p.alpha for p in points})
    dropouts = sorted({p.dropout for p in points})
    present = [s for s in STRATEGIES if any(p.strategy == s for p in points)]
    step = 0.6 / max(len(present), 1)
    offset = {s: (i - (len(present) - 1) / 2) * step for i, s in enumerate(present)}   # side-by-side within a dropout level

    by_cell = defaultdict(list)                                  # (strategy, alpha, dropout) -> [Point]
    for p in points:
        by_cell[(p.strategy, p.alpha, p.dropout)].append(p)

    fig = make_subplots(rows=1, cols=len(alphas), shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=[f"alpha = {a:g}" for a in alphas])
    for col, a in enumerate(alphas, start=1):
        for s in present:
            xs, ys, tips, mx, my = [], [], [], [], []
            for j, dr in enumerate(dropouts):
                pts = by_cell.get((s, a, dr), [])
                for p in pts:
                    xs.append(j + offset[s]); ys.append(p.value)
                    tips.append(f"{s} | seed {p.seed} | dropout {dr:g}<br>{metric} = {p.value:.4f}")
                if pts:
                    mx.append(j + offset[s]); my.append(sum(q.value for q in pts) / len(pts))
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=s, legendgroup=s, showlegend=(col == 1),
                                     marker=dict(color=COLOR[s], symbol=MARKER[s], size=9, opacity=0.8,
                                                 line=dict(width=1, color="white")),
                                     text=tips, hoverinfo="text"), row=1, col=col)
            fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", legendgroup=s, showlegend=False,
                                     marker=dict(color=COLOR[s], symbol="line-ew", size=24, line=dict(width=3, color=COLOR[s])),
                                     hovertemplate=f"{s} mean = %{{y:.4f}}<extra></extra>"), row=1, col=col)
        fig.update_xaxes(tickmode="array", tickvals=list(range(len(dropouts))), ticktext=[f"{d:g}" for d in dropouts],
                         range=[-0.6, len(dropouts) - 0.4], title_text="dropout rate", row=1, col=col)
    fig.update_yaxes(title_text=f"final test {metric}", gridcolor=GRID, row=1, col=1)
    fig.update_layout(title=f"Per-seed final test {metric} (dots = seeds, dash = mean)", height=440,
                      plot_bgcolor="white", paper_bgcolor="white", font=dict(color=INK, size=13),
                      legend=dict(orientation="h", y=-0.22), margin=dict(t=80, b=90))
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", type=Path, help="folder holding the run directories (e.g. merged/runs)")
    ap.add_argument("--metric", default="accuracy", help="key in final_test_metrics (default: accuracy)")
    ap.add_argument("--out", type=Path, default=Path("per_seed.html"))
    a = ap.parse_args(argv)

    points = read_points(a.runs, a.metric)
    if not points:
        print(f"no completed runs with final test '{a.metric}' found in {a.runs}")
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    build_figure(points, a.metric).write_html(a.out, include_plotlyjs="cdn")
    counts = {s: sum(p.strategy == s for p in points) for s in STRATEGIES if any(p.strategy == s for p in points)}
    print(f"wrote {a.out} ({len(points)} runs: {counts})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
