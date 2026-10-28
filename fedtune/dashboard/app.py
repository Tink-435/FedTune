"""Read-only diagnostic dashboard (FR 29-32, UC7, Section 2.1.7).

A local Plotly Dash app on 127.0.0.1 that polls run directories every 2 s
(FR 31: refresh within 5 s) and never writes to them or controls runs.
Series use the Okabe-Ito colour-blind-safe palette *and* distinct markers/dashes,
and every trace shows exact values on hover (FR 32).

Layout: a KPI strip for the selected run, then four views -
  Overview      global-validation curves (final test marked once), comparison table, upload
  Clients/data  per-client validation, participation matrix, partition make-up
  Aggregation   weights, cluster assignments, update-similarity diagnostics (degeneracy)
  Grid          heatmap (absolute or paired difference vs FedAvg) and a CI forest plot
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
MARKERS = ["circle", "square", "diamond", "triangle-up", "x", "star", "cross", "hexagon"]
DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot", "solid", "dash"]
STRATEGY_COLOR = {"fedavg": "#0072B2", "diversity": "#E69F00", "clustered": "#009E73", "hybrid": "#D55E00"}

INK, MUTED, GRID, CARD = "#1b2330", "#5b6576", "#e6e9ef", "#ffffff"
FONT = "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


# --------------------------------------------------------------------------- readers (no writes)
def list_runs(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if (p / "rounds.jsonl").exists()) if root.exists() else []


def read_rounds(run: Path) -> list[dict]:
    out = []
    p = run / "rounds.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break                          # partial line from a live run
    return out


def read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_csv(p: Path) -> list[dict]:
    try:
        with open(p, newline="") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def val_metrics(rec: dict) -> dict:
    """Per-round global validation metrics (older runs logged them as global_metrics)."""
    return rec.get("val_metrics") or rec.get("global_metrics") or {}


def short(run: str) -> str:
    """Drop the 8-char config hash for legends; it is shown in the KPI strip and table."""
    head, _, tail = run.rpartition("-")
    return head if len(tail) == 8 and all(c in "0123456789abcdef" for c in tail) else run


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- styling
def _style(i: int) -> dict:
    return {"line": {"color": PALETTE[i % 8], "dash": DASHES[i % 8], "width": 2},
            "marker": {"symbol": MARKERS[i % 8], "size": 7}}


def _layout(fig: go.Figure, title: str, x="round", y="", height=340) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=14, color=INK)),
        xaxis_title=x, yaxis_title=y, template="plotly_white", hovermode="closest",
        font=dict(family=FONT, size=12, color=INK), paper_bgcolor=CARD, plot_bgcolor=CARD,
        margin=dict(l=56, r=18, t=46, b=40), height=height,
        legend=dict(orientation="h", y=-0.22, font=dict(size=11)),
        hoverlabel=dict(font=dict(family=FONT)))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    if x == "round":
        fig.update_xaxes(tickmode="linear", dtick=1 if _n_rounds(fig) <= 25 else 5)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def _n_rounds(fig: go.Figure) -> int:
    xs = [v for t in fig.data for v in (t.x or []) if isinstance(v, (int, float))]
    return int(max(xs) - min(xs)) if xs else 0


def _empty(title: str, note: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=note, showarrow=False, font=dict(color=MUTED, size=13), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return _layout(fig, title)


# --------------------------------------------------------------------------- figures: overview
def fig_global(root: Path, runs: list[str], metric: str) -> go.Figure:
    fig = go.Figure()
    for i, r in enumerate(runs):
        recs = read_rounds(root / r)
        fig.add_trace(go.Scatter(x=[x["round"] for x in recs], y=[val_metrics(x).get(metric) for x in recs],
                                 mode="lines+markers", name=short(r), legendgroup=r, **_style(i),
                                 hovertemplate="round %{x}<br>val " + metric + " = %{y:.4f}<extra>" + short(r) + "</extra>"))
        ft = ((read_json(root / r / "run_summary.json") or {}).get("final_test_metrics") or {}).get(metric)
        if ft is not None and recs:
            fig.add_trace(go.Scatter(x=[recs[-1]["round"]], y=[ft], mode="markers", legendgroup=r, showlegend=False,
                                     marker=dict(symbol="star", size=15, color=PALETTE[i % 8], line=dict(width=1, color=INK)),
                                     hovertemplate="final TEST " + metric + " = %{y:.4f}<extra>" + short(r) + "</extra>"))
    if not runs:
        return _empty("Global validation per round", "select runs above")
    return _layout(fig, f"Global validation {metric} per round  ·  ★ = final test (evaluated once)", y=metric)


def fig_comm(root: Path, runs: list[str]) -> go.Figure:
    fig = go.Figure()
    for i, r in enumerate(runs):
        recs = read_rounds(root / r)
        fig.add_trace(go.Scatter(x=[x["round"] for x in recs], y=[x["cumulative_bytes"]["up"] / 2**20 for x in recs],
                                 mode="lines+markers", name=short(r), **_style(i),
                                 hovertemplate="round %{x}<br>%{y:.3f} MB uploaded<extra>" + short(r) + "</extra>"))
    return _layout(fig, "Cumulative upload", y="MB")


def comparison_rows(root: Path, runs: list[str]) -> list[dict]:
    rows = []
    for r in runs:
        s = read_json(root / r / "run_summary.json") or {}
        recs = read_rounds(root / r)
        v = s.get("final_val_metrics") or (val_metrics(recs[-1]) if recs else {})
        t = s.get("final_test_metrics") or {}
        d = s.get("client_dispersion") or {}
        rows.append({"run": short(r), "hash": s.get("config_hash", "")[:8], "status": s.get("status", "running"),
                     "val acc": v.get("accuracy"), "test acc": t.get("accuracy"), "test F1": t.get("macro_f1"),
                     "worst client": d.get("worst_client_accuracy"),
                     "MB up": (s.get("cumulative_bytes") or {}).get("up", 0) / 2**20 if s else None,
                     "skipped": s.get("rounds_skipped"), "time s": s.get("total_time_s")})
    return rows


# --------------------------------------------------------------------------- figures: clients / data
def fig_clients(root: Path, run: str, metric: str) -> go.Figure:
    fig = go.Figure()
    recs = [x for x in read_rounds(root / run) if x.get("client_metrics")]
    ks = sorted({int(k) for x in recs for k in x["client_metrics"]})
    for i, k in enumerate(ks):
        fig.add_trace(go.Scatter(x=[x["round"] for x in recs], y=[x["client_metrics"][str(k)][metric] for x in recs],
                                 mode="lines+markers", name=f"client {k}", **_style(i),
                                 hovertemplate=f"client {k}<br>round %{{x}}<br>{metric} = %{{y:.4f}}<extra></extra>"))
    return _layout(fig, f"Per-client local validation {metric}", y=metric)


def fig_participation(root: Path, run: str) -> go.Figure:
    """Client-by-round matrix; distinct symbols for present / dropped / skipped / excluded / stale."""
    recs = read_rounds(root / run)
    kinds = {"present": ("circle", "#009E73"), "dropped": ("x-thin-open", "#8a8f98"),
             "skipped round": ("square-open", "#D55E00"), "excluded (non-finite)": ("triangle-down", "#CC79A7"),
             "stale update": ("diamond-open", "#0072B2")}
    pts = {k: ([], []) for k in kinds}
    for x in recs:
        r = x["round"]
        excl = {e["client"] for e in x.get("excluded_clients", [])}
        for k in x["participating_clients"]:
            kind = "skipped round" if x["skipped"] else ("excluded (non-finite)" if k in excl else "present")
            pts[kind][0].append(r); pts[kind][1].append(k)
        for k in x["dropped_clients"]:
            kind = "stale update" if k in {s["client"] for s in x.get("stale_clients", [])} else "dropped"
            pts[kind][0].append(r); pts[kind][1].append(k)
    fig = go.Figure()
    for kind, (sym, col) in kinds.items():
        fig.add_trace(go.Scatter(x=pts[kind][0], y=pts[kind][1], mode="markers", name=kind,
                                 marker=dict(symbol=sym, color=col, size=10, line=dict(width=1.5, color=col)),
                                 hovertemplate="round %{x}, client %{y}: " + kind + "<extra></extra>"))
    fig.update_yaxes(dtick=1, autorange="reversed")
    return _layout(fig, "Participation matrix", y="client")


def fig_partition(root: Path, run: str) -> go.Figure:
    rep = read_json(root / run / "partition_report.json") or {"clients": [], "label_names": []}
    fig = go.Figure()
    names = rep["label_names"]
    for c, nm in enumerate(names):
        fig.add_trace(go.Bar(x=[f"c{x['client']}" for x in rep["clients"]], y=[x["label_hist"][c] for x in rep["clients"]],
                             name=nm, marker=dict(color=PALETTE[c % 8], pattern_shape=["", "/", "\\", "x", ".", "+", "-", "|"][c % 8]),
                             hovertemplate="%{x}<br>" + nm + ": %{y}<extra></extra>"))
    fig.update_layout(barmode="stack")
    sizes = f" · test {rep.get('test_size', '?')} · global val {rep.get('global_val_size', '?')} · reference {rep.get('reference_size', '?')}"
    return _layout(fig, f"Label make-up per client  ·  H = {rep.get('H', float('nan')):.3f}{sizes}", x="client", y="examples")


# --------------------------------------------------------------------------- figures: aggregation
def fig_weights(root: Path, run: str) -> go.Figure:
    recs = [x for x in read_rounds(root / run) if x.get("aggregation_weights")]
    ks = sorted({int(k) for x in recs for k in x["aggregation_weights"]})
    fig = go.Figure()
    for i, k in enumerate(ks):
        fig.add_trace(go.Scatter(x=[x["round"] for x in recs], y=[x["aggregation_weights"].get(str(k)) for x in recs],
                                 mode="markers+lines", name=f"client {k}", **_style(i),
                                 hovertemplate=f"client {k}<br>round %{{x}}<br>weight %{{y:.4f}}<extra></extra>"))
    resid = [(x["round"], x["merge_residual"]) for x in recs if x.get("merge_residual")]
    title = "Aggregation weights"
    if resid and max(v for _, v in resid) > 0.1:
        title += f"   ⚠ large merge_residual (max {max(v for _, v in resid):.3f})"
    return _layout(fig, title, y="weight")


def fig_clusters(root: Path, run: str) -> go.Figure:
    fig = go.Figure()
    xs, ys, cs = [], [], []
    for x in read_rounds(root / run):
        for ci, members in enumerate(x.get("clusters") or []):
            for k in members:
                xs.append(x["round"]); ys.append(k); cs.append(ci)
    for ci in sorted(set(cs)):
        idx = [i for i, c in enumerate(cs) if c == ci]
        fig.add_trace(go.Scatter(x=[xs[i] for i in idx], y=[ys[i] for i in idx], mode="markers", name=f"cluster {ci}",
                                 marker=dict(symbol=MARKERS[ci % 8], color=PALETTE[ci % 8], size=11),
                                 hovertemplate="round %{x}, client %{y}: cluster " + str(ci) + "<extra></extra>"))
    if not xs:
        return _empty("Cluster assignments", "no clusters logged (clustered / hybrid only)")
    fig.update_yaxes(dtick=1, autorange="reversed")
    return _layout(fig, "Cluster assignments (labels aligned across rounds)", y="client")


DEGENERATE = {"degenerate_all_singletons", "degenerate_no_spread"}


def fig_similarity(root: Path, run: str) -> go.Figure:
    """Off-diagonal update cosine per round (mean, min-max band), the cut used, and rounds where the
    clustering degenerated - the direct view of why clustered/hybrid can collapse to FedAvg."""
    recs = [x for x in read_rounds(root / run) if (x.get("strategy_info") or {}).get("similarity")]
    if not recs:
        return _empty("Update similarity", "no similarity logged (clustered / hybrid only)")
    r = [x["round"] for x in recs]
    st = [x["strategy_info"]["similarity"] for x in recs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r + r[::-1], y=[s["max"] for s in st] + [s["min"] for s in st][::-1], fill="toself",
                             fillcolor="rgba(0,114,178,0.12)", line=dict(width=0), name="min-max", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=r, y=[s["mean"] for s in st], mode="lines+markers", name="mean cosine",
                             line=dict(color="#0072B2", width=2), marker=dict(size=7),
                             customdata=[[s["std"], s["min"], s["max"]] for s in st],
                             hovertemplate="round %{x}<br>mean %{y:.4f} (sd %{customdata[0]:.4f})"
                                           "<br>min %{customdata[1]:.4f} · max %{customdata[2]:.4f}<extra></extra>"))
    cut = [(x["round"], 1 - x["strategy_info"]["cut_distance"]) for x in recs if "cut_distance" in x["strategy_info"]]
    if cut:
        fig.add_trace(go.Scatter(x=[c[0] for c in cut], y=[c[1] for c in cut], mode="lines", name="merge threshold (1 − cut)",
                                 line=dict(color="#D55E00", dash="dash"),
                                 hovertemplate="round %{x}<br>pairs above %{y:.4f} merge<extra></extra>"))
    deg = [x for x in recs if x["strategy_info"].get("cluster_status") in DEGENERATE]
    if deg:
        fig.add_trace(go.Scatter(x=[x["round"] for x in deg], y=[x["strategy_info"]["similarity"]["mean"] for x in deg],
                                 mode="markers", name="degenerate → one group",
                                 marker=dict(symbol="x", size=13, color="#CC79A7", line=dict(width=2)),
                                 text=[x["strategy_info"]["cluster_status"] for x in deg],
                                 hovertemplate="round %{x}: %{text}<extra></extra>"))
    n_deg = len(deg)
    return _layout(fig, f"Update cosine similarity  ·  degenerate rounds: {n_deg}/{len(recs)}", y="cosine")


def strategy_notes(root: Path, run: str) -> list[str]:
    recs = [x for x in read_rounds(root / run) if x.get("strategy_info")]
    if not recs:
        return []
    from collections import Counter
    status = Counter(x["strategy_info"].get("cluster_status", "?") for x in recs)
    damp = Counter(x["strategy_info"]["damping"] for x in recs if "damping" in x["strategy_info"])
    k = [x["strategy_info"].get("num_clusters") for x in recs if x["strategy_info"].get("num_clusters")]
    notes = ["cluster status: " + ", ".join(f"{s} ×{n}" for s, n in status.most_common())]
    if k:
        notes.append(f"clusters per round: mean {np.mean(k):.2f}, range {min(k)}–{max(k)}")
    if damp:
        notes.append("hybrid damping: " + ", ".join(f"{s} ×{n}" for s, n in damp.most_common()))
    return notes


# --------------------------------------------------------------------------- figures: grid
def grid_files(root: Path) -> list[Path]:
    g = root.parent / "grids"
    return sorted(g.glob("*/grid_summary.csv")) if g.exists() else []


def grid_strategies(path: str | None) -> list[str]:
    return sorted({r["strategy"] for r in read_csv(Path(path))}) if path else []


def fig_heatmap(path: str | None, strategy: str | None, metric: str, mode: str = "absolute") -> go.Figure:
    title_none = "Heterogeneity × dropout heatmap"
    if not path or not Path(path).exists():
        return _empty(title_none, "no grid summary yet: run `fedtune grid`")
    if mode == "paired":
        rows = [r for r in read_csv(Path(path).with_name("paired_vs_fedavg.csv"))
                if r["strategy"] == strategy and r["metric"] == metric and r["alpha"] != "all"]
        if not rows:
            return _empty(title_none, f"no paired comparison for {strategy or '—'} (needs FedAvg in the grid)")
    else:
        rows = [r for r in read_csv(Path(path)) if not strategy or r["strategy"] == strategy]
    alphas = sorted({float(r["alpha"]) for r in rows})
    rates = sorted({float(r["dropout_rate"]) for r in rows})
    Z = np.full((len(alphas), len(rates)), np.nan)
    T = [["" for _ in rates] for _ in alphas]
    for r in rows:
        i, j = alphas.index(float(r["alpha"])), rates.index(float(r["dropout_rate"]))
        if mode == "paired":
            m, lo, hi = _num(r["mean_diff"]), _num(r["ci95_low"]), _num(r["ci95_high"])
            if m is not None:
                Z[i, j] = m
                ci = f"[{lo:+.3f}, {hi:+.3f}]" if lo is not None else "(n<2)"
                mark = " ✓" if r["verdict"] in ("better", "worse") else ""
                T[i][j] = f"{m:+.3f}{mark}<br>{ci}"
        else:
            m, s = _num(r.get(f"{metric}_mean")), _num(r.get(f"{metric}_std"))
            if m is not None:
                Z[i, j] = m
                T[i][j] = f"{m:.3f}<br>± {s or 0:.3f}"
    if mode == "paired":
        lim = float(np.nanmax(np.abs(Z))) if np.isfinite(Z).any() else 1.0
        hm = go.Heatmap(z=Z, x=[str(x) for x in rates], y=[str(a) for a in alphas], text=T, texttemplate="%{text}",
                        colorscale="RdBu", zmid=0, zmin=-lim, zmax=lim, colorbar=dict(title="Δ"),
                        hovertemplate="alpha %{y}, p %{x}<br>Δ vs FedAvg %{text}<extra></extra>")
        title = f"{strategy} − FedAvg: paired Δ {metric} [95% CI]  ·  ✓ = CI excludes 0"
    else:
        hm = go.Heatmap(z=Z, x=[str(x) for x in rates], y=[str(a) for a in alphas], text=T, texttemplate="%{text}",
                        colorscale="Cividis", hovertemplate="alpha %{y}, p %{x}<br>%{text}<extra></extra>")
        title = f"{metric}: mean ± std over seeds ({strategy or 'all'})  ·  smaller α = more skew"
    fig = go.Figure(hm)
    return _layout(fig, title, x="dropout rate", y="alpha (heterogeneity)", height=380)


def fig_paired(path: str | None, metric: str) -> go.Figure:
    """Forest plot: per-cell and pooled paired differences vs FedAvg with 95% CIs."""
    rows = [r for r in read_csv(Path(path).with_name("paired_vs_fedavg.csv")) if r["metric"] == metric] if path else []
    if not rows:
        return _empty("Paired difference vs FedAvg", "no paired comparison yet (grid must include fedavg)")
    fig = go.Figure()
    strategies = sorted({r["strategy"] for r in rows})
    labels = []
    for r in rows:
        lab = "pooled (all cells)" if r["alpha"] == "all" else f"α={float(r['alpha']):g}, p={float(r['dropout_rate']):g}"
        if lab not in labels:
            labels.append(lab)
    for si, s in enumerate(strategies):
        mine = [r for r in rows if r["strategy"] == s]
        ys, xs, lo, hi, txt = [], [], [], [], []
        for r in mine:
            m = _num(r["mean_diff"])
            if m is None:
                continue
            lab = "pooled (all cells)" if r["alpha"] == "all" else f"α={float(r['alpha']):g}, p={float(r['dropout_rate']):g}"
            ys.append(lab); xs.append(m)
            lo.append(m - (_num(r["ci95_low"]) if _num(r["ci95_low"]) is not None else m))
            hi.append((_num(r["ci95_high"]) if _num(r["ci95_high"]) is not None else m) - m)
            txt.append(f"n={r['n_pairs']} · p={_num(r['p_value']) if _num(r['p_value']) is not None else float('nan'):.3f} · {r['verdict']}")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=s, text=txt,
                                 marker=dict(symbol=MARKERS[si % 8], size=10, color=STRATEGY_COLOR.get(s, PALETTE[si % 8])),
                                 error_x=dict(type="data", symmetric=False, array=hi, arrayminus=lo, thickness=1.5, width=4),
                                 hovertemplate="%{y}<br>Δ = %{x:+.4f}<br>%{text}<extra>" + s + "</extra>"))
    fig.add_vline(x=0, line=dict(color=MUTED, dash="dot"))
    fig.update_yaxes(categoryorder="array", categoryarray=labels[::-1])
    fig.update_layout(scattermode="group")
    return _layout(fig, f"Paired Δ {metric} vs FedAvg (same seed = same partition & participation) · 95% CI",
                   x=f"Δ {metric}", y="", height=max(340, 60 + 34 * len(labels)))


# --------------------------------------------------------------------------- app
CSS = """
:root { --ink:#1b2330; --muted:#5b6576; --line:#e6e9ef; --bg:#f4f6f9; --card:#ffffff; --accent:#0072B2; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:%s; }
.wrap { max-width:1440px; margin:0 auto; padding:18px 20px 40px; }
.top { display:flex; align-items:baseline; justify-content:space-between; gap:12px; flex-wrap:wrap; margin-bottom:14px; }
.brand { font-size:22px; font-weight:700; letter-spacing:-0.01em; }
.brand span { color:var(--accent); }
.sub { color:var(--muted); font-size:13px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px 14px; }
.controls { display:grid; grid-template-columns: 2.2fr 1.4fr 0.8fr; gap:12px; }
.lbl { font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin:0 0 4px 2px; }
.kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:12px 0; }
.kpi .k { font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
.kpi .v { font-size:22px; font-weight:700; margin-top:4px; font-variant-numeric: tabular-nums; }
.kpi .n { font-size:12px; color:var(--muted); margin-top:2px; }
.bar { height:6px; background:var(--line); border-radius:3px; margin-top:8px; overflow:hidden; }
.bar > div { height:100%%; background:var(--accent); }
.pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:13px; font-weight:600; }
.pill.completed { background:#e3f4ec; color:#0b6b45; } .pill.failed { background:#fbe6e1; color:#9c2b10; }
.pill.running { background:#e5f0fa; color:#0b4f80; }
.tabs { display:flex; gap:6px; margin:6px 0 12px; flex-wrap:wrap; }
.tabs label { padding:7px 14px; border:1px solid var(--line); border-radius:999px; background:var(--card);
              cursor:pointer; font-size:13px; font-weight:600; color:var(--muted); }
.tabs input { display:none; }
.tabs label:has(input:checked), .tabs label:has(input:checked) * { background:var(--ink); color:#fff !important; border-color:var(--ink); }
.g2 { display:grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap:12px; }
.g2 > .wide { grid-column: 1 / -1; }
.g2 > .card, .stack > .card { padding:6px; }
.stack { display:grid; grid-template-columns: minmax(0,1fr); gap:12px; }
.controls > div, .stack > *, .g2 > * { min-width:0; }
table.cmp { width:100%%; border-collapse:collapse; font-size:13px; font-variant-numeric: tabular-nums; }
table.cmp th { text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted);
               border-bottom:1px solid var(--line); padding:8px 8px; }
table.cmp td { border-bottom:1px solid var(--line); padding:7px 8px; }
table.cmp tr:last-child td { border-bottom:none; }
.best { font-weight:700; color:#0b6b45; }
.note { font-size:12.5px; color:var(--muted); line-height:1.5; }
.note b { color:var(--ink); }
ul.notes { margin:6px 0 0; padding-left:18px; font-size:13px; }
@media (max-width: 900px) { .controls, .g2 { grid-template-columns: 1fr; } }
""" % FONT


def _fmt(v, nd=3, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}" if not pct else f"{100 * v:.1f}%"


def build_app(root: Path):
    from dash import Dash, Input, Output, dcc, html, no_update

    app = Dash(__name__, title="FedTune dashboard")
    app.index_string = app.index_string.replace("{%css%}", "{%css%}<style>" + CSS + "</style>")
    runs = list_runs(root)
    grids = [str(p) for p in grid_files(root)]
    dd = lambda i, opts, val, multi=False: dcc.Dropdown(id=i, options=opts, value=val, multi=multi, clearable=not multi)
    card = lambda *c, **kw: html.Div(list(c), className="card", **kw)
    graph = lambda i, cls="card": html.Div(dcc.Graph(id=i, config={"displaylogo": False}), className=cls)

    app.layout = html.Div(className="wrap", children=[
        html.Div(className="top", children=[
            html.Div([html.Span("Fed", className="brand"), html.Span("Tune", className="brand", style={"color": "#0072B2"}),
                      html.Span("  diagnostics", className="sub")]),
            html.Div(f"read-only · {root.resolve()} · refreshes every 2 s", className="sub")]),
        card(html.Div(className="controls", children=[
            html.Div([html.Div("Compare runs", className="lbl"), dd("runs", runs, runs[:4], multi=True)]),
            html.Div([html.Div("Detail run", className="lbl"), dd("detail", runs, runs[0] if runs else None)]),
            html.Div([html.Div("Metric", className="lbl"), dd("metric", ["accuracy", "macro_f1"], "accuracy")])])),
        html.Div(id="kpis", className="kpis"),
        dcc.RadioItems(id="tab", className="tabs", value="overview", options=[
            {"label": "Overview", "value": "overview"}, {"label": "Clients & data", "value": "clients"},
            {"label": "Aggregation", "value": "agg"}, {"label": "Grid & significance", "value": "grid"}]),
        html.Div(id="sec-overview", className="stack", children=[
            graph("global"),
            card(html.Div(id="cmp"), style={"padding": "10px 14px"}),
            graph("comm")]),
        html.Div(id="sec-clients", className="g2", children=[graph("clients"), graph("participation"), graph("partition", "card wide")]),
        html.Div(id="sec-agg", className="stack", children=[
            card(html.Div(id="aggnotes"), style={"padding": "10px 14px"}),
            html.Div(className="g2", children=[graph("weights"), graph("clusters")]),
            graph("similarity")]),
        html.Div(id="sec-grid", className="stack", children=[
            card(html.Div(className="controls", style={"gridTemplateColumns": "2fr 1fr 1fr"}, children=[
                html.Div([html.Div("Grid", className="lbl"), dd("grid", grids, grids[0] if grids else None)]),
                html.Div([html.Div("Strategy", className="lbl"), dd("gstrat", grid_strategies(grids[0]) if grids else [], "fedavg")]),
                html.Div([html.Div("Heatmap shows", className="lbl"),
                          dd("gmode", [{"label": "mean ± std", "value": "absolute"},
                                       {"label": "Δ vs FedAvg (paired)", "value": "paired"}], "absolute")])])),
            html.Div(className="g2", children=[
                html.Div(className="stack", children=[graph("heatmap"), card(html.Div(id="cell"), style={"padding": "10px 14px"})]),
                graph("paired")]),
            card(html.Div(className="note", children=[
                html.B("Reading the paired view. "),
                "Each seed fixes the partition and the participation schedule, so a strategy and FedAvg on the same seed "
                "form a pair; the CI is a paired-t interval on those differences. With 3 seeds t₀.₉₇₅,₂ = 4.30, so a gap "
                "of ~0.003 is usually inside the noise. ✓ marks cells whose CI excludes 0. "
                "Training on MPS/CUDA is statistically, not bitwise, reproducible."]), style={"padding": "10px 14px"})]),
        dcc.Interval(id="tick", interval=2000),          # poll every 2 s (FR 31)
    ])

    @app.callback(Output("runs", "options"), Output("detail", "options"), Output("grid", "options"),
                  Input("tick", "n_intervals"))
    def refresh_lists(_):
        r = list_runs(root)
        return r, r, [str(p) for p in grid_files(root)]

    @app.callback(Output("gstrat", "options"), Input("grid", "value"))
    def refresh_strats(g):
        return grid_strategies(g)

    @app.callback(*[Output(f"sec-{s}", "style") for s in ("overview", "clients", "agg", "grid")], Input("tab", "value"))
    def show(tab):
        return [{} if tab == s else {"display": "none"} for s in ("overview", "clients", "agg", "grid")]

    @app.callback(Output("kpis", "children"), Input("tick", "n_intervals"), Input("detail", "value"))
    def kpis(_, detail):
        if not detail:
            return [card(html.Div("No runs yet", className="v"),
                         html.Div("start one with `fedtune run -c configs/smoke.yaml`", className="n"), className="card kpi")]
        import yaml
        s = read_json(root / detail / "run_summary.json") or {}
        recs = read_rounds(root / detail)
        rp = root / detail / "config.resolved.yaml"
        rc = yaml.safe_load(rp.read_text()) if rp.exists() else {}
        R = rc.get("experiment", {}).get("rounds") or 1
        r_now = recs[-1]["round"] if recs else 0
        status = s.get("status", "running")
        v = s.get("final_val_metrics") or (val_metrics(recs[-1]) if recs else {})
        t = s.get("final_test_metrics")
        d = s.get("client_dispersion") or {}
        repro = (s.get("reproducibility") or {})
        k = lambda key, val, note="", extra=None: html.Div(
            [html.Div(key, className="k"), html.Div(val, className="v"), html.Div(note, className="n")] + ([extra] if extra else []),
            className="card kpi")
        return [
            k("Status", html.Span(status, className=f"pill {status}"),
              f"{rc.get('experiment', {}).get('strategy', ['?'])[0]} · seed {rc.get('experiment', {}).get('seed', '?')}"),
            k("Progress", f"{r_now}/{R}", "rounds", html.Div(html.Div(style={"width": f"{100 * r_now / R:.0f}%"}), className="bar")),
            k("Val accuracy", _fmt(v.get("accuracy")), f"macro-F1 {_fmt(v.get('macro_f1'))} · global V_g"),
            k("Test accuracy", _fmt((t or {}).get("accuracy")),
              f"macro-F1 {_fmt((t or {}).get('macro_f1'))}" if t else ("held out until the end" if status != "completed" else "not evaluated")),
            k("Worst client", _fmt(d.get("worst_client_accuracy")), f"variance {_fmt(d.get('variance'), 4)}"),
            k("Upload", f"{(s.get('cumulative_bytes') or {}).get('up', 0) / 2**20:.1f} MB" if s else "—",
              f"adapter {_fmt(s['adapter_bytes'] / s['full_model_bytes'], pct=True)} of model" if s.get("full_model_bytes") else ""),
            k("Skipped / excluded", f"{s.get('rounds_skipped', sum(x['skipped'] for x in recs))} / {s.get('excluded_updates', '—')}",
              "rounds / non-finite updates"),
            k("Config", (s.get("config_hash") or "")[:8] or "—",
              f"{repro.get('device', rc.get('model', {}).get('device', '?'))} · "
              f"{'bitwise' if 'bitwise on' in repro.get('training', '') else 'statistical'} training repro"),
        ]

    @app.callback(Output("global", "figure"), Output("cmp", "children"), Output("comm", "figure"),
                  Input("tick", "n_intervals"), Input("runs", "value"), Input("metric", "value"), Input("tab", "value"))
    def overview(_, sel, metric, tab):
        if tab != "overview":
            return no_update, no_update, no_update
        sel = sel or []
        rows = comparison_rows(root, sel)
        cols = ["run", "hash", "status", "val acc", "test acc", "test F1", "worst client", "MB up", "skipped", "time s"]
        best = {}
        for c in ("val acc", "test acc", "test F1", "worst client"):
            vals = [r[c] for r in rows if isinstance(r[c], (int, float))]
            if len(set(vals)) > 1:                     # highlight only a real leader, not a tie
                best[c] = max(vals)
        cell = lambda r, c: html.Td(
            (_fmt(r[c], 1) if c in ("MB up", "time s") else _fmt(r[c])) if isinstance(r[c], float) else (r[c] if r[c] is not None else "—"),
            className="best" if c in best and r[c] is not None and r[c] == best[c] and len(rows) > 1 else None)
        table = html.Table(className="cmp", children=[html.Thead(html.Tr([html.Th(c) for c in cols])),
                                                      html.Tbody([html.Tr([cell(r, c) for c in cols]) for r in rows])]) \
            if rows else html.Div("select runs to compare", className="note")
        return fig_global(root, sel, metric), table, fig_comm(root, sel)

    @app.callback(Output("clients", "figure"), Output("participation", "figure"), Output("partition", "figure"),
                  Input("tick", "n_intervals"), Input("detail", "value"), Input("metric", "value"), Input("tab", "value"))
    def clients(_, detail, metric, tab):
        if tab != "clients" or not detail:
            return no_update, no_update, no_update
        return fig_clients(root, detail, metric), fig_participation(root, detail), fig_partition(root, detail)

    @app.callback(Output("weights", "figure"), Output("clusters", "figure"), Output("similarity", "figure"),
                  Output("aggnotes", "children"), Input("tick", "n_intervals"), Input("detail", "value"), Input("tab", "value"))
    def agg(_, detail, tab):
        if tab != "agg" or not detail:
            return no_update, no_update, no_update, no_update
        notes = strategy_notes(root, detail)
        body = [html.Div([html.B("Clustering diagnostics  "), html.Span(short(detail), className="sub")]),
                html.Ul([html.Li(n) for n in notes], className="notes")] if notes else \
            html.Div("This strategy logs no clustering diagnostics (only clustered and hybrid do).", className="note")
        return fig_weights(root, detail), fig_clusters(root, detail), fig_similarity(root, detail), body

    @app.callback(Output("heatmap", "figure"), Output("paired", "figure"), Input("tick", "n_intervals"),
                  Input("grid", "value"), Input("gstrat", "value"), Input("metric", "value"), Input("gmode", "value"),
                  Input("tab", "value"))
    def refresh_heat(_, g, s, metric, mode, tab):
        if tab != "grid":
            return no_update, no_update
        return fig_heatmap(g, s, metric, mode), fig_paired(g, metric)

    @app.callback(Output("cell", "children"), Input("heatmap", "clickData"), Input("gstrat", "value"))
    def open_cell(click, strat):                     # selecting a cell lists the underlying runs
        if not click:
            return html.Div("Click a heatmap cell to list its runs.", className="note")
        a, p = click["points"][0]["y"], click["points"][0]["x"]
        hits = [r for r in list_runs(root) if f"-{strat}-a{float(a):g}-p{float(p):g}-" in r]
        return html.Div([html.Div(f"{strat} · α={a} · p={p}", className="lbl"),
                         html.Ul([html.Li(h) for h in hits], className="notes") if hits else
                         html.Div("no runs found for this cell", className="note")])

    return app


def serve(root: Path, port: int = 8050) -> None:
    build_app(root).run(host="127.0.0.1", port=port, debug=False)    # NFR 6: bind to localhost only
