# FedTune

Single-machine simulator for **federated LoRA fine-tuning** of language models under controlled
**data heterogeneity** (label skew, language skew, domain skew) and **client dropout**.
It compares four aggregation strategies: **FedAvg**, **Diversity-aware**, **Clustered** and **Hybrid**.

CSC-303 Computer Networks (July–Dec 2026), IIT Roorkee, Group 7.

## 1. What is in this bundle

```
fedtune/
├── README.md                  this file
├── pyproject.toml             dependencies and the `fedtune` command
├── configs/                   YAML experiment configurations (see section 4)
├── data/                      datasets (see data/README.md for sources and licences)
├── fedtune/                   the source code
│   ├── cli.py                 command-line interface
│   ├── config.py              config loading, validation, run hash
│   ├── rng.py                 named, seeded random streams
│   ├── protocol.py            client <-> coordinator messages and byte accounting
│   ├── evaluation.py          accuracy and macro-F1 (validation per round, test once)
│   ├── experiment.py          experiment, grid and tuning runners, paired statistics, export
│   ├── data/                  dataset loading and partitioning (Dirichlet / language skew)
│   ├── client/                local LoRA training and summaries
│   ├── coordinator/           round loop, participation schedule, the four strategies, LoRA merge
│   ├── infra/                 model host, checkpoints and result store
│   └── dashboard/app.py       read-only Plotly Dash dashboard
├── examples/custom_strategy.py   plug-in example (unweighted mean)
└── tests/                     unit and integration tests
```

## 2. Requirements and installation

- Python (see `requires-python` in `pyproject.toml`), `pip`, and about 5 GB of free disk space.
- A GPU (CUDA or Apple MPS) is recommended for the real experiments; the smoke test runs on CPU.
- Internet access is needed once, to download the pretrained DistilBERT model. Everything else is offline.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

This installs the `fedtune` command. `fedtune --help` lists every command and configuration key with its default and allowed range.

## 3. Quick start

```bash
# 1. Pipeline check (about 2 minutes, offline, tiny random model; accuracy stays near chance)
fedtune run -c configs/smoke.yaml

# 2. Open the dashboard (http://127.0.0.1:8050); press Refresh to load new results
fedtune dashboard --runs runs

# 3. Run the tests (59 tests, fully offline)
pytest
```

A real run (DistilBERT on AG News, all four strategies on the same partition, seed and dropout schedule):

```bash
fedtune run -c configs/reference.yaml
```

## 4. Configurations

| File | Dataset | Purpose |
|---|---|---|
| `configs/smoke.yaml` | AG News subset | Smoke test / pipeline check only (random tiny model, offline) |
| `configs/reference.yaml` | AG News | All four strategies, Dirichlet alpha = 0.3, one seed |
| `configs/tune.yaml` | AG News | Fair tuning (6 trials per strategy on seed 1000) plus the final 3 alpha x 3 dropout x 3 seeds x 4 strategies grid |
| `configs/grid.yaml` | AG News | The same grid without tuning (ablation) |
| `configs/xnli_language.yaml` | XNLI | Language skew, product merge |
| `configs/mnli_domain.yaml` | MNLI | Genre (domain) skew |

Any value can be overridden on the command line, for example
`fedtune run -c configs/reference.yaml --set "experiment.strategy=[fedavg]"`.

## 5. Commands

| Command | What it does |
|---|---|
| `fedtune validate -c CFG [--set k=v ...]` | Checks a config; a bad value names the key and the allowed range |
| `fedtune run -c CFG` | Runs every listed strategy on the same partition, seed and participation; writes `<name>-comparison.csv` |
| `fedtune resume -c CFG` | Resumes a run from its latest atomic checkpoint |
| `fedtune tune -c CFG [--grid]` | Fair tuning: same trial budget per strategy, scored on global validation; writes `tunes/<name>/tuned.yaml` (and runs its grid with `--grid`) |
| `fedtune grid -c CFG [--shard i/n]` | Runs the heterogeneity x dropout x strategy x seed grid with mean +/- std and paired differences against FedAvg (95% intervals); `--shard` splits the work across machines |
| `fedtune dashboard --runs runs` | Read-only dashboard on 127.0.0.1; refreshes only when you click Refresh |
| `fedtune export --runs runs --out exports` | Exports metrics (CSV/JSON), resolved config, partition report and plots |
| `--plugin examples/custom_strategy.py` | Registers a new aggregation strategy without changing core code |
| `--cpu-deterministic` (run/resume/grid) | CPU with deterministic kernels for bitwise reproducibility |

## 6. Reproducing the final results

```bash
# Tune all four strategies, freeze the settings, then run the 108-run grid
fedtune tune -c configs/tune.yaml --grid

# Or in two steps, sharded over several machines:
fedtune tune -c configs/tune.yaml                        # writes tunes/agnews/tuned.yaml
fedtune grid -c tunes/agnews/tuned.yaml --shard 0/3      # then 1/3 and 2/3 on the other machines

# Look at the results
fedtune dashboard --runs runs                            # open the "Grid & significance" tab
```

Language skew (XNLI) uses `configs/xnli_language.yaml` in the same way.
Each grid writes `grid_summary.csv` and `paired_vs_fedavg.csv` under `grids/<grid_id>/`.

## 7. Dashboard

Four views: **Overview** (global validation accuracy per round, run table), **Clients & data**
(per-client accuracy, participation matrix, label make-up per client), **Aggregation**
(aggregation weights, cluster assignments, update similarity) and **Grid & significance**
(mean +/- std heatmap and paired differences with 95% intervals). The dashboard only reads result
files; it never changes a run. After new results are written, click **Refresh**.

## 8. Output layout

```
runs/<name>-<strategy>-a<alpha>-p<rate>-s<seed>-<hash8>/
  config.resolved.yaml  partition_report.json  participation.npy  summary_log.jsonl
  rounds.jsonl          checkpoints/ckpt_<r>.pt  run_summary.json
grids/<grid_id>/grid_summary.csv, paired_vs_fedavg.csv, missing_cells.json
tunes/<name>/tuning_results.csv, best_params.json, tuned.yaml
exports/<run_id>/rounds.csv, rounds.json, plots.png
```

The 8-character run hash covers every setting that can change the numbers, so a changed
configuration never reuses old results.

## 9. Data

AG News (127,600 texts), MultiNLI (412,349 sentence pairs, 10 genres) and XNLI (112,500 pairs,
15 languages). Datasets are used unmodified. Only `data/ag_news/` is included in the bundle because
the MNLI training file is very large; see `data/README.md` for the sources and for how to obtain MNLI
and XNLI. Only the pretrained model is downloaded, once, on first use.

## 10. Reproducibility

- **Bitwise reproducible (any device):** dataset subsample, partitions, validation/test/reference
  splits and the client-availability schedule (seeded NumPy streams).
- **Training on CUDA or MPS:** statistically reproducible, not bitwise. Compare results as
  distributions over seeds, not as exact numbers.
- **CPU with `--cpu-deterministic`:** bitwise on the same machine and library versions.

## 11. Adding your own strategy

Copy `examples/custom_strategy.py`, register your class with `@register`, and run with
`--plugin path/to/your_strategy.py`. No core code needs to change.

## 12. Notes and limitations

- Single-machine simulation of the federation, not a real network deployment.
- Generative / instruction-tuning tasks are not implemented; the system handles classification of
  single texts and text pairs.
- No formal privacy guarantee is claimed: client summaries reveal coarse topical and label make-up by design.
- With 3 seeds, paired 95% intervals are wide and only exclude zero for large effects.
