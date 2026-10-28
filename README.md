# FedTune

Single-machine simulation of **federated LoRA fine-tuning** of language models under
controlled **data heterogeneity** and **client dropout** (CSC-303, IIT Roorkee, Group 7).
Implements the Requirements and Specifications Document v1.0 (2 Sep 2026) and the
Design Document v1.0 (5 Oct 2026); `FR n`, `NFR n`, `UC n` and `Alg. n` below refer to them.

## Quick start (≤ 5 commands, NFR 9)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
fedtune run -c configs/reference.yaml       # DistilBERT + AG News, 4 strategies (downloads the model once)
fedtune dashboard --runs runs               # http://127.0.0.1:8050 (read-only)
pytest                                      # 41 tests (39 test functions, one parametrised x3), fully offline
```

**Final results** (fair tuning, then the grid with paired statistics):

```bash
fedtune tune -c configs/tune.yaml --grid    # tune 4 strategies x 6 trials on seed 1000, freeze, run 108-cell grid
# or in two steps / sharded:
fedtune tune -c configs/tune.yaml           # -> tunes/agnews/tuned.yaml
fedtune grid -c tunes/agnews/tuned.yaml --shard 0/3      # 1/3, 2/3 on the other machines
```

**Datasets are bundled** in `data/`, unmodified: AG News (127,600 texts), MultiNLI (412,349
sentence pairs, 10 genres) and XNLI (112,500 pairs, 15 languages). See `data/README.md` for
sources and licences. Only the pretrained model is downloaded, once, on first use.

| Config | Dataset | Purpose |
|---|---|---|
| `configs/tune.yaml` | AG News, pretrained DistilBERT | **final results**: fair tuning (6 trials/strategy, tuning seed 1000, selected on global validation) + the 3 α × 3 dropout × 3 seeds × 4 strategies grid |
| `configs/reference.yaml` | AG News, pretrained DistilBERT | all 4 strategies, Dirichlet α = 0.3, one seed; NFR 1 timing with `--set "experiment.strategy=[fedavg]"` |
| `configs/grid.yaml` | AG News, pretrained DistilBERT | the same grid **untuned** (shared defaults), as an ablation |
| `configs/mnli_domain.yaml` | MNLI | genre (domain) skew |
| `configs/xnli_language.yaml` | XNLI | language skew, product merge |
| `configs/smoke.yaml` | AG News subset | **smoke test / pipeline check only**: random tiny model, offline, ~2 min; accuracy stays near chance |

## Commands

| Command | What it does |
|---|---|
| `fedtune validate -c CFG [--set k=v ...]` | Validate config; bad values name the key and range (FR 2) |
| `fedtune run -c CFG` | Run every listed strategy on the same partition, seed and participation; writes `<name>-comparison.csv` (UC5) |
| `fedtune resume -c CFG` | Resume from the latest atomic checkpoint (UC9) |
| `--cpu-deterministic` (on run/resume/grid) | CPU with deterministic kernels, for the tightest FR 3 reproducibility (Design 2.3.2) |
| `fedtune tune -c CFG [--grid]` | Fair tuning: same trial budget per strategy on `tune.seed`, scored on global validation; writes `tunes/<name>/tuned.yaml` (and runs its grid with `--grid`) |
| `fedtune grid -c CFG [--shard i/n]` | Heterogeneity × dropout × strategy × seed grid: mean ± std per cell **and** paired differences vs FedAvg with 95% CIs (UC6, FR 26) |
| `fedtune dashboard --runs runs` | Read-only Plotly Dash app on 127.0.0.1, polls every 2 s (FR 29–32) |
| `fedtune export --runs runs --out exports` | CSV/JSON metrics, resolved config, partition report, plots (FR 28) |
| `--plugin examples/custom_strategy.py` | Register a new strategy without touching core code (FR 18, UC11) |

`fedtune --help` lists every configuration key with its default and allowed range.

## Folder structure

```
fedtune/
├── README.md
├── pyproject.toml                 dependencies + `fedtune` console script
├── data/                          real datasets, unmodified (see data/README.md)
│   ├── README.md                  sources, sizes, licences
│   ├── ag_news/                   train.csv, test.csv, classes.txt, readme.txt
│   ├── mnli/                      train / validation_matched / validation_mismatched .parquet
│   └── xnli/                      validation.parquet, test.parquet (15 languages)
├── configs/
│   ├── tune.yaml                  fair tuning + final grid (pretrained DistilBERT)
│   ├── reference.yaml             DistilBERT + AG News, 4 strategies, 10 clients × 20 rounds
│   ├── grid.yaml                  untuned 3 α × 3 dropout × 3 seeds × 4 strategies (FR 26)
│   ├── mnli_domain.yaml           MNLI genre skew
│   ├── xnli_language.yaml         XNLI language skew, product merge
│   └── smoke.yaml                 offline smoke test (pipeline check)
├── fedtune/
│   ├── cli.py                     interface layer: validate/run/resume/grid/dashboard/export
│   ├── config.py                  ConfigLoader & Validator, typed schema, run hash, per-strategy overrides
│   ├── rng.py                     RngService: named independent streams
│   ├── protocol.py                the boundary: GlobalBroadcast, ClientUpdate, byte accounting
│   ├── evaluation.py              Evaluator: accuracy, macro-F1 (global validation per round, test once)
│   ├── experiment.py              ExperimentRunner, GridRunner (+ paired stats), TuningRunner, Exporter
│   ├── data/
│   │   ├── loader.py              reads data/ag_news, data/mnli, data/xnli or your own CSV/JSONL
│   │   └── partition.py           Partitioner (Alg. 1): test / reference / global-val splits, shards, H
│   ├── client/
│   │   └── runtime.py             ClientRuntime + LocalTrainer (Alg. 3), SummaryBuilder
│   ├── coordinator/               imports nothing from client/ or data/ (tested)
│   │   ├── coordinator.py         round loop (Alg. 10), CommAccountant, codebook k-means
│   │   ├── participation.py       pre-computed R×K schedule (Alg. 2)
│   │   ├── strategies.py          registry, Cap (Alg. 5), FedAvg, Diversity (6), Clustered (7), Hybrid (8), ClusterTracker
│   │   └── linalg.py              factor-space similarity (Alg. 4) and LoRAMerger (Alg. 9)
│   ├── infra/
│   │   ├── model_host.py          ModelHost: one frozen base, swapped PEFT LoRA adapters
│   │   └── artifacts.py           ArtifactStore (JSONL append + fsync), CheckpointManager (atomic)
│   └── dashboard/
│       └── app.py                 read-only Dash dashboard (KPIs, 4 views, similarity + paired-CI plots)
├── examples/
│   └── custom_strategy.py         plug-in strategy (unweighted mean)
└── tests/
    ├── test_units.py              datasets, partition, global val, schedule, boundary, trace identity, merge,
    │                              weights, cluster rules/degeneracy, run hash, overrides, tuning, paired stats
    └── test_integration.py        smoke, NaN exclusion, skip, kill-and-resume, plug-in, grid/export/dashboard,
                                   held-out test, stale-result protection, tuning end to end
```

Generated at run time:

```
runs/<name>-<strategy>-a<alpha>-p<rate>-s<seed>-<hash8>/
  config.resolved.yaml  partition_report.json  participation.npy  summary_log.jsonl
  rounds.jsonl          checkpoints/ckpt_<r>.pt  run_summary.json
grids/<grid_id>/grid_summary.csv, paired_vs_fedavg.csv, missing_cells.json
tunes/<name>/tuning_results.csv, best_params.json, tuned.yaml
exports/<run_id>/rounds.csv, rounds.json, plots.png, ...
```

## Requirements traceability

| Requirement | Where |
|---|---|
| FR 1–2 YAML + CLI overrides, validation naming key and range | `config.py`, `cli.py` |
| FR 3, NFR 5 reproducibility | `rng.py` (SeedSequence streams), `participation.npy`, run hash; see *Reproducibility* below |
| FR 4–7 loading, 4 partition modes, report with H, test / global-val / client-val sets | `data/loader.py`, `data/partition.py` |
| FR 9–10 local LoRA training, ≤ 1B base models | `client/runtime.py`, `infra/model_host.py` |
| FR 11–12 bounded summary, logged with what it reveals | `SummaryBuilder`, `summary_log.jsonl` |
| FR 13–15a FedAvg, diversity, clustered, hybrid | `coordinator/strategies.py` |
| Design 2.3.4 fair tuning protocol | `experiment.py::run_tuning`, `configs/tune.yaml` |
| FR 16 adapter averaging + product aggregation with re-factorisation | `coordinator/linalg.py::merge` |
| FR 17 identical partitions/participation for every strategy | schedule built once from its own stream |
| FR 18, UC11 plug-in strategies | `@register`, `--plugin` |
| FR 19–23 dropout, renormalisation, m_min skip, stale updates, participation log | `coordinator.py`, `participation.py` |
| FR 24–25 metrics, bytes, dispersion | round records, `run_summary.json` |
| FR 26 grid with mean ± std, paired CIs vs FedAvg, sharding, missing cells | `experiment.py::run_grid`, `paired_vs_baseline` |
| FR 27, 34 checkpoint/resume, readable after any exit | `infra/artifacts.py`, `Coordinator.run` |
| FR 29–32 dashboard views, 2 s polling, no colour-only encoding, hover values | `dashboard/app.py` |
| FR 33 non-finite updates excluded and logged | `Coordinator._round` |
| NFR 6 privacy boundary | `protocol.py` type checks + static import test |

## Not implemented (deliberately)

* **FR 8 — generative / instruction-tuning task** ("Useful"). The Design Document (2.1.1) schedules it
  after all "Must" items; FedTune supports classification of single texts and text pairs only.
* **NFR 2 memory bound at 50 clients** is supported by the design (one shared base model, adapters on CPU)
  but has no automated test; check it with the `peak MB` column at `experiment.num_clients=10` vs `50`.

## Design notes surfaced during implementation

* **Clustered with `rule: volume` gives the same weights as FedAvg.** With
  ω_G = N_G/N and w_k = ω_G·n_k/N_G, w_k reduces to n_k/N. The default is therefore `rule: equal`
  (ω_G = 1/|clusters|); `sqrt` (ω_G ∝ √N_G) sits in between. `volume` is kept only as an explicit
  ablation and raises a `ConfigWarning`.
* **λ = 0 reduces exactly to FedAvg** only if the cap is not applied, so the cap is skipped when λ = 0.
* **Windows** (NFR 8, best effort): supported; peak memory is read through `psutil` there.
* **Codebook delivery.** The design sends the codebook "once in round 1"; a client that drops
  out of round 1 would never receive it, so it is sent on each client's first participation.
* **Heterogeneity key.** `partition.alpha` is mapped per mode (α, η = 1/(1+α), β) so that
  a smaller value always means more skew on the heatmap.
* No formal privacy is claimed (SRS 4.9): summaries reveal coarse topical and label make-up by design.

## Changes in 1.1 (review fixes)

| Problem found | Fix |
|---|---|
| **Stale results reused silently.** Run IDs had no config hash and the grid resumed any completed summary. | Run IDs end in an 8-char **run hash** of everything that can change the numbers (`config.run_hash`). A changed lr / κ / τ gets a new directory. On resume a summary or checkpoint is reused only if its stored `config_hash` matches; otherwise the run starts fresh and says so. Only the aggregation blocks a strategy reads enter its hash, so changing κ re-runs `diversity`/`hybrid` but not the FedAvg baseline. `output_dir`, `checkpoint_every` and the grid lists do not change the hash. |
| **No global validation set; test evaluated every round.** | New coordinator-held **global validation set** V_g (`data.global_val_frac`, default 5%, stratified, disjoint from T, Q and every shard). Every round logs `val_metrics` on V_g; the test set T is evaluated **once, after the last round** (`final_test_metrics`). `evaluation.test_every_round: true` restores per-round test curves for diagnostics only. Tuning never evaluates T. |
| **`clustered` with `rule: volume` is exactly FedAvg.** | Default rule is now `equal`; `sqrt` added; `volume` warns. Unit test proves the equivalence and that `equal`/`sqrt` differ. |
| **Clustering degenerates when update cosines are ≈ 0** (τ = 0.6 → all singletons; hybrid damping inert). | `tau_mode: quantile` (default) cuts the dendrogram at a quantile of *this round's* pairwise distances, so it follows the real cosine scale. A **degeneracy guard** falls back to one group (= FedAvg / diversity) when the spread is below `min_spread` or every client is a singleton, and logs `cluster_status`. Hybrid damping is now relative: if every cluster (or none) is flagged it is skipped and logged `inert`, instead of rescaling uniformly and cancelling. Each round logs the mean/sd/min/max off-diagonal cosine (`strategy_info`), shown in the dashboard. Whether clustering helps remains a research question; the code now makes the answer visible. |
| **Shipped configs could not produce final results** (grid on the tiny random model, reference FedAvg-only, no tuning). | `configs/tune.yaml` + `fedtune tune` implement the fair tuning protocol and write a frozen `tuned.yaml` (per-strategy winners in `strategy_overrides`, limited to `train.*`/`aggregation.*` keys so partitions and participation stay identical). `grid.yaml` and `reference.yaml` now use pretrained DistilBERT; `reference.yaml` runs all four strategies. |
| **Reproducibility overstated for MPS.** | See *Reproducibility* below; every `run_summary.json` now carries a `reproducibility` block naming the device and the guarantee. |
| **mean ± std over 3 seeds, no paired comparison.** | The grid also writes `paired_vs_fedavg.csv`: per cell and pooled, the per-seed differences strategy − FedAvg (same seed = same partition and participation), mean, sd, 95% paired-t CI, t, p, wins/losses and a verdict. The dashboard shows them as a Δ-heatmap and a forest plot. |
| **README test count.** | The suite has **39 test functions = 41 collected tests** (`test_other_partition_modes` is parametrised over 3 modes). The previous version had 26 functions = 29 collected tests (the earlier README number was a collected count, not a function count). |
| Dashboard | Rebuilt: KPI strip (status, progress, val/test accuracy, worst client, upload, skips, config hash and repro mode), four views (Overview, Clients & data, Aggregation, Grid & significance), run comparison table, update-similarity/degeneracy plot, paired Δ heatmap and CI forest plot; only the visible view is recomputed on each 2 s poll. |

## Reproducibility

* **Bitwise** (any device): the dataset subsample, partitions, the global-validation / test / reference
  splits and the participation schedule — they come from seeded NumPy streams and never touch the accelerator.
* **Training on MPS or CUDA: statistically reproducible, not bitwise.** Accelerator kernels are
  non-deterministic, so re-running a seed gives slightly different adapters and metrics. Report and compare
  results as distributions over seeds (which is what the paired CIs do), not as exact numbers.
* **Training on CPU with `--cpu-deterministic`**: bitwise on the same machine and library versions.
  The kill-and-resume test allows a 0.005 accuracy tolerance for this reason.

## Statistical comparison

With 3 seeds the paired 95% CI is mean ± 4.30·sd/√3, so it only excludes 0 when a gap is large relative to
its seed-to-seed variation. A 0.003 difference with an sd of a similar size is reported as *inconclusive*,
which is the correct reading. The pooled row (all 9 cells) has more pairs and therefore more power, but its
cells are not independent replicates, so treat it as a summary, not a separate test. Add seeds
(`grid.seeds`) if you need to resolve smaller effects.
