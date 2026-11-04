# tools/

Helper scripts for running FedTune's tuning and final grids across several machines (for example Kaggle accounts), turning
the results into tables and figures, and checking the repository. They never change the experiment code, so they do not
affect run hashes or the code fingerprint. Run everything from the repository root.

| Script | What it does |
|---|---|
| `fingerprint.py` | Prints the code fingerprint (SHA-256 over `fedtune/**/*.py` and `configs/*.yaml`); every machine must print the same value. |
| `merge_tuning.py` | Combines one tuning result per strategy (`best_params.json`, or that strategy's `tuned.yaml`) into one frozen `tuned.yaml` for the grid. Refuses mismatched seeds or configs, and missing or duplicated strategies. |
| `merge_grid.py` | Summarises the finished grid from all machines' run folders. Reads results only; never starts training. |
| `check_clustering.py` | Counts how often `clustered` and `hybrid` really clustered rather than falling back to one group. |
| `cluster_alignment.py` | Adjusted Rand index between the clusters found and each client's dominant class or language, with a shuffled baseline. |
| `report_assets.py` | Report tables (CSV and Markdown) and figures (PNG) from finished runs: accuracy, paired differences to FedAvg, stability, collapse counts, cost. |
| `seed_plot.py` | Per-seed final test accuracy of every run, as an interactive HTML figure. |
| `make_latex_inputs.py` | LaTeX tables, number macros and figures for `report/main.tex`, generated from the result files. |
| `get_xnli.py` | Downloads the XNLI files from a pinned Hugging Face revision and checks them (used by the XNLI Kaggle notebook). |
| `repo_audit.py` | Pre-submission audit: sync with GitHub, tags, expected files, hygiene, code fingerprint, README claims. |

## Workflow

```bash
python3 tools/fingerprint.py
python3 tools/merge_tuning.py tuning_in/fedavg tuning_in/diversity tuning_in/clustered tuning_in/hybrid --config configs/tune.yaml --out tuned_final
python3 tools/merge_grid.py -c tuned_final/tuned.yaml --runs merged/runs
python3 tools/check_clustering.py merged/runs
python3 tools/cluster_alignment.py merged/runs --out results/report/tables/cluster_alignment.csv
python3 tools/report_assets.py merged/runs --out results/report
python3 tools/seed_plot.py merged/runs --out results/figures/per_seed_accuracy.html
python3 tools/make_latex_inputs.py
python3 tools/repo_audit.py
```

For XNLI use `configs/tune_xnli.yaml`, `merged_xnli/runs`, and add `--collapse-below 0.5` to `report_assets.py`.

## If `check_clustering.py` reports mostly fallback

`clustered` then behaves like FedAvg and `hybrid` like diversity-only weighting. Decide how to proceed before spending the
grid's compute: report it as a finding, or change the clustering configuration and re-tune.

## Tests

`pytest tests/test_tools.py tests/test_seed_plot.py tests/test_report_assets.py tests/test_cluster_alignment.py tests/test_make_latex_inputs.py`
