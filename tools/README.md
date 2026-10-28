# tools/

Helper scripts for running FedTune's tuning and final grid across several machines (e.g. Kaggle
accounts). They never change the experiment code, so they do not affect run hashes or the code
fingerprint. Run everything from the repository root.

| Script | What it does |
|---|---|
| `merge_tuning.py` | Combines one tuning result per strategy (`best_params.json`, or that strategy's `tuned.yaml`) into one frozen `tuned.yaml` for the grid. Refuses mismatched seeds/configs, missing or duplicated strategies. |
| `check_clustering.py` | Counts, per run, how often `clustered`/`hybrid` really clustered versus falling back to one group. Run it before the grid. |
| `fingerprint.py` | Prints the code fingerprint (same value as the Kaggle notebook) so you can verify a checkout equals the code that produced the results. |
| `merge_grid.py` | Summarises the finished grid from all machines' run folders. Reads results only; never starts training. |

## Workflow

```bash
# 0. confirm this checkout is the frozen code (expected: f237c5a38e540668)
python tools/fingerprint.py

# 1. one tuning result per strategy in tuning_in/<strategy>/ (unzipped Kaggle output)
python tools/merge_tuning.py tuning_in/fedavg tuning_in/diversity tuning_in/clustered tuning_in/hybrid \
    --config configs/tune.yaml --out tuned_final

# 2. does clustering do anything? (exit 0 = yes, 1 = mostly falls back, 2 = no such runs)
python tools/check_clustering.py tuning_in/clustered/runs tuning_in/hybrid/runs

# 3. after the sharded grid: copy every machine's runs/ into merged/runs/, then
python tools/merge_grid.py -c tuned_final/tuned.yaml --runs merged/runs
```

## If `check_clustering.py` reports mostly fallback

`clustered` then behaves like FedAvg and `hybrid` like diversity-only weighting, so a grid would
mostly repeat other strategies. Options: run it anyway and report "LoRA update similarity is too
uniform to cluster" as a finding, or change the clustering configuration and re-tune. Decide
before spending the grid's compute.

## Tests

`pytest tests/test_tools.py`
