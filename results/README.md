# Results

## AG News (final protocol)
- Code fingerprint: f237c5a38e540668 (`python tools/fingerprint.py`); tag `code-freeze-v1`.
- Model: distilbert-base-uncased, LoRA rank 8, alpha 16, target q_lin and v_lin.
- Data: AG News, 8000 examples, max length 96, 10 clients, 15 rounds, 1 local epoch, batch size 16.
- Tuning: one seed (1000), 6 trials per strategy, at alpha 0.5 and dropout 0.2; selected on the mean global-validation accuracy of the last 3 rounds. Never on test.
- Grid: alpha 0.1, 0.5, 5.0 x dropout 0, 0.2, 0.5 x seeds 1, 2, 3 x 4 strategies = 108 runs, run in 5 shards on Kaggle GPUs (FILL IN GPU type), merged with `tools/merge_grid.py`.
- Files: `tuning/` (frozen tuned.yaml and trial table), `grid/` (grid_summary.csv, paired_vs_fedavg.csv), `lr_control/` (FedAvg at lr 1e-3, alpha 0.1), `figures/` (per-seed plot), `clustering_check.txt`.

## Known limitations
- 3 seeds per setting; tuning used a single seed and a single setting.
- FedAvg's tuned learning rate (2e-3) was at the top of its search range; a control at 1e-3 behaved the same at alpha 0.1.
- AG News only so far.
- Raw run folders are not in git; they are kept in a separate backup.

## XNLI
To be added after the XNLI runs finish (code fingerprint b7c4dfb096585b10, config `configs/tune_xnli.yaml`).
