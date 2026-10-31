# FedAvg learning-rate control (AG News, alpha 0.1)

FedAvg only, train.lr 1e-3 instead of its tuned 2e-3; all other settings as in the final grid
(3 dropout rates x 3 seeds = 9 runs). Code fingerprint f237c5a38e540668.
Produced with: python tools/merge_grid.py -c results/lr_control/fedavg_lr_control.yaml --runs merged_lr/runs
Compare grid_summary.csv with the FedAvg rows at alpha 0.1 in results/grid/grid_summary.csv.
