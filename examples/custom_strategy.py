"""UC11 / FR 18: plug in a new aggregation strategy without changing core code.

    fedtune run -c configs/smoke.yaml --plugin examples/custom_strategy.py \
                --set "experiment.strategy=[fedavg,uniform]"

A strategy only returns per-client weights (and optional clusters); the LoRA merger,
dropout handling, logging and the dashboard work unchanged.
"""
import numpy as np

from fedtune.coordinator.strategies import RoundContext, Strategy, register


@register("uniform")
class UnweightedMean(Strategy):
    needs_summary = False            # set True to receive ctx.summaries_c / ctx.summaries_y

    def compute_weights(self, ctx: RoundContext):
        m = len(ctx.n)
        return np.full(m, 1.0 / m), None
