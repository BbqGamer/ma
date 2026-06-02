"""Baseline schedule policies for policy-driven curriculum training."""

from __future__ import annotations

from ma_thesis.schedule_api import ScheduleContext, ScheduleInitContext


class EmaPlateauPolicy:
    """Start easy-heavy, then shift toward hard levels when hard loss plateaus."""

    def __init__(self) -> None:
        self._n_levels = 0

    def reset(self, ctx: ScheduleInitContext) -> None:
        self._n_levels = len(ctx.sigma_cols)

    def get_weights(self, ctx: ScheduleContext) -> list[float]:
        if self._n_levels == 0:
            self._n_levels = len(ctx.sigma_cols)

        progress = ctx.epoch / max(1, ctx.total_epochs - 1)
        weights = [0.0] * self._n_levels
        for i in range(self._n_levels):
            easy_rank = self._n_levels - 1 - i
            weights[i] = max(0.05, (1.0 - progress) * (easy_rank + 1))
        weights[-1] += 1.0 + 3.0 * progress

        if (
            ctx.current_val_losses is not None
            and ctx.prev_val_losses is not None
            and len(ctx.current_val_losses) == self._n_levels
        ):
            hard_improvement = ctx.prev_val_losses[-1] - ctx.current_val_losses[-1]
            if hard_improvement < 1e-4:
                weights[-1] += 2.0

        return weights


def get_weights(ctx: ScheduleContext) -> list[float]:
    """Simple function-style baseline: linear easy-to-hard interpolation."""
    n_levels = len(ctx.sigma_cols)
    progress = ctx.epoch / max(1, ctx.total_epochs - 1)
    weights = []
    for i in range(n_levels):
        easy_bias = float(n_levels - i)
        hard_bias = float(i + 1)
        weights.append((1.0 - progress) * easy_bias + progress * hard_bias)
    return weights


policy = EmaPlateauPolicy()
