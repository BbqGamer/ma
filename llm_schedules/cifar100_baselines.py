"""Baseline policies for CIFAR-100 easy-vs-hard weighting."""

from __future__ import annotations

from ma_thesis.cifar100_schedule import TaskScheduleContext, TaskScheduleInitContext


class UniformPolicy:
    def reset(self, ctx: TaskScheduleInitContext) -> None:
        return None

    def get_weights(self, ctx: TaskScheduleContext) -> tuple[float, float]:
        return (0.5, 0.5)


class LinearHardeningPolicy:
    def reset(self, ctx: TaskScheduleInitContext) -> None:
        return None

    def get_weights(self, ctx: TaskScheduleContext) -> tuple[float, float]:
        t = ctx.epoch / max(1, ctx.total_epochs - 1)
        hard = min(0.95, 0.15 + 0.8 * t)
        return (1.0 - hard, hard)


class PlateauPushPolicy:
    def reset(self, ctx: TaskScheduleInitContext) -> None:
        return None

    def get_weights(self, ctx: TaskScheduleContext) -> tuple[float, float]:
        hard = 0.35
        if ctx.current_val_losses and ctx.best_val_losses:
            cur = ctx.current_val_losses[ctx.hard_index]
            best = ctx.best_val_losses[ctx.hard_index]
            gap = max(0.0, (cur - best) / (abs(best) + 1e-8))
            hard += min(0.45, 2.0 * gap)
        t = ctx.epoch / max(1, ctx.total_epochs - 1)
        hard += 0.25 * t
        hard = min(0.98, max(0.02, hard))
        return (1.0 - hard, hard)


policy = LinearHardeningPolicy()
