"""Schedule policy API for pluggable multi-loss weighting experiments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import importlib
from types import ModuleType
from typing import Protocol


@dataclass(frozen=True)
class ScheduleInitContext:
    """Static run metadata passed once before training starts."""

    sigma_cols: tuple[str, ...]
    sigma_indices: tuple[int, ...]
    hard_index: int
    total_epochs: int
    history_window: int
    seed: int
    run_name: str


@dataclass(frozen=True)
class ScheduleContext:
    """Compressed training state exposed to schedule policies.

    `current_*` values refer to the most recently completed epoch. They are therefore
    `None` before epoch 0 starts.
    """

    epoch: int
    total_epochs: int
    sigma_cols: tuple[str, ...]
    sigma_indices: tuple[int, ...]
    hard_index: int
    current_train_losses: tuple[float, ...] | None
    current_val_losses: tuple[float, ...] | None
    prev_train_losses: tuple[float, ...] | None
    prev_val_losses: tuple[float, ...] | None
    ema_train_losses: tuple[float, ...] | None
    ema_val_losses: tuple[float, ...] | None
    best_val_losses: tuple[float, ...] | None
    prev_weights: tuple[float, ...] | None
    best_hard_val_loss: float
    recent_train_losses: tuple[tuple[float, ...], ...]
    recent_val_losses: tuple[tuple[float, ...], ...]


class SchedulePolicy(Protocol):
    def reset(self, ctx: ScheduleInitContext) -> None: ...

    def get_weights(self, ctx: ScheduleContext) -> Sequence[float]: ...


class FunctionSchedulePolicy:
    """Adapter for function-style policies."""

    def __init__(self, fn: Callable[[ScheduleContext], Sequence[float]]) -> None:
        self._fn = fn

    def reset(self, ctx: ScheduleInitContext) -> None:
        return None

    def get_weights(self, ctx: ScheduleContext) -> Sequence[float]:
        return self._fn(ctx)


@dataclass(frozen=True)
class LoadedPolicy:
    module_path: str
    entry_name: str
    policy: SchedulePolicy


def normalize_weights(weights: Sequence[float], *, n_levels: int) -> tuple[float, ...]:
    """Clamp to non-negative and renormalize, falling back to uniform if needed."""
    if len(weights) != n_levels:
        raise ValueError(f"Expected {n_levels} weights, got {len(weights)}.")

    cleaned = []
    for value in weights:
        value = float(value)
        if value != value or value == float("inf") or value == float("-inf"):
            cleaned.append(0.0)
        else:
            cleaned.append(max(0.0, value))

    total = sum(cleaned)
    if total <= 0.0:
        return tuple(1.0 / n_levels for _ in range(n_levels))
    return tuple(value / total for value in cleaned)


def _resolve_module(module_path: str) -> ModuleType:
    try:
        return importlib.import_module(module_path)
    except Exception as exc:  # pragma: no cover - importlib error surface is enough
        raise ImportError(f"Could not import schedule module '{module_path}': {exc}") from exc


def _coerce_policy(entry: object, *, entry_name: str) -> SchedulePolicy:
    if isinstance(entry, type):
        entry = entry()

    if hasattr(entry, "get_weights"):
        if not hasattr(entry, "reset"):
            class _WrappedObjectPolicy:
                def __init__(self, obj: object) -> None:
                    self._obj = obj

                def reset(self, ctx: ScheduleInitContext) -> None:
                    return None

                def get_weights(self, ctx: ScheduleContext) -> Sequence[float]:
                    return self._obj.get_weights(ctx)

            return _WrappedObjectPolicy(entry)
        return entry  # type: ignore[return-value]

    if callable(entry):
        return FunctionSchedulePolicy(entry)

    raise TypeError(
        f"Entry '{entry_name}' must be a function, an object with get_weights(), or a class."
    )


def load_schedule_policy(
    module_path: str,
    *,
    object_name: str | None = None,
    function_name: str | None = None,
) -> LoadedPolicy:
    """Load a pluggable schedule policy from a Python module."""
    module = _resolve_module(module_path)

    if object_name is not None and function_name is not None:
        raise ValueError("Provide at most one of object_name or function_name.")

    if object_name is None and function_name is None:
        if hasattr(module, "policy"):
            object_name = "policy"
        elif hasattr(module, "get_weights"):
            function_name = "get_weights"
        else:
            raise AttributeError(
                f"Module '{module_path}' must define 'policy' or 'get_weights'."
            )

    entry_name = object_name or function_name or "policy"
    entry = getattr(module, entry_name)
    return LoadedPolicy(
        module_path=module_path,
        entry_name=entry_name,
        policy=_coerce_policy(entry, entry_name=entry_name),
    )
