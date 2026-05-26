"""Schedule API for CIFAR-100 easy-vs-hard weighting experiments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import importlib
from types import ModuleType
from typing import Protocol


@dataclass(frozen=True)
class TaskScheduleInitContext:
    task_names: tuple[str, ...]
    easy_index: int
    hard_index: int
    total_epochs: int
    history_window: int
    seed: int
    run_name: str


@dataclass(frozen=True)
class TaskScheduleContext:
    epoch: int
    total_epochs: int
    task_names: tuple[str, ...]
    easy_index: int
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


class TaskSchedulePolicy(Protocol):
    def reset(self, ctx: TaskScheduleInitContext) -> None: ...

    def get_weights(self, ctx: TaskScheduleContext) -> Sequence[float]: ...


class FunctionTaskSchedulePolicy:
    def __init__(self, fn: Callable[[TaskScheduleContext], Sequence[float]]) -> None:
        self._fn = fn

    def reset(self, ctx: TaskScheduleInitContext) -> None:
        return None

    def get_weights(self, ctx: TaskScheduleContext) -> Sequence[float]:
        return self._fn(ctx)


@dataclass(frozen=True)
class LoadedTaskPolicy:
    module_path: str
    entry_name: str
    policy: TaskSchedulePolicy


def normalize_weights(weights: Sequence[float], *, n_tasks: int) -> tuple[float, ...]:
    if len(weights) != n_tasks:
        raise ValueError(f"Expected {n_tasks} weights, got {len(weights)}.")

    cleaned = []
    for value in weights:
        value = float(value)
        if value != value or value == float("inf") or value == float("-inf"):
            cleaned.append(0.0)
        else:
            cleaned.append(max(0.0, value))

    total = sum(cleaned)
    if total <= 0.0:
        return tuple(1.0 / n_tasks for _ in range(n_tasks))
    return tuple(value / total for value in cleaned)


def _resolve_module(module_path: str) -> ModuleType:
    try:
        return importlib.import_module(module_path)
    except Exception as exc:
        raise ImportError(f"Could not import task schedule module '{module_path}': {exc}") from exc


def _coerce_policy(entry: object, *, entry_name: str) -> TaskSchedulePolicy:
    if isinstance(entry, type):
        entry = entry()

    if hasattr(entry, "get_weights"):
        if not hasattr(entry, "reset"):

            class _WrappedObjectPolicy:
                def __init__(self, obj: object) -> None:
                    self._obj = obj

                def reset(self, ctx: TaskScheduleInitContext) -> None:
                    return None

                def get_weights(self, ctx: TaskScheduleContext) -> Sequence[float]:
                    return self._obj.get_weights(ctx)

            return _WrappedObjectPolicy(entry)
        return entry  # type: ignore[return-value]

    if callable(entry):
        return FunctionTaskSchedulePolicy(entry)

    raise TypeError(
        f"Entry '{entry_name}' must be a function, an object with get_weights(), or a class."
    )


def load_task_schedule_policy(
    module_path: str,
    *,
    object_name: str | None = None,
    function_name: str | None = None,
) -> LoadedTaskPolicy:
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
    return LoadedTaskPolicy(
        module_path=module_path,
        entry_name=entry_name,
        policy=_coerce_policy(entry, entry_name=entry_name),
    )
