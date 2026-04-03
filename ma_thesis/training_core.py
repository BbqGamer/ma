"""Shared training-loop primitives used by train and sweep workflows."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.optim as optim


def build_adamw_cosine(
    model: nn.Module,
    *,
    lr: float,
    epochs: int,
    eta_min_ratio: float = 0.01,
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    """Build the default optimizer/scheduler pair used across experiments."""
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * eta_min_ratio,
    )
    return optimizer, scheduler


def compute_steps_per_epoch(num_samples: int, batch_size: int) -> int:
    """Return ceil(num_samples / batch_size), clamped to at least 1."""
    return max(1, (num_samples + batch_size - 1) // batch_size)


def shuffle_pair(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tensors shuffled by a shared permutation."""
    perm = torch.randperm(X.shape[0], device=device)
    return X[perm], y[perm]


def iter_minibatches(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield contiguous minibatches from already-prepared epoch tensors."""
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = start + batch_size
        yield X[start:end], y[start:end]
