"""Shared dataset IO/loading utilities for training pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ma_thesis.common import paired_test_path


@dataclass(frozen=True)
class DatasetBundle:
    """Normalized dataset tensors and metadata used by train/eval loops."""

    input_path: Path
    df: pl.DataFrame
    X: torch.Tensor
    x_min: np.ndarray
    x_max: np.ndarray
    sigma_cols: list[str]
    hard_col: str
    y_hard_all: torch.Tensor
    test_path: Path
    X_test: torch.Tensor | None
    y_hard_test: torch.Tensor | None


def load_dataset_bundle(
    input_path: Path,
    device: torch.device,
) -> DatasetBundle:
    """Load train parquet + optional paired test set and scale inputs to [-1, 1]."""
    df = pl.read_parquet(input_path)
    X_np = df.select(["x1", "x2"]).to_numpy()
    x_min = X_np.min(axis=0)
    x_max = X_np.max(axis=0)
    X_scaled = 2.0 * (X_np - x_min) / (x_max - x_min) - 1.0
    X = torch.from_numpy(X_scaled).float().to(device)

    sigma_cols = sorted(
        [col for col in df.columns if col.startswith("y_sigma_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    hard_col = sigma_cols[-1]
    y_hard_all = torch.from_numpy(df[hard_col].to_numpy()).float().unsqueeze(1).to(device)

    test_path = paired_test_path(input_path)
    X_test: torch.Tensor | None = None
    y_hard_test: torch.Tensor | None = None
    if test_path.exists():
        df_test = pl.read_parquet(test_path)
        if hard_col in df_test.columns:
            X_test_np = df_test.select(["x1", "x2"]).to_numpy()
            X_test_scaled = 2.0 * (X_test_np - x_min) / (x_max - x_min) - 1.0
            X_test = torch.from_numpy(X_test_scaled).float().to(device)
            y_hard_test = torch.from_numpy(df_test[hard_col].to_numpy()).float().unsqueeze(1).to(device)

    return DatasetBundle(
        input_path=input_path,
        df=df,
        X=X,
        x_min=x_min,
        x_max=x_max,
        sigma_cols=sigma_cols,
        hard_col=hard_col,
        y_hard_all=y_hard_all,
        test_path=test_path,
        X_test=X_test,
        y_hard_test=y_hard_test,
    )


def select_sigma_columns(df: pl.DataFrame, *, num_losses: int) -> list[str]:
    """Select an evenly spaced subset of sigma columns, always including hardest level."""
    sigma_cols = sorted(
        [c for c in df.columns if c.startswith("y_sigma_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    if len(sigma_cols) < num_losses:
        raise ValueError(
            f"Dataset has {len(sigma_cols)} y_sigma_* columns, but num_losses={num_losses}. "
            "Regenerate dataset with more sigma levels."
        )
    if num_losses < 2:
        raise ValueError("num_losses must be >= 2.")
    if num_losses == len(sigma_cols):
        return sigma_cols

    idx = np.linspace(0, len(sigma_cols) - 1, num_losses)
    idx = sorted({int(round(i)) for i in idx})
    if idx[-1] != len(sigma_cols) - 1:
        idx[-1] = len(sigma_cols) - 1
    return [sigma_cols[i] for i in idx]


def split_train_val_indices(
    num_samples: int,
    device: torch.device,
    *,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic train/val index tensors."""
    n_val = int(val_split * num_samples)
    n_train = num_samples - n_val
    split_gen = torch.Generator(device=device)
    split_gen.manual_seed(seed)
    perm = torch.randperm(num_samples, device=device, generator=split_gen)
    return perm[:n_train], perm[n_train:]
