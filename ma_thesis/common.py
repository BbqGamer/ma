"""Shared utilities for training and hyperparameter sweeps.

This module contains common functionality used across sweep.py and train.py:
- Data loading and train/val splitting
- Plotting utilities for model surfaces
- Constants and configurations
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.interpolate import griddata
import torch
import torch.nn as nn

# Default random seed for reproducible train/val splits
TRAIN_VAL_SPLIT_SEED = 42


def load_and_split(
    input_path: Path,
    device: torch.device,
    val_split: float = 0.2,
    seed: int = TRAIN_VAL_SPLIT_SEED,
) -> tuple[
    torch.Tensor,  # X_train
    torch.Tensor,  # y_train
    torch.Tensor,  # X_val
    torch.Tensor,  # y_val
    str,  # func_name
    str,  # hard_col (last sigma column)
    pl.DataFrame,  # df
    np.ndarray,  # x_min (shape: 2,)
    np.ndarray,  # x_max (shape: 2,)
]:
    """Load parquet, scale inputs to [-1, 1], return train/val split.

    Input features are scaled to [-1, 1] so that SIREN / Fourier models
    receive coordinates in a range where their sine activations are
    well-behaved. The original coordinate bounds (x_min, x_max) are
    returned so that plotting utilities can undo the mapping.

    Parameters
    ----------
    input_path : Path
        Path to .parquet file with columns: x1, x2, y_sigma_0, y_sigma_1, ...
    device : torch.device
        Device to place tensors on
    val_split : float
        Fraction of data to use for validation (default: 0.2)
    seed : int
        Random seed for reproducible split (default: 42)

    Returns
    -------
    X_train : torch.Tensor
        Training inputs, scaled to [-1, 1], shape (n_train, 2)
    y_train : torch.Tensor
        Training targets (hardest sigma level), shape (n_train, 1)
    X_val : torch.Tensor
        Validation inputs, scaled to [-1, 1], shape (n_val, 2)
    y_val : torch.Tensor
        Validation targets (hardest sigma level), shape (n_val, 1)
    func_name : str
        Function name (stem of input_path)
    hard_col : str
        Name of the hardest (last) sigma column
    df : pl.DataFrame
        Original dataframe (for plotting / multi-target access)
    x_min : np.ndarray
        Minimum values of original coordinates (before scaling), shape (2,)
    x_max : np.ndarray
        Maximum values of original coordinates (before scaling), shape (2,)
    """
    df = pl.read_parquet(input_path)

    # Scale inputs to [-1, 1]
    X_np = df.select(["x1", "x2"]).to_numpy()
    x_min = X_np.min(axis=0)  # shape (2,)
    x_max = X_np.max(axis=0)  # shape (2,)
    X_scaled = 2.0 * (X_np - x_min) / (x_max - x_min) - 1.0
    X = torch.from_numpy(X_scaled).float().to(device)

    # Pick the hardest (last) sigma column as default target
    sigma_cols = sorted(
        [c for c in df.columns if c.startswith("y_sigma_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    hard_col = sigma_cols[-1]
    y = torch.from_numpy(df[hard_col].to_numpy()).float().unsqueeze(1).to(device)

    # Deterministic 80/20 split
    n = X.shape[0]
    n_val = int(val_split * n)
    n_train = n - n_val
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    perm = torch.randperm(n, device=device, generator=gen)

    func_name = input_path.stem
    return (
        X[perm[:n_train]],
        y[perm[:n_train]],
        X[perm[n_train:]],
        y[perm[n_train:]],
        func_name,
        hard_col,
        df,
        x_min,
        x_max,
    )


def plot_model_surface(
    model: nn.Module,
    device: torch.device,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    grid_res: int,
    title: str,
    save_path: Path | str,
    Zg_true: np.ndarray | None = None,
    x_min: np.ndarray | None = None,
    x_max: np.ndarray | None = None,
) -> Path:
    """Render the model's current learned surface as a 3D plot.

    Parameters
    ----------
    model : nn.Module
        Trained model to evaluate
    device : torch.device
        Device the model is on
    x_range : tuple[float, float]
        (min, max) for x-axis in original coordinates
    y_range : tuple[float, float]
        (min, max) for y-axis in original coordinates
    grid_res : int
        Grid resolution (number of points per dimension)
    title : str
        Plot title
    save_path : Path or str
        Where to save the figure
    Zg_true : np.ndarray, optional
        Ground truth surface (grid_res × grid_res) for comparison subplot
    x_min : np.ndarray, optional
        Minimum values of original coordinates (shape: 2,). If provided,
        grid coordinates are scaled to [-1, 1] before feeding to model.
    x_max : np.ndarray, optional
        Maximum values of original coordinates (shape: 2,). Required when
        the model was trained on normalised inputs (e.g. SIREN / Fourier).

    Returns
    -------
    Path
        Path to the saved figure
    """
    xg = np.linspace(x_range[0], x_range[1], grid_res)
    yg = np.linspace(y_range[0], y_range[1], grid_res)
    Xg, Yg = np.meshgrid(xg, yg)
    grid_np = np.column_stack([Xg.ravel(), Yg.ravel()])

    # Scale to [-1, 1] when the model was trained on normalised inputs
    if x_min is not None and x_max is not None:
        grid_np = 2.0 * (grid_np - x_min) / (x_max - x_min) - 1.0

    grid_points = torch.from_numpy(grid_np).float().to(device)

    model.eval()
    with torch.no_grad():
        Zg_pred = model(grid_points).cpu().numpy().reshape(grid_res, grid_res)

    ncols = 2 if Zg_true is not None else 1
    fig = plt.figure(figsize=(7 * ncols, 5))

    if Zg_true is not None:
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(Xg, Yg, Zg_true, cmap="viridis", edgecolor="none", alpha=0.95)
        ax1.set_title("True function")
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.set_zlabel("f(x)")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        ax2 = fig.add_subplot(1, 1, 1, projection="3d")

    ax2.plot_surface(Xg, Yg, Zg_pred, cmap="plasma", edgecolor="none", alpha=0.95)
    ax2.set_title(title)
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("f(x)")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(save_path)


def prepare_surface_grid(
    df: pl.DataFrame,
    hard_col: str,
    grid_res: int,
) -> tuple[tuple[float, float], tuple[float, float], np.ndarray]:
    """Pre-compute interpolated ground truth surface for plotting.

    Parameters
    ----------
    df : pl.DataFrame
        Dataframe with x1, x2, and target columns
    hard_col : str
        Name of the target column to interpolate
    grid_res : int
        Grid resolution

    Returns
    -------
    x_range : tuple[float, float]
        (min, max) for x-axis
    y_range : tuple[float, float]
        (min, max) for y-axis
    Zg_true : np.ndarray
        Interpolated surface (grid_res × grid_res)
    """
    X_np = df.select(["x1", "x2"]).to_numpy()
    Y_np = df[hard_col].to_numpy()
    x_range = (float(X_np[:, 0].min()), float(X_np[:, 0].max()))
    y_range = (float(X_np[:, 1].min()), float(X_np[:, 1].max()))
    xg = np.linspace(x_range[0], x_range[1], grid_res)
    yg = np.linspace(y_range[0], y_range[1], grid_res)
    Xg, Yg = np.meshgrid(xg, yg)
    Zg_true = griddata(X_np, Y_np, (Xg, Yg), method="cubic", fill_value=np.nan)
    return x_range, y_range, Zg_true
