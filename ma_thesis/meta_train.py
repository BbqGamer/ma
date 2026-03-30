"""Meta-learning curriculum training with weighted multi-loss optimization.

This module implements the curriculum learning framework from docs/design/2026-01-15-proposal.md:
- Multi-loss weighted training (all sigma levels simultaneously)
- Softmax-parameterized loss weights: w_i(u) = exp(u_i) / Σ exp(u_j)
- Inner loop: gradient descent on model parameters θ
- Outer loop: meta-update on weight parameters u with monotonic regularization
- Validation-based meta-objective

The curriculum schedule enforces:
- Crude losses (high sigma): weights must decrease over time
- Detailed losses (low sigma): weights must increase over time
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from loguru import logger
import mlflow
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import typer

from ma_thesis.common import plot_model_surface, prepare_surface_grid
from ma_thesis.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from ma_thesis.models import build_model

app = typer.Typer()


def _log_run_config(output_dir: Path, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    mlflow.log_artifact(str(config_path), artifact_path="config")
    return config_path


def softmax_weights(u: torch.Tensor) -> torch.Tensor:
    """Convert unconstrained parameters u to normalized weights via softmax.

    Parameters
    ----------
    u : torch.Tensor
        Unconstrained weight parameters, shape (N,)

    Returns
    -------
    torch.Tensor
        Normalized weights summing to 1, shape (N,)
    """
    return F.softmax(u, dim=0)


def compute_weighted_loss(
    model: nn.Module,
    X: torch.Tensor,
    Y_multi: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted sum of losses across all sigma levels.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    X : torch.Tensor
        Input features, shape (batch_size, 2)
    Y_multi : torch.Tensor
        Multi-target labels, shape (batch_size, num_sigma_levels)
    weights : torch.Tensor
        Loss weights, shape (num_sigma_levels,)

    Returns
    -------
    torch.Tensor
        Weighted loss (scalar)
    """
    predictions = model(X)  # (batch_size, 1)

    # Compute individual losses for each sigma level
    losses = []
    for i in range(Y_multi.shape[1]):
        target = Y_multi[:, i : i + 1]  # (batch_size, 1)
        loss_i = F.mse_loss(predictions, target)
        losses.append(loss_i)

    # Stack and weight
    losses_tensor = torch.stack(losses)  # (num_sigma_levels,)
    weighted_loss = torch.sum(weights * losses_tensor)

    return weighted_loss


def compute_monotonic_regularization(
    u_grad: torch.Tensor,
    num_crude: int,
    lambda_reg: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute monotonic gradient regularization.

    Penalizes gradients that push weights in the wrong direction:
    - Crude losses (indices 0 to num_crude-1): should decrease (Δu ≤ 0, so g ≥ 0 is bad)
    - Detailed losses (indices num_crude to end): should increase (Δu ≥ 0, so g ≤ 0 is bad)

    Parameters
    ----------
    u_grad : torch.Tensor
        Gradient of meta loss w.r.t. u, shape (num_sigma_levels,)
    num_crude : int
        Number of crude (high-sigma) loss levels
    lambda_reg : float
        Regularization strength

    Returns
    -------
    regularization : torch.Tensor
        Regularization term (scalar)
    stats : dict
        Statistics for logging (crude_penalty, detailed_penalty)
    """
    # For crude losses: penalize positive gradients (g > 0 means weight would increase)
    crude_penalty = torch.sum(torch.relu(u_grad[:num_crude]))

    # For detailed losses: penalize negative gradients (g < 0 means weight would decrease)
    detailed_penalty = torch.sum(torch.relu(-u_grad[num_crude:]))

    regularization = lambda_reg * (crude_penalty + detailed_penalty)

    stats = {
        "crude_penalty": crude_penalty.item(),
        "detailed_penalty": detailed_penalty.item(),
        "total_reg": regularization.item(),
    }

    return regularization, stats


def meta_train_epoch(
    model: nn.Module,
    u: torch.Tensor,
    X_train: torch.Tensor,
    Y_train_multi: torch.Tensor,
    X_val: torch.Tensor,
    Y_val_multi: torch.Tensor,
    *,
    model_optimizer: optim.Optimizer,
    meta_optimizer: optim.Optimizer,
    batch_size: int,
    inner_steps: int,
    num_crude: int,
    lambda_reg: float,
    device: torch.device,
) -> dict[str, float]:
    """Perform one meta-training epoch (inner + outer loop).

    Parameters
    ----------
    model : nn.Module
        Model to train
    u : torch.Tensor
        Unconstrained weight parameters (requires_grad=True)
    X_train, Y_train_multi : torch.Tensor
        Training data
    X_val, Y_val_multi : torch.Tensor
        Validation data
    model_optimizer : optim.Optimizer
        Optimizer for model parameters θ
    meta_optimizer : optim.Optimizer
        Optimizer for weight parameters u
    batch_size : int
        Batch size for inner loop
    inner_steps : int
        Number of inner loop updates per meta-epoch
    num_crude : int
        Number of crude (high-sigma) levels
    lambda_reg : float
        Monotonic regularization strength
    device : torch.device
        Device

    Returns
    -------
    dict
        Training statistics (train_loss, val_loss, weights, penalties, etc.)
    """
    model.train()
    n_train = X_train.shape[0]

    # --- Inner Loop: Update model parameters θ ---
    inner_losses = []
    for _ in range(inner_steps):
        # Sample batch
        indices = torch.randint(0, n_train, (batch_size,), device=device)
        X_batch = X_train[indices]
        Y_batch_multi = Y_train_multi[indices]

        # Compute current weights
        with torch.no_grad():
            weights = softmax_weights(u)

        # Inner loss and update
        model_optimizer.zero_grad()
        loss = compute_weighted_loss(model, X_batch, Y_batch_multi, weights)
        loss.backward()
        model_optimizer.step()
        inner_losses.append(loss.item())

    avg_inner_loss = np.mean(inner_losses)

    # --- Outer Loop: Update weight parameters u ---
    model.eval()
    with torch.enable_grad():
        weights = softmax_weights(u)
        val_loss = compute_weighted_loss(model, X_val, Y_val_multi, weights)

        # Compute gradient of validation loss w.r.t. u
        u_grad = torch.autograd.grad(val_loss, u, create_graph=True)[0]

        # Monotonic regularization
        reg, reg_stats = compute_monotonic_regularization(u_grad, num_crude, lambda_reg)

        # Total meta-objective
        meta_loss = val_loss + reg

        # Update u
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

    # Gather stats
    weights_np = weights.detach().cpu().numpy()
    stats = {
        "train_loss": avg_inner_loss,
        "val_loss": val_loss.item(),
        "meta_loss": meta_loss.item(),
        **reg_stats,
        **{f"weight_{i}": weights_np[i] for i in range(len(weights_np))},
    }

    return stats


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "eggholder_n20000_k3_ss7_seed42_train.parquet",
    output_dir: Path = FIGURES_DIR / "meta_curriculum",
    # --- Training hyperparameters ---
    epochs: int = 200,
    batch_size: int = 128,
    lr_model: float = 3e-4,
    lr_meta: float = 1e-3,
    inner_steps: int = 10,
    lambda_reg: float = 0.1,
    # --- Model architecture ---
    model_arch: str = "fourier",
    hidden_dim: int = 512,
    num_blocks: int = 4,
    activation: str = "gelu",
    num_fourier: int = 64,
    fourier_sigma: float = 1.45,
    # --- Curriculum settings ---
    num_crude: int | None = typer.Option(
        None,
        help="Number of crude (high-sigma) levels. Auto-detected as N//2 if not provided.",
    ),
    # --- Plotting and logging ---
    grid_res: int = 100,
    snapshot_interval: int = 20,
    experiment_name: str = "meta-curriculum",
    run_name: str | None = None,
) -> None:
    """
    Train a model using meta-learning curriculum with weighted multi-loss.

    This implements the framework from docs/design/2026-01-15-proposal.md:
    - Inner loop trains model on weighted combination of all sigma levels
    - Outer loop updates loss weights via meta-gradient with monotonic regularization
    - Curriculum schedule: crude weights decrease, detailed weights increase

    Examples
    --------
      # Train with best eggholder hyperparameters
      python -m ma_thesis.meta_train --input-path data/processed/eggholder.parquet

      # Train on ackley with custom regularization
      python -m ma_thesis.meta_train \\
          --input-path data/processed/ackley.parquet \\
          --lambda-reg 0.5 \\
          --epochs 300
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    func_name = input_path.stem
    logger.info(f"Training on {func_name} dataset from {input_path}")

    # --- Load data ---
    df = pl.read_parquet(input_path)
    X_np = df.select(["x1", "x2"]).to_numpy()
    x_min = X_np.min(axis=0)
    x_max = X_np.max(axis=0)
    X_scaled = 2.0 * (X_np - x_min) / (x_max - x_min) - 1.0
    X = torch.from_numpy(X_scaled).float().to(device)

    # Identify all sigma columns
    sigma_cols = sorted(
        [col for col in df.columns if col.startswith("y_sigma_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    num_sigma_levels = len(sigma_cols)
    logger.info(f"Found {num_sigma_levels} smoothing levels: {sigma_cols}")

    # Stack all targets into a multi-target tensor
    Y_multi = (
        torch.from_numpy(np.column_stack([df[col].to_numpy() for col in sigma_cols]))
        .float()
        .to(device)
    )

    # Auto-detect number of crude levels
    if num_crude is None:
        num_crude = num_sigma_levels // 2
    logger.info(f"Crude levels: {num_crude}, Detailed levels: {num_sigma_levels - num_crude}")

    # Train/val split (80/20)
    n = X.shape[0]
    n_val = int(0.2 * n)
    n_train = n - n_val
    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    perm = torch.randperm(n, device=device, generator=gen)

    X_train, Y_train_multi = X[perm[:n_train]], Y_multi[perm[:n_train]]
    X_val, Y_val_multi = X[perm[n_train:]], Y_multi[perm[n_train:]]

    logger.info(f"Train: {n_train} samples, Val: {n_val} samples")

    # --- Build model ---
    hp = {
        "model_arch": model_arch,
        "hidden_dim": hidden_dim,
        "num_blocks": num_blocks,
        "activation": activation,
        "num_fourier": num_fourier,
        "sigma": fourier_sigma,
    }
    model = build_model(hp, device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model_arch} ({n_params:,} params)")

    # --- Initialize weight parameters u (uniform initialization) ---
    u = torch.zeros(num_sigma_levels, device=device, requires_grad=True)
    logger.info(f"Initialized weight parameters u: {u.detach().cpu().numpy()}")

    # --- Optimizers ---
    model_optimizer = optim.AdamW(model.parameters(), lr=lr_model)
    meta_optimizer = optim.AdamW([u], lr=lr_meta)

    # --- MLflow setup ---
    output_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(experiment_name)

    actual_run_name = run_name or f"{func_name}_meta_{model_arch}"

    # Prepare ground truth surface for plotting
    hard_col = sigma_cols[-1]
    x_range, y_range, Zg_true = prepare_surface_grid(df, hard_col, grid_res)

    with mlflow.start_run(run_name=actual_run_name) as run:
        mlflow.set_tags(
            {
                "strategy": "meta",
                "function": func_name,
                "model_arch": str(model_arch),
                "entrypoint": "ma_thesis.meta_train",
            }
        )
        mlflow.log_params(
            {
                **hp,
                "function": func_name,
                "n_params": n_params,
                "num_samples": n,
                "n_train": n_train,
                "n_val": n_val,
                "num_sigma_levels": num_sigma_levels,
                "num_crude": num_crude,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr_model": lr_model,
                "lr_meta": lr_meta,
                "inner_steps": inner_steps,
                "lambda_reg": lambda_reg,
                "device": str(device),
            }
        )
        mlflow.log_artifact(str(input_path), artifact_path="data")
        _log_run_config(
            output_dir / "configs",
            {
                "argv": sys.argv,
                "input_path": str(input_path),
                "run_name": actual_run_name,
                "experiment_name": experiment_name,
                "hyperparameters": hp,
                "training": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr_model": lr_model,
                    "lr_meta": lr_meta,
                    "inner_steps": inner_steps,
                    "lambda_reg": lambda_reg,
                    "num_crude": num_crude,
                },
                "dataset": {
                    "function": func_name,
                    "num_samples": int(n),
                    "train_samples": int(n_train),
                    "val_samples": int(n_val),
                    "sigma_columns": sigma_cols,
                },
            },
        )

        logger.info(f"Starting meta-training for {epochs} epochs...")

        # Training loop
        for epoch in range(epochs):
            stats = meta_train_epoch(
                model=model,
                u=u,
                X_train=X_train,
                Y_train_multi=Y_train_multi,
                X_val=X_val,
                Y_val_multi=Y_val_multi,
                model_optimizer=model_optimizer,
                meta_optimizer=meta_optimizer,
                batch_size=batch_size,
                inner_steps=inner_steps,
                num_crude=num_crude,
                lambda_reg=lambda_reg,
                device=device,
            )

            # Log to MLflow
            mlflow.log_metrics(stats, step=epoch)

            # Print progress
            weights_str = ", ".join(f"{stats[f'weight_{i}']:.3f}" for i in range(num_sigma_levels))
            logger.info(
                f"Epoch {epoch:3d} | Train: {stats['train_loss']:.4f} | "
                f"Val: {stats['val_loss']:.4f} | Weights: [{weights_str}]"
            )

            # Snapshot
            if snapshot_interval > 0 and (epoch + 1) % snapshot_interval == 0:
                snap_path = output_dir / f"{func_name}_epoch{epoch}.png"
                plot_model_surface(
                    model,
                    device,
                    x_range,
                    y_range,
                    grid_res,
                    title=f"{func_name} - Meta-curriculum - Epoch {epoch}",
                    save_path=snap_path,
                    Zg_true=Zg_true,
                    x_min=x_min,
                    x_max=x_max,
                )
                mlflow.log_artifact(str(snap_path), artifact_path="snapshots")
                logger.info(f"Saved snapshot → {snap_path.name}")

        # Save final model
        model_path = MODELS_DIR / f"{func_name}_meta_{model_arch}_{run.info.run_id[:8]}.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.pytorch.log_model(model, artifact_path="model_pt")
        mlflow.log_param("checkpoint_path", str(model_path))
        logger.success(f"Final model saved to {model_path}")

        # Final surface plot
        final_surf_path = output_dir / f"{func_name}_meta_final.png"
        plot_model_surface(
            model,
            device,
            x_range,
            y_range,
            grid_res,
            title=f"{func_name} - Meta-curriculum (Final)",
            save_path=final_surf_path,
            Zg_true=Zg_true,
            x_min=x_min,
            x_max=x_max,
        )
        mlflow.log_artifact(str(final_surf_path), artifact_path="figures")

        logger.success(f"MLflow run completed: {run.info.run_id}")


if __name__ == "__main__":
    app()
