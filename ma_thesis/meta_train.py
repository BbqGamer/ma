"""Experimental meta-learning curriculum training with weighted multi-loss optimization.

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

from pathlib import Path
import json
import os
import sys

from loguru import logger
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import typer

from ma_thesis.common import (
    fit_hidden_dim_to_param_budget,
    plot_model_surface,
    prepare_surface_grid,
)
from ma_thesis.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from ma_thesis.io import load_dataset_bundle, select_sigma_columns, split_train_val_indices
from ma_thesis.mlflow_utils import log_dataset_reference, log_run_config
from ma_thesis.models import build_model
from ma_thesis.optimization_metrics import (
    StepMetricsState,
    compute_step_metrics,
    critical_sharpness,
    hutchinson_hessian_trace,
    layerwise_spectral_alpha,
)

app = typer.Typer()


def _write_run_status(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _configure_torch_threads_from_env() -> None:
    num_threads = os.getenv("MA_TORCH_NUM_THREADS")
    if num_threads:
        torch.set_num_threads(max(1, int(num_threads)))
    interop_threads = os.getenv("MA_TORCH_NUM_INTEROP_THREADS")
    if interop_threads:
        try:
            torch.set_num_interop_threads(max(1, int(interop_threads)))
        except RuntimeError:
            pass


def _all_finite_model_params(model: nn.Module) -> bool:
    return all(torch.isfinite(param.detach()).all() for param in model.parameters())


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
    model_scheduler: optim.lr_scheduler.LRScheduler | None,
    meta_scheduler: optim.lr_scheduler.LRScheduler | None,
    batch_size: int,
    inner_steps: int,
    num_crude: int,
    lambda_reg: float,
    sigma_cols: list[str],
    step_metrics_state: StepMetricsState,
    lr_ref: float,
    grad_clip_norm: float | None,
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
    step_metric_history: list[dict[str, float]] = []
    last_train_batch: tuple[torch.Tensor, torch.Tensor] | None = None
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
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        step_metrics = compute_step_metrics(model, step_metrics_state)
        if step_metrics:
            step_metric_history.append(step_metrics)
        model_optimizer.step()
        inner_losses.append(loss.item())
        last_train_batch = (X_batch, Y_batch_multi[:, -1:])

    avg_inner_loss = np.mean(inner_losses)

    # --- Outer Loop: Update weight parameters u ---
    model.eval()
    with torch.enable_grad():
        weights = softmax_weights(u)
        weighted_val_loss = compute_weighted_loss(model, X_val, Y_val_multi, weights)
        val_predictions = model(X_val)
        hard_val_loss = F.mse_loss(val_predictions, Y_val_multi[:, -1:])

        # Compute gradient of validation loss w.r.t. u
        u_grad = torch.autograd.grad(weighted_val_loss, u, create_graph=True)[0]

        # Monotonic regularization
        reg, reg_stats = compute_monotonic_regularization(u_grad, num_crude, lambda_reg)

        # Total meta-objective
        meta_loss = weighted_val_loss + reg

        # Update u
        meta_optimizer.zero_grad()
        meta_loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_([u], max_norm=grad_clip_norm)
        meta_optimizer.step()

    if model_scheduler is not None:
        model_scheduler.step()
    if meta_scheduler is not None:
        meta_scheduler.step()

    if (
        not torch.isfinite(weighted_val_loss)
        or not torch.isfinite(hard_val_loss)
        or not torch.isfinite(meta_loss)
    ):
        raise RuntimeError("Non-finite meta objective encountered; likely optimizer divergence.")
    if not torch.isfinite(u).all():
        raise RuntimeError("Non-finite meta weights encountered; likely optimizer divergence.")
    if not _all_finite_model_params(model):
        raise RuntimeError("Non-finite model parameters encountered; likely optimizer divergence.")

    # Gather stats
    weights_np = weights.detach().cpu().numpy()
    stats = {
        "train_loss": avg_inner_loss,
        "val_loss": hard_val_loss.item(),
        "hard_val_loss": hard_val_loss.item(),
        "weighted_val_loss": weighted_val_loss.item(),
        "meta_loss": meta_loss.item(),
        "optim/lr_model": float(model_optimizer.param_groups[0]["lr"]),
        "optim/lr_meta": float(meta_optimizer.param_groups[0]["lr"]),
        "meta/grad_norm": float(u_grad.detach().norm().item()),
        "meta/weight_entropy": float(-(weights * torch.log(weights + 1e-12)).sum().item()),
        "meta/effective_num_losses": float(
            torch.exp(-(weights * torch.log(weights + 1e-12)).sum()).item()
        ),
        "weights/max": float(weights.max().item()),
        "weights/min": float(weights.min().item()),
        "meta/crude_penalty": reg_stats["crude_penalty"],
        "meta/detailed_penalty": reg_stats["detailed_penalty"],
        "meta/total_reg": reg_stats["total_reg"],
        **{f"weights/by_index/{i}": weights_np[i] for i in range(len(weights_np))},
        **{
            f"weights/by_sigma/{sigma_cols[i]}": weights_np[i] for i in range(len(weights_np))
        },
    }

    if step_metric_history:
        step_df = np.asarray(
            [[m.get(k, np.nan) for k in sorted(step_metric_history[0])] for m in step_metric_history],
            dtype=float,
        )
        for idx, key in enumerate(sorted(step_metric_history[0])):
            stats[f"difficulty/{key}"] = float(np.nanmean(step_df[:, idx]))

    probe_batch = last_train_batch
    if probe_batch is not None:
        x_probe, y_probe = probe_batch
        criterion = nn.MSELoss()
        prev_mode = model.training
        model.eval()
        try:
            stats["difficulty/hessian_trace"] = hutchinson_hessian_trace(
                model,
                criterion,
                x_probe,
                y_probe,
            )
            stats["difficulty/critical_sharpness"] = critical_sharpness(
                model,
                criterion,
                x_probe,
                y_probe,
                lr_ref=lr_ref,
            )
            alpha_mean, alpha_by_layer = layerwise_spectral_alpha(model)
            stats["difficulty/weight_alpha"] = alpha_mean
            for layer_name, alpha in alpha_by_layer.items():
                stats[f"weights/spectral_alpha/{layer_name}"] = alpha
        finally:
            model.train(prev_mode)

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
    momentum: float = 0.9,
    lr_decay_gamma: float = 0.999,
    inner_steps: int = 10,
    lambda_reg: float = 0.1,
    grad_clip_norm: float | None = 1.0,
    split_seed: int = 42,
    val_samples: int | None = None,
    noise_ratio: float | None = None,
    dataset_function: str | None = None,
    # --- Model architecture ---
    model_arch: str = "fourier",
    hidden_dim: int = 16,
    num_blocks: int = 4,
    activation: str = "gelu",
    num_fourier: int = 64,
    fourier_sigma: float = 1.45,
    min_train_per_param: float = 10.0,
    log_dataset_artifact: bool = False,
    # --- Curriculum settings ---
    num_crude: int | None = typer.Option(
        None,
        help="Number of crude (high-sigma) levels. Auto-detected as N//2 if not provided.",
    ),
    num_losses: int | None = typer.Option(
        None,
        help="Use an evenly spaced subset of sigma levels, always including the final loss. Use 1 for the baseline final-loss-only run.",
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
    status_path_env = os.getenv("MA_META_STATUS_PATH")
    status_path = Path(status_path_env) if status_path_env else None

    _configure_torch_threads_from_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if min_train_per_param <= 0:
        raise typer.BadParameter("min_train_per_param must be > 0.")

    dataset_name = input_path.stem
    canonical_function = dataset_function or dataset_name.split("_n", 1)[0]
    logger.info(f"Training on {dataset_name} dataset from {input_path}")

    # --- Load data ---
    ds = load_dataset_bundle(input_path, device)
    df = ds.df
    X = ds.X
    x_min = ds.x_min
    x_max = ds.x_max
    available_sigma_cols = ds.sigma_cols
    sigma_cols = available_sigma_cols
    if num_losses is not None:
        sigma_cols = select_sigma_columns(df, num_losses=num_losses)
    num_sigma_levels = len(sigma_cols)
    logger.info(
        f"Using {num_sigma_levels} smoothing levels out of {len(available_sigma_cols)}: {sigma_cols}"
    )

    # Stack selected targets into a multi-target tensor
    Y_multi = (
        torch.from_numpy(np.column_stack([df[col].to_numpy() for col in sigma_cols]))
        .float()
        .to(device)
    )

    # Auto-detect number of crude levels
    if num_crude is None:
        num_crude = num_sigma_levels // 2
    logger.info(f"Crude levels: {num_crude}, Detailed levels: {num_sigma_levels - num_crude}")

    # Train/val split
    n = X.shape[0]
    train_idx, val_idx = split_train_val_indices(
        n,
        device,
        val_split=0.2,
        val_samples=val_samples,
        seed=split_seed,
    )
    n_train = int(train_idx.shape[0])
    n_val = int(val_idx.shape[0])

    X_train, Y_train_multi = X[train_idx], Y_multi[train_idx]
    X_val, Y_val_multi = X[val_idx], Y_multi[val_idx]

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
    hp, budget = fit_hidden_dim_to_param_budget(
        hp,
        n_train=int(n_train),
        min_train_per_param=min_train_per_param,
    )
    model = build_model(hp, device)
    n_params = int(budget["effective_n_params"])
    if bool(budget["was_adjusted"]):
        logger.warning(
            "Auto-adjusted hidden_dim to meet ratio target: "
            f"{budget['requested_hidden_dim']} -> {budget['effective_hidden_dim']} "
            f"(train/param={budget['effective_train_per_param']:.2f})"
        )
    if not bool(budget["budget_satisfied"]):
        logger.warning(
            "Requested ratio target could not be fully satisfied at min hidden_dim="
            f"{budget['effective_hidden_dim']} (train/param={budget['effective_train_per_param']:.2f})."
        )
    logger.info(f"Model: {model_arch} ({n_params:,} params)")

    # --- Initialize weight parameters u (uniform initialization) ---
    u = torch.zeros(num_sigma_levels, device=device, requires_grad=True)
    logger.info(f"Initialized weight parameters u: {u.detach().cpu().numpy()}")

    # --- Optimizers ---
    model_optimizer = optim.SGD(model.parameters(), lr=lr_model, momentum=momentum)
    meta_optimizer = optim.SGD([u], lr=lr_meta, momentum=momentum)
    model_scheduler = optim.lr_scheduler.ExponentialLR(model_optimizer, gamma=lr_decay_gamma)
    meta_scheduler = optim.lr_scheduler.ExponentialLR(meta_optimizer, gamma=lr_decay_gamma)

    # --- MLflow setup ---
    output_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(experiment_name)

    actual_run_name = run_name or f"{dataset_name}_meta_{model_arch}"
    _write_run_status(
        status_path,
        {
            "stage": "initializing",
            "run_name": actual_run_name,
            "function": canonical_function,
            "dataset_name": dataset_name,
            "epochs_total": epochs,
            "epoch": 0,
            "progress": 0.0,
            "input_path": str(input_path),
        },
    )

    # Prepare ground truth surface for plotting
    hard_col = sigma_cols[-1]
    x_range, y_range, Zg_true = prepare_surface_grid(df, hard_col, grid_res)

    with mlflow.start_run(run_name=actual_run_name) as run:
        logger.info(
            f"MLflow run started: {run.info.run_id} "
            f"(experiment='{experiment_name}', run_name='{actual_run_name}')"
        )
        _write_run_status(
            status_path,
            {
                "stage": "training",
                "run_name": actual_run_name,
                "function": canonical_function,
                "dataset_name": dataset_name,
                "mlflow_run_id": run.info.run_id,
                "epochs_total": epochs,
                "epoch": 0,
                "progress": 0.0,
                "input_path": str(input_path),
            },
        )
        mlflow.set_tags(
            {
                "strategy": "meta",
                "track": "experimental",
                "function": canonical_function,
                "dataset_name": dataset_name,
                "model_arch": str(model_arch),
                "entrypoint": "ma_thesis.meta_train",
                "seed": str(split_seed),
            }
        )
        mlflow.log_params(
            {
                **hp,
                "function": canonical_function,
                "dataset_name": dataset_name,
                "n_params": n_params,
                "num_samples": n,
                "n_train": n_train,
                "n_val": n_val,
                "num_sigma_levels": num_sigma_levels,
                "available_num_sigma_levels": len(available_sigma_cols),
                "num_crude": num_crude,
                "selected_sigma_cols": ",".join(sigma_cols),
                "epochs": epochs,
                "batch_size": batch_size,
                "lr_model": lr_model,
                "lr_meta": lr_meta,
                "inner_steps": inner_steps,
                "lambda_reg": lambda_reg,
                "momentum": momentum,
                "lr_decay_gamma": lr_decay_gamma,
                "noise_ratio": noise_ratio,
                "device": str(device),
                "grad_clip_norm": grad_clip_norm,
                "split_seed": split_seed,
                "seed": split_seed,
                "val_samples": n_val,
                "requested_hidden_dim": int(budget["requested_hidden_dim"]),
                "effective_hidden_dim": int(budget["effective_hidden_dim"]),
                "target_max_params": int(budget["target_max_params"]),
                "effective_train_per_param": float(budget["effective_train_per_param"]),
                "min_train_per_param": float(min_train_per_param),
                "log_dataset_artifact": log_dataset_artifact,
            }
        )
        mlflow.log_metric("run_started", 1.0, step=0)
        log_dataset_reference(
            input_path,
            key="input_dataset_path",
            log_artifact=log_dataset_artifact,
        )
        log_run_config(
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
                    "momentum": momentum,
                    "lr_decay_gamma": lr_decay_gamma,
                    "noise_ratio": noise_ratio,
                    "grad_clip_norm": grad_clip_norm,
                    "num_crude": num_crude,
                    "num_losses": num_sigma_levels,
                    "split_seed": split_seed,
                    "val_samples": n_val,
                },
                "dataset": {
                    "function": canonical_function,
                    "dataset_name": dataset_name,
                    "num_samples": int(n),
                    "train_samples": int(n_train),
                    "val_samples": int(n_val),
                    "sigma_columns": sigma_cols,
                    "all_sigma_columns": available_sigma_cols,
                },
            },
        )

        logger.info(f"Starting meta-training for {epochs} epochs...")

        step_metrics_state = StepMetricsState()

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
                model_scheduler=model_scheduler,
                meta_scheduler=meta_scheduler,
                batch_size=batch_size,
                inner_steps=inner_steps,
                num_crude=num_crude,
                lambda_reg=lambda_reg,
                sigma_cols=sigma_cols,
                step_metrics_state=step_metrics_state,
                lr_ref=lr_model,
                grad_clip_norm=grad_clip_norm,
                device=device,
            )

            # Log to MLflow
            mlflow.log_metrics(stats, step=epoch)

            # Print progress
            weights_str = ", ".join(
                f"{stats[f'weights/by_index/{i}']:.3f}" for i in range(num_sigma_levels)
            )
            logger.info(
                f"Epoch {epoch:3d} | Train: {stats['train_loss']:.4f} | "
                f"Val: {stats['val_loss']:.4f} | Weights: [{weights_str}]"
            )
            _write_run_status(
                status_path,
                {
                    "stage": "training",
                    "run_name": actual_run_name,
                    "function": canonical_function,
                    "dataset_name": dataset_name,
                    "mlflow_run_id": run.info.run_id,
                    "epochs_total": epochs,
                    "epoch": epoch + 1,
                    "progress": float((epoch + 1) / epochs),
                    "train_loss": float(stats["train_loss"]),
                    "val_loss": float(stats["val_loss"]),
                    "meta_loss": float(stats["meta_loss"]),
                    "lr_model": float(stats["optim/lr_model"]),
                    "lr_meta": float(stats["optim/lr_meta"]),
                    "weights": [
                        float(stats[f"weights/by_index/{i}"]) for i in range(num_sigma_levels)
                    ],
                },
            )

            # Snapshot
            if snapshot_interval > 0 and (epoch + 1) % snapshot_interval == 0:
                snap_path = output_dir / f"{dataset_name}_epoch{epoch}.png"
                plot_model_surface(
                    model,
                    device,
                    x_range,
                    y_range,
                    grid_res,
                    title=f"{canonical_function} - Meta-curriculum - Epoch {epoch}",
                    save_path=snap_path,
                    Zg_true=Zg_true,
                    x_min=x_min,
                    x_max=x_max,
                )
                mlflow.log_artifact(str(snap_path), artifact_path="snapshots")
                logger.info(f"Saved snapshot → {snap_path.name}")

        # Final surface plot
        final_surf_path = output_dir / f"{dataset_name}_meta_final.png"
        plot_model_surface(
            model,
            device,
            x_range,
            y_range,
            grid_res,
            title=f"{canonical_function} - Meta-curriculum (Final)",
            save_path=final_surf_path,
            Zg_true=Zg_true,
            x_min=x_min,
            x_max=x_max,
        )
        mlflow.log_artifact(str(final_surf_path), artifact_path="figures")

        logger.success(f"MLflow run completed: {run.info.run_id}")
        _write_run_status(
            status_path,
            {
                "stage": "completed",
                "run_name": actual_run_name,
                "function": canonical_function,
                "dataset_name": dataset_name,
                "mlflow_run_id": run.info.run_id,
                "epochs_total": epochs,
                "epoch": epochs,
                "progress": 1.0,
                "final_figure_path": str(final_surf_path),
            },
        )


if __name__ == "__main__":
    app()
