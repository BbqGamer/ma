import json
from pathlib import Path
import sys
from typing import Any

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import typer

from ma_thesis.common import paired_test_path, plot_model_surface, prepare_surface_grid
from ma_thesis.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from ma_thesis.models import build_model
from ma_thesis.optimization_metrics import (
    StepMetricsState,
    compute_step_metrics,
    critical_sharpness,
    hutchinson_hessian_trace,
    layerwise_spectral_alpha,
)

app = typer.Typer()


def _log_run_config(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    mlflow.log_artifact(str(config_path), artifact_path="config")
    return config_path


# ---------------------------------------------------------------------------
# Model definitions moved to ma_thesis/models.py
# Plotting utilities moved to ma_thesis/common.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Core training loop (one sigma level)
# ---------------------------------------------------------------------------


def train_one_level(
    model: nn.Module,
    level: str,
    level_idx: int,
    total_levels: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    y_hard_val: torch.Tensor,
    hard_col: str,
    func_name: str,
    device: torch.device,
    *,
    lr: float,
    epochs: int,
    batch_size: int,
    patience: int,
    min_delta: float,
    grid_res: int,
    snapshot_interval: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    Zg_true: np.ndarray,
    output_dir: Path,
    parent_run_id: str,
    global_step: int,
    x_min: np.ndarray | None = None,
    x_max: np.ndarray | None = None,
) -> int:
    """Train the model on a single sigma level. Returns updated global_step."""
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]

    with mlflow.start_run(run_name=f"{func_name}_{level}", nested=True):
        mlflow.log_params({"level": level, "level_idx": level_idx})

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )
        criterion = nn.MSELoss()
        steps_per_epoch = max(1, n_train // batch_size)
        logger.info(f"Starting training on {n_train} samples (validation: {n_val})...")
        step_metrics_state = StepMetricsState()
        opt_step = 0

        train_loss_history = []
        val_loss_history = []
        hard_val_loss_history = []

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            epoch_perm = torch.randperm(n_train, device=device)
            x_perm = X_train[epoch_perm]
            y_perm = y_train[epoch_perm]

            epoch_loss = 0.0
            model.train()

            for i in range(steps_per_epoch):
                start = i * batch_size
                end = start + batch_size
                x_batch = x_perm[start:end]
                y_batch = y_perm[start:end]

                optimizer.zero_grad(set_to_none=True)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                step_metrics = compute_step_metrics(model, step_metrics_state)
                if step_metrics:
                    mlflow.log_metrics(step_metrics, step=opt_step)
                    opt_step += 1
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / steps_per_epoch
            train_loss_history.append(avg_loss)
            scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                val_loss_history.append(val_loss)

                hard_val_loss = criterion(val_outputs, y_hard_val).item()
                hard_val_loss_history.append(hard_val_loss)
                hard_msg = f" | Hard Val Loss: {hard_val_loss:.5f}"

            # Log metrics to child run
            mlflow.log_metrics(
                {"train_loss": avg_loss, "val_loss": val_loss, "hard_val_loss": hard_val_loss},
                step=epoch,
            )
            # Log to parent run with global step for continuous view
            mlflow.log_metrics(
                {
                    "global/train_loss": avg_loss,
                    "global/val_loss": val_loss,
                    "global/hard_val_loss": hard_val_loss,
                },
                step=global_step,
                run_id=parent_run_id,
            )
            global_step += 1

            probe_n = min(batch_size, n_train)
            x_probe = x_perm[:probe_n]
            y_probe = y_perm[:probe_n]
            prev_mode = model.training
            model.eval()
            try:
                h_trace = hutchinson_hessian_trace(model, criterion, x_probe, y_probe)
                c_sharp = critical_sharpness(model, criterion, x_probe, y_probe, lr_ref=lr)
                alpha_mean, alpha_by_layer = layerwise_spectral_alpha(model)
            finally:
                model.train(prev_mode)
            epoch_metrics = {
                "hessian_trace": h_trace,
                "critical_sharpness": c_sharp,
                "weight_alpha": alpha_mean,
            }
            mlflow.log_metrics(epoch_metrics, step=epoch)
            for layer_name, alpha in alpha_by_layer.items():
                mlflow.log_metric(f"weight_alpha_{layer_name}", alpha, step=epoch)

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            logger.info(
                f"Epoch {epoch} | Train Loss: {avg_loss:.5f} | Val Loss: {val_loss:.5f}{hard_msg}"
            )

            # Snapshot the learned surface at regular intervals
            if snapshot_interval > 0 and (epoch + 1) % snapshot_interval == 0:
                snap_path = output_dir / f"{func_name}_{level}_epoch{epoch}.png"
                plot_model_surface(
                    model,
                    device,
                    x_range,
                    y_range,
                    grid_res,
                    title=f"{func_name} — {level} — epoch {epoch}",
                    save_path=snap_path,
                    Zg_true=Zg_true,
                    x_min=x_min,
                    x_max=x_max,
                )
                mlflow.log_artifact(str(snap_path), artifact_path="snapshots")
                logger.info(f"Saved snapshot at epoch {epoch} → {snap_path.name}")

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        logger.success(f"Training complete for {level}.")

        mlflow.log_metrics(
            {
                "best_val_loss": best_val_loss,
                "final_hard_val_loss": hard_val_loss_history[-1],
                "total_epochs": len(train_loss_history),
            }
        )

        if best_model_state is not None:
            logger.info(f"Restoring best model with val loss: {best_val_loss:.5f}")
            model.load_state_dict(best_model_state)
        model.eval()

        # --- Plot Learning Curves ---
        fig_lc = plt.figure(figsize=(10, 6))
        plt.plot(train_loss_history[1:], label="Train Loss")
        plt.plot(val_loss_history[1:], label="Val Loss")
        if level != hard_col:
            plt.plot(hard_val_loss_history[1:], label="Hard Val Loss", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(f"{func_name} - Learning Curves - {level}")
        plt.legend()
        plt.grid(True)
        lc_path = output_dir / f"{func_name}_learning_curve_{level}.png"
        fig_lc.savefig(lc_path)
        plt.close(fig_lc)
        mlflow.log_artifact(str(lc_path), artifact_path="figures")

        # --- Plot Surface (true vs predicted) at end of level ---
        surf_path = output_dir / f"{func_name}_surface_{level}.png"
        plot_model_surface(
            model,
            device,
            x_range,
            y_range,
            grid_res,
            title=f"{func_name} after {level} (stage {level_idx + 1}/{total_levels})",
            save_path=surf_path,
            Zg_true=Zg_true,
            x_min=x_min,
            x_max=x_max,
        )
        mlflow.log_artifact(str(surf_path), artifact_path="figures")
        mlflow.log_artifact(
            str(surf_path),
            artifact_path="stage_surfaces",
            run_id=parent_run_id,
        )
        logger.info(f"Saved end-of-stage surface → {surf_path.name}")

    return global_step


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "ackley_n20000_k3_ss5_seed42_train.parquet",
    output_dir: Path = FIGURES_DIR,
    patience: int = 30,
    min_delta: float = 1e-5,
    epochs: int = 1000,
    batch_size: int = 64,
    lr: float = 1e-3,
    grid_res: int = 100,
    snapshot_interval: int = 50,
    experiment_name: str = "gaussian-continuation",
    mode: str = typer.Option(
        "curriculum",
        help="Training mode: 'curriculum' (all sigmas sequentially) or 'single' (one sigma only).",
    ),
    sigma_level: int | None = typer.Option(
        None,
        help="Sigma level index to train on (0 = most smoothed, -1 = raw). "
        "Required for mode='single'. Supports negative indexing.",
    ),
    run_name: str | None = typer.Option(
        None,
        help="Custom MLflow run name. Auto-generated if not provided.",
    ),
    # --- Model architecture (manual) ---
    model_arch: str = typer.Option("mlp", help="Model architecture: mlp, siren, or fourier."),
    hidden_dim: int = typer.Option(256, help="Hidden dimension."),
    num_blocks: int = typer.Option(4, help="Residual blocks (mlp / fourier)."),
    activation: str = typer.Option("silu", help="Activation (mlp / fourier)."),
    num_layers: int = typer.Option(4, help="Number of SIREN layers."),
    omega_0: float = typer.Option(30.0, help="SIREN frequency multiplier ω₀."),
    num_fourier: int = typer.Option(128, help="Number of Fourier features."),
    fourier_sigma: float = typer.Option(10.0, help="Fourier feature scale σ."),
    # --- Auto-load best trial from a completed Optuna sweep ---
    from_sweep: str | None = typer.Option(
        None,
        help="Path to Optuna SQLite DB (e.g. reports/figures/sweep/sweep_eggholder.db). "
        "Loads the best trial's hyper-parameters, overriding manual arch options.",
    ),
    study_name: str | None = typer.Option(
        None,
        help="Optuna study name inside --from-sweep DB. Required with --from-sweep.",
    ),
) -> None:
    """
    Train a model on Gaussian-continuation datasets (curriculum learning).

    Modes
    -----
      - curriculum: train sequentially through all sigma levels (default).
      - single: train on a single sigma level only (use --sigma-level).

    The model architecture and hyper-parameters can be set manually via CLI
    options **or** loaded automatically from a completed Optuna sweep using
    ``--from-sweep`` + ``--study-name``.

    Examples
    --------
      # Full curriculum with best sweep hyper-parameters
      python -m ma_thesis.train \\
          --input-path data/processed/eggholder.parquet \\
          --from-sweep reports/figures/sweep/sweep_eggholder.db \\
          --study-name baseline-sweep-eggholder

      # Manual SIREN curriculum
      python -m ma_thesis.train --model-arch siren --omega-0 30 --hidden-dim 256

      # Train only on the raw (hardest) target
      python -m ma_thesis.train --mode single --sigma-level -1
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    func_name = input_path.stem
    logger.info(f"Training on {func_name} dataset from {input_path}")

    # ------------------------------------------------------------------
    # Resolve hyper-parameters (from sweep DB or manual CLI options)
    # ------------------------------------------------------------------
    if from_sweep is not None:
        if study_name is None:
            logger.error("--study-name is required when using --from-sweep")
            raise typer.Exit(code=1)
        storage = f"sqlite:///{from_sweep}"
        study = optuna.load_study(study_name=study_name, storage=storage)
        best = study.best_trial
        hp = dict(best.params)
        logger.success(
            f"Loaded best trial #{best.number} from study '{study_name}' "
            f"(val_loss={best.value:.6f})"
        )
        logger.info(f"  Hyper-parameters: {hp}")
        # Override CLI defaults with sweep values
        model_arch = hp.get("model_arch", model_arch)
        hidden_dim = hp.get("hidden_dim", hidden_dim)
        batch_size = hp.get("batch_size", batch_size)
        lr = hp.get("lr", lr)
        num_blocks = hp.get("num_blocks", num_blocks)
        activation = hp.get("activation", activation)
        num_layers = hp.get("num_layers", num_layers)
        omega_0 = hp.get("omega_0", omega_0)
        num_fourier = hp.get("num_fourier", num_fourier)
        fourier_sigma = hp.get("sigma", fourier_sigma)
    else:
        hp = {
            "model_arch": model_arch,
            "hidden_dim": hidden_dim,
            "batch_size": batch_size,
            "lr": lr,
            "num_blocks": num_blocks,
            "activation": activation,
            "num_layers": num_layers,
            "omega_0": omega_0,
            "num_fourier": num_fourier,
            "sigma": fourier_sigma,
        }

    # ------------------------------------------------------------------
    # Data loading with [-1, 1] input scaling
    # ------------------------------------------------------------------
    df = pl.read_parquet(input_path)
    X_np = df.select(["x1", "x2"]).to_numpy()
    x_min = X_np.min(axis=0)  # shape (2,)
    x_max = X_np.max(axis=0)
    X_scaled = 2.0 * (X_np - x_min) / (x_max - x_min) - 1.0
    X = torch.from_numpy(X_scaled).float().to(device)
    logger.info(f"Input scaled to [-1, 1]  (x_min={x_min}, x_max={x_max})")

    # Identify sigma level columns (y_sigma_0, y_sigma_1, ...)
    sigma_cols = sorted(
        [col for col in df.columns if col.startswith("y_sigma_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    logger.info(f"Found {len(sigma_cols)} smoothing levels: {sigma_cols}")

    # The last sigma column (sigma=0) is the hard/raw target
    hard_col = sigma_cols[-1]
    y_hard_all = torch.from_numpy(df[hard_col].to_numpy()).float().unsqueeze(1).to(device)
    logger.info(f"Hard (raw) target column: {hard_col}")

    # Optional paired holdout set (same schema as train set)
    test_path = paired_test_path(input_path)
    X_test: torch.Tensor | None = None
    y_hard_test: torch.Tensor | None = None
    if test_path.exists():
        df_test = pl.read_parquet(test_path)
        if hard_col not in df_test.columns:
            logger.warning(
                f"Paired test file exists but missing '{hard_col}': {test_path}. "
                "Skipping final_test_loss logging."
            )
        else:
            X_test_np = df_test.select(["x1", "x2"]).to_numpy()
            X_test_scaled = 2.0 * (X_test_np - x_min) / (x_max - x_min) - 1.0
            X_test = torch.from_numpy(X_test_scaled).float().to(device)
            y_hard_test = (
                torch.from_numpy(df_test[hard_col].to_numpy()).float().unsqueeze(1).to(device)
            )
            logger.info(f"Using holdout test set from {test_path} ({X_test.shape[0]} samples)")
    else:
        logger.warning(f"No paired test file found at {test_path}; test loss will not be logged.")

    # Determine which levels to train on
    if mode == "single":
        if sigma_level is None:
            logger.error("--sigma-level is required when mode='single'")
            raise typer.Exit(code=1)
        idx = sigma_level if sigma_level >= 0 else len(sigma_cols) + sigma_level
        if idx < 0 or idx >= len(sigma_cols):
            logger.error(
                f"sigma-level {sigma_level} out of range "
                f"(valid: 0..{len(sigma_cols) - 1} or -{len(sigma_cols)}..-1)"
            )
            raise typer.Exit(code=1)
        levels_to_train = [(idx, sigma_cols[idx])]
        default_run_name = f"{func_name}_single_{sigma_cols[idx]}"
    elif mode == "curriculum":
        levels_to_train = list(enumerate(sigma_cols))
        default_run_name = f"{func_name}_curriculum_{model_arch}"
    else:
        logger.error(f"Unknown mode '{mode}'. Use 'curriculum' or 'single'.")
        raise typer.Exit(code=1)

    actual_run_name = run_name or default_run_name

    # Deterministic 80/20 split — same seed = same partition as sweep.py
    num_samples = X.shape[0]
    n_val = int(0.2 * num_samples)
    n_train = num_samples - n_val
    split_gen = torch.Generator(device=device)
    split_gen.manual_seed(42)
    perm = torch.randperm(num_samples, device=device, generator=split_gen)
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    X_train = X[train_indices]
    X_val = X[val_indices]
    y_hard_val = y_hard_all[val_indices]

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build model (mlp / siren / fourier)
    # ------------------------------------------------------------------
    model = build_model(hp, device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model_arch}  ({n_params:,} params)")

    # Pre-compute true surface grid for snapshot plots
    x_range, y_range, Zg_true = prepare_surface_grid(df, hard_col, grid_res)

    # --- MLflow setup ---
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=actual_run_name) as parent_run:
        mlflow.set_tags(
            {
                "strategy": mode,
                "function": func_name,
                "model_arch": str(model_arch),
                "entrypoint": "ma_thesis.train",
            }
        )
        mlflow.log_params(
            {
                **hp,
                "function": func_name,
                "mode": mode,
                "n_params": n_params,
                "num_samples": num_samples,
                "n_train": n_train,
                "n_val": n_val,
                "n_test": int(X_test.shape[0]) if X_test is not None else 0,
                "num_sigma_levels": len(sigma_cols),
                "levels_trained": [level for _, level in levels_to_train],
                "patience": patience,
                "min_delta": min_delta,
                "max_epochs": epochs,
                "snapshot_interval": snapshot_interval,
                "device": str(device),
                "from_sweep": str(from_sweep) if from_sweep else "manual",
            }
        )
        mlflow.log_artifact(str(input_path), artifact_path="data")
        if test_path.exists():
            mlflow.log_artifact(str(test_path), artifact_path="data")
        _log_run_config(
            output_dir / "configs",
            {
                "argv": sys.argv,
                "input_path": str(input_path),
                "test_path": str(test_path),
                "mode": mode,
                "run_name": actual_run_name,
                "experiment_name": experiment_name,
                "hyperparameters": hp,
                "training": {
                    "patience": patience,
                    "min_delta": min_delta,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                },
                "dataset": {
                    "function": func_name,
                    "num_samples": int(num_samples),
                    "train_samples": int(n_train),
                    "val_samples": int(n_val),
                    "sigma_columns": sigma_cols,
                },
            },
        )

        global_step = 0

        for level_idx, level in levels_to_train:
            logger.info(f"Training level: {level} ({level_idx + 1}/{len(levels_to_train)})")
            y = torch.from_numpy(df[level].to_numpy()).float().unsqueeze(1).to(device)
            y_train = y[train_indices]
            y_val = y[val_indices]

            global_step = train_one_level(
                model=model,
                level=level,
                level_idx=level_idx,
                total_levels=len(levels_to_train),
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                y_hard_val=y_hard_val,
                hard_col=hard_col,
                func_name=func_name,
                device=device,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                min_delta=min_delta,
                grid_res=grid_res,
                snapshot_interval=snapshot_interval,
                x_range=x_range,
                y_range=y_range,
                Zg_true=Zg_true,
                output_dir=output_dir,
                parent_run_id=parent_run.info.run_id,
                global_step=global_step,
                x_min=x_min,
                x_max=x_max,
            )

        # Save and log final model
        run_id = parent_run.info.run_id
        model_path = MODELS_DIR / f"{func_name}_{mode}_{model_arch}_{run_id[:8]}.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.pytorch.log_model(model, artifact_path="model_pt")
        mlflow.log_param("checkpoint_path", str(model_path))
        logger.success(f"Final model saved to {model_path}")

        # Log the final hard validation loss to the parent run
        model.eval()
        with torch.no_grad():
            final_pred = model(X_val)
            final_hard_loss = nn.MSELoss()(final_pred, y_hard_val).item()
        mlflow.log_metric("final_hard_val_loss", final_hard_loss)
        if X_test is not None and y_hard_test is not None:
            with torch.no_grad():
                final_test_pred = model(X_test)
                final_test_loss = nn.MSELoss()(final_test_pred, y_hard_test).item()
            mlflow.log_metric("final_test_loss", final_test_loss)
        logger.success(f"MLflow run completed: {parent_run.info.run_id}")


if __name__ == "__main__":
    app()
