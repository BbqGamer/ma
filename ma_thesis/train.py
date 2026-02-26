from pathlib import Path
from typing import Optional

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import polars as pl
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import torch.optim as optim
import typer

from ma_thesis.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def init_weights_lecun(m):
    """
    Matches Flax's default 'lecun_normal' initialization.
    Flax Dense uses truncated normal with stddev = 1/sqrt(fan_in).
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ResBlock(nn.Module):
    """Residual block with two linear layers and SiLU activation."""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class MLP(nn.Module):
    def __init__(self, hidden_dim=256, num_blocks=4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, 1)
        self.apply(init_weights_lecun)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_proj(x)


def plot_model_surface(model, device, x_range, y_range, grid_res, title, save_path, Zg_true=None):
    """Render the model's current learned surface as a 3D plot."""
    xg = np.linspace(x_range[0], x_range[1], grid_res)
    yg = np.linspace(y_range[0], y_range[1], grid_res)
    Xg, Yg = np.meshgrid(xg, yg)
    grid_points = torch.from_numpy(np.column_stack([Xg.ravel(), Yg.ravel()])).float().to(device)

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
    return save_path


# ---------------------------------------------------------------------------
# Core training loop (one sigma level)
# ---------------------------------------------------------------------------


def train_one_level(
    model,
    level: str,
    level_idx: int,
    total_levels: int,
    X_train,
    y_train,
    X_val,
    y_val,
    y_hard_val,
    hard_col: str,
    func_name: str,
    device,
    *,
    lr: float,
    epochs: int,
    batch_size: int,
    patience: int,
    min_delta: float,
    grid_res: int,
    snapshot_interval: int,
    x_range: tuple,
    y_range: tuple,
    Zg_true,
    output_dir: Path,
    parent_run_id: str,
    global_step: int,
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
        steps_per_epoch = n_train // batch_size
        logger.info(f"Starting training on {n_train} samples (validation: {n_val})...")

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
            title=f"MLP after {level} (stage {level_idx + 1}/{total_levels})",
            save_path=surf_path,
            Zg_true=Zg_true,
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
    input_path: Path = PROCESSED_DATA_DIR / "ackley.parquet",
    output_dir: Path = FIGURES_DIR,
    patience: int = 10,
    min_delta: float = 1e-4,
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
    sigma_level: Optional[int] = typer.Option(
        None,
        help="Sigma level index to train on (0 = most smoothed, -1 = raw). "
        "Required for mode='single'. Supports negative indexing.",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        help="Custom MLflow run name. Auto-generated if not provided.",
    ),
):
    """
    Train an MLP on Gaussian continuation datasets.

    Modes:
      - curriculum: train sequentially through all sigma levels (default).
      - single: train on a single sigma level only (use --sigma-level).

    Examples:
      # Full curriculum
      python -m ma_thesis.train --input-path data/processed/ackley.parquet

      # Train only on the raw (hardest) target
      python -m ma_thesis.train --mode single --sigma-level -1

      # Train only on the most smoothed target
      python -m ma_thesis.train --mode single --sigma-level 0
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    func_name = input_path.stem
    logger.info(f"Training on {func_name} dataset from {input_path}")

    # Read parquet dataset
    df = pl.read_parquet(input_path)
    X = torch.from_numpy(df.select(["x1", "x2"]).to_numpy()).float().to(device)

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

    # Determine which levels to train on
    if mode == "single":
        if sigma_level is None:
            logger.error("--sigma-level is required when mode='single'")
            raise typer.Exit(code=1)
        # Support negative indexing
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
        default_run_name = f"{func_name}_curriculum"
    else:
        logger.error(f"Unknown mode '{mode}'. Use 'curriculum' or 'single'.")
        raise typer.Exit(code=1)

    actual_run_name = run_name or default_run_name

    # Split into train and validation (80/20) - consistent across levels
    num_samples = X.shape[0]
    n_val = int(0.2 * num_samples)
    n_train = num_samples - n_val

    perm = torch.randperm(num_samples, device=device)
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    X_train = X[train_indices]
    X_val = X[val_indices]
    y_hard_val = y_hard_all[val_indices]

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = MLP().to(device)

    # Pre-compute true surface grid for snapshot plots
    X_np = df.select(["x1", "x2"]).to_numpy()
    Y_hard_np = df[hard_col].to_numpy()
    x_range = (X_np[:, 0].min(), X_np[:, 0].max())
    y_range = (X_np[:, 1].min(), X_np[:, 1].max())
    xg = np.linspace(x_range[0], x_range[1], grid_res)
    yg = np.linspace(y_range[0], y_range[1], grid_res)
    Xg, Yg = np.meshgrid(xg, yg)
    Zg_true = griddata(X_np, Y_hard_np, (Xg, Yg), method="cubic", fill_value=np.nan)

    # --- MLflow setup ---
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=actual_run_name) as parent_run:
        mlflow.log_params(
            {
                "function": func_name,
                "mode": mode,
                "num_samples": num_samples,
                "n_train": n_train,
                "n_val": n_val,
                "num_sigma_levels": len(sigma_cols),
                "levels_trained": [l for _, l in levels_to_train],
                "patience": patience,
                "min_delta": min_delta,
                "max_epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "snapshot_interval": snapshot_interval,
                "device": str(device),
            }
        )
        mlflow.log_artifact(str(input_path), artifact_path="data")

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
            )

        # Save and log final model
        model_path = MODELS_DIR / f"{func_name}_mlp.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        logger.success(f"Final model saved to {model_path}")

        # Log the final hard validation loss to the parent run
        model.eval()
        with torch.no_grad():
            final_pred = model(X_val)
            final_hard_loss = nn.MSELoss()(final_pred, y_hard_val).item()
        mlflow.log_metric("final_hard_val_loss", final_hard_loss)
        logger.success(f"MLflow run completed: {parent_run.info.run_id}")


if __name__ == "__main__":
    app()
