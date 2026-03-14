"""
Baseline hyperparameter sweep with Optuna.

Uses TPE sampler for efficient search and MedianPruner for early stopping
of unpromising trials.  Results are stored in a local SQLite database for
resumability and logged to MLflow for visualisation.

Usage
-----
    # 80 trials on Ackley
    python -m ma_thesis.sweep --input-path data/processed/ackley.parquet --n-trials 80

    # Resume a previous sweep (Optuna auto-resumes from the SQLite DB)
    python -m ma_thesis.sweep --input-path data/processed/ackley.parquet --n-trials 20

    # Override the Optuna storage location
    python -m ma_thesis.sweep --storage sqlite:///my_sweep.db --n-trials 40

Results are logged to MLflow experiment ``baseline-sweep-{func_name}``.
Sort by ``best_val_loss`` to find the winning configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Optional

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

from ma_thesis.common import load_and_split, plot_model_surface, prepare_surface_grid
from ma_thesis.config import FIGURES_DIR, PROCESSED_DATA_DIR
from ma_thesis.models import ACTIVATIONS, build_model

app = typer.Typer()


def _log_run_config(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "sweep_config.json"
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    mlflow.log_artifact(str(config_path), artifact_path="config")
    return config_path


# ---------------------------------------------------------------------------
# Data container – holds everything the objective needs
# ---------------------------------------------------------------------------


@dataclass
class SweepContext:
    """Immutable bundle of tensors and metadata shared across all trials."""

    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    func_name: str
    device: torch.device
    # Training budget
    epochs: int
    patience: int
    min_delta: float
    # Plotting helpers
    grid_res: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    Zg_true: np.ndarray
    output_dir: Path
    df: pl.DataFrame | None = None
    # Input scaling params (shape (2,) each) — used to map grid back for plots
    x_min: np.ndarray | None = None
    x_max: np.ndarray | None = None
    # Restrict model architectures to search over (None = all)
    model_archs: tuple[str, ...] = ("mlp", "siren", "fourier")
    # How often to report to pruner (every N epochs)
    report_interval: int = 10


# ---------------------------------------------------------------------------
# Objective – one call = one trial
# ---------------------------------------------------------------------------


class Objective:
    """Optuna objective that trains an MLP and returns the best val loss.

    Designed as a callable class so that ``SweepContext`` is injected once
    and reused across all trials without globals.
    """

    def __init__(self, ctx: SweepContext) -> None:
        self.ctx = ctx

    # ------------------------------------------------------------------
    # Search space (edit here to extend)
    # ------------------------------------------------------------------

    @staticmethod
    def suggest(trial: optuna.Trial, model_archs: tuple[str, ...]) -> dict[str, Any]:
        """Define the search space.  Easy to extend with new hyper-params."""
        arch = trial.suggest_categorical("model_arch", list(model_archs))
        hp: dict[str, Any] = {
            "model_arch": arch,
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        }
        if arch == "siren":
            hp["num_layers"] = trial.suggest_int("num_layers", 3, 8)
            hp["omega_0"] = trial.suggest_float("omega_0", 10.0, 60.0)
            hp["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        elif arch == "fourier":
            hp["num_blocks"] = trial.suggest_int("num_blocks", 2, 6, step=2)
            hp["activation"] = trial.suggest_categorical("activation", list(ACTIVATIONS.keys()))
            hp["num_fourier"] = trial.suggest_categorical("num_fourier", [64, 128, 256])
            hp["sigma"] = trial.suggest_float("sigma", 1.0, 30.0)
            hp["lr"] = trial.suggest_float("lr", 1e-4, 3e-2, log=True)
        else:  # mlp
            hp["num_blocks"] = trial.suggest_int("num_blocks", 2, 6, step=2)
            hp["activation"] = trial.suggest_categorical("activation", list(ACTIVATIONS.keys()))
            hp["lr"] = trial.suggest_float("lr", 1e-4, 3e-2, log=True)
        return hp

    # ------------------------------------------------------------------

    def __call__(self, trial: optuna.Trial) -> float:
        ctx = self.ctx
        hp = self.suggest(trial, ctx.model_archs)

        # Build model using shared utility from models module
        model = build_model(hp, ctx.device)
        arch = hp["model_arch"]

        n_params = sum(p.numel() for p in model.parameters())

        if arch == "siren":
            run_label = (
                f"trial{trial.number:03d}_siren"
                f"_h{hp['hidden_dim']}_l{hp['num_layers']}"
                f"_w{hp['omega_0']:.0f}_lr{hp['lr']:.1e}_bs{hp['batch_size']}"
            )
        elif arch == "fourier":
            run_label = (
                f"trial{trial.number:03d}_fourier"
                f"_h{hp['hidden_dim']}_b{hp['num_blocks']}"
                f"_f{hp['num_fourier']}_s{hp['sigma']:.1f}"
                f"_lr{hp['lr']:.1e}_bs{hp['batch_size']}"
            )
        else:
            run_label = (
                f"trial{trial.number:03d}_mlp"
                f"_h{hp['hidden_dim']}_b{hp['num_blocks']}"
                f"_{hp['activation']}_lr{hp['lr']:.1e}_bs{hp['batch_size']}"
            )

        if ctx.df is not None:
            target_cols = [c for c in ctx.df.columns if c.startswith("y_sigma_")]
            target_col = target_cols[-1] if target_cols else None
            dataset = mlflow.data.from_polars(
                ctx.df,
                source=f"{ctx.func_name}.parquet",
                targets=target_col,
            )
        else:
            dataset = None
        with mlflow.start_run(run_name=run_label, nested=False):
            mlflow.set_tags(
                {
                    "strategy": "sweep_trial",
                    "function": ctx.func_name,
                    "model_arch": arch,
                    "entrypoint": "ma_thesis.sweep",
                    "optuna_trial": str(trial.number),
                }
            )
            mlflow.log_params({**hp, "n_params": n_params, "function": ctx.func_name})
            _log_run_config(
                ctx.output_dir / "configs",
                {
                    "argv": sys.argv,
                    "trial_number": trial.number,
                    "run_label": run_label,
                    "function": ctx.func_name,
                    "hyperparameters": hp,
                    "training_budget": {
                        "epochs": ctx.epochs,
                        "patience": ctx.patience,
                        "min_delta": ctx.min_delta,
                        "report_interval": ctx.report_interval,
                    },
                },
            )
            if dataset is not None:
                mlflow.log_input(dataset, context="training")

            best_val_loss = self._train(trial, model, hp, run_label)

            mlflow.log_metric("best_val_loss", best_val_loss)

        return best_val_loss

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _train(
        self,
        trial: optuna.Trial,
        model: nn.Module,
        hp: dict[str, Any],
        run_label: str,
    ) -> float:
        ctx = self.ctx

        optimizer = optim.AdamW(model.parameters(), lr=hp["lr"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=ctx.epochs,
            eta_min=hp["lr"] * 0.01,
        )
        criterion = nn.MSELoss()

        n_train = ctx.X_train.shape[0]
        steps_per_epoch = max(1, n_train // hp["batch_size"])

        best_val_loss = float("inf")
        patience_counter = 0
        best_state: dict | None = None
        train_hist: list[float] = []
        val_hist: list[float] = []

        for epoch in range(ctx.epochs):
            # --- train ---
            model.train()
            perm = torch.randperm(n_train, device=ctx.device)
            x_perm = ctx.X_train[perm]
            y_perm = ctx.y_train[perm]

            epoch_loss = 0.0
            for i in range(steps_per_epoch):
                s = i * hp["batch_size"]
                e = s + hp["batch_size"]
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x_perm[s:e]), y_perm[s:e])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_train = epoch_loss / steps_per_epoch
            train_hist.append(avg_train)
            scheduler.step()

            # --- validate ---
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(ctx.X_val), ctx.y_val).item()
            val_hist.append(val_loss)

            mlflow.log_metrics({"train_loss": avg_train, "val_loss": val_loss}, step=epoch)

            # --- early stopping ---
            if val_loss < best_val_loss - ctx.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            # --- Optuna pruning ---
            if (epoch + 1) % ctx.report_interval == 0:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    mlflow.log_metric("best_val_loss", best_val_loss)
                    mlflow.set_tag("pruned", "true")
                    mlflow.end_run(status="KILLED")
                    raise optuna.TrialPruned()

            if patience_counter >= ctx.patience:
                logger.info(f"  Early stop at epoch {epoch}")
                break

        total_epochs = len(train_hist)
        mlflow.log_metric("total_epochs", total_epochs)

        # Restore best checkpoint
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        # --- artefacts: learning curve + surface plot ---
        self._log_artefacts(model, train_hist, val_hist, run_label, best_val_loss)

        # --- log learned surface at end of training ---
        surf_path = ctx.output_dir / f"{run_label}_final_surface.png"
        plot_model_surface(
            model,
            ctx.device,
            ctx.x_range,
            ctx.y_range,
            ctx.grid_res,
            title=f"Learned Surface — {ctx.func_name} — {run_label}\nval_loss={best_val_loss:.5f}",
            save_path=surf_path,
            Zg_true=ctx.Zg_true,
            x_min=ctx.x_min,
            x_max=ctx.x_max,
        )
        mlflow.log_artifact(str(surf_path), artifact_path="final_surface")

        logger.info(
            f"  Trial {trial.number} done — "
            f"val={best_val_loss:.6f}  ({total_epochs} epochs, "
            f"{sum(p.numel() for p in model.parameters())} params)"
        )
        return best_val_loss

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def _log_artefacts(
        self,
        model: nn.Module,
        train_hist: list[float],
        val_hist: list[float],
        run_label: str,
        best_val_loss: float,
    ) -> None:
        ctx = self.ctx

        # Learning curve
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_hist[1:], label="Train")
        ax.plot(val_hist[1:], label="Val")
        ax.set(xlabel="Epoch", ylabel="MSE", title=f"{ctx.func_name} — {run_label}")
        ax.legend()
        ax.grid(True)
        lc_path = ctx.output_dir / f"{run_label}_lc.png"
        fig.savefig(lc_path, dpi=100)
        plt.close(fig)
        mlflow.log_artifact(str(lc_path), artifact_path="figures")

        # Surface plot
        surf_path = ctx.output_dir / f"{run_label}_surface.png"
        plot_model_surface(
            model,
            ctx.device,
            ctx.x_range,
            ctx.y_range,
            ctx.grid_res,
            title=f"{ctx.func_name} — {run_label}\nval_loss={best_val_loss:.5f}",
            save_path=surf_path,
            Zg_true=ctx.Zg_true,
            x_min=ctx.x_min,
            x_max=ctx.x_max,
        )
        mlflow.log_artifact(str(surf_path), artifact_path="figures")


# ---------------------------------------------------------------------------
# Data loading & splitting (now imported from common.py)
# ---------------------------------------------------------------------------
# See ma_thesis.common.load_and_split for implementation


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "ackley.parquet",
    output_dir: Path = FIGURES_DIR / "sweep",
    n_trials: int = 80,
    epochs: int = 1000,
    patience: int = 30,
    min_delta: float = 1e-5,
    report_interval: int = 10,
    grid_res: int = 80,
    seed: int = 0,
    storage: Optional[str] = None,
    experiment_name: Optional[str] = None,
    model_archs: str = typer.Option(
        "mlp,siren,fourier",
        help="Comma-separated list of model architectures to search over: mlp, siren, fourier.",
    ),
):
    """
    Run a TPE hyperparameter sweep to find the best baseline model.

    Trains on the raw (hardest) target only — no curriculum.
    The Optuna study is persisted in SQLite so re-running adds trials.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- data ---
    X_train, y_train, X_val, y_val, func_name, hard_col, df, x_min, x_max = load_and_split(
        input_path, device
    )
    logger.info(
        f"Sweep on {func_name} ({hard_col})  |  train={X_train.shape[0]}  val={X_val.shape[0]}"
    )
    logger.info(f"Input scaled to [-1, 1]  (x_min={x_min}, x_max={x_max})")

    # Pre-compute true surface for the plots
    x_range, y_range, Zg_true = prepare_surface_grid(df, hard_col, grid_res)

    output_dir.mkdir(parents=True, exist_ok=True)

    archs_tuple = tuple(a.strip() for a in model_archs.split(",") if a.strip())
    logger.info(f"Model architectures in search space: {archs_tuple}")

    # --- sweep context ---
    ctx = SweepContext(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        func_name=func_name,
        device=device,
        epochs=epochs,
        patience=patience,
        min_delta=min_delta,
        grid_res=grid_res,
        x_range=x_range,
        y_range=y_range,
        Zg_true=Zg_true,
        output_dir=output_dir,
        df=df,
        x_min=x_min,
        x_max=x_max,
        model_archs=archs_tuple,
        report_interval=report_interval,
    )

    # --- MLflow ---
    experiment_name = experiment_name or f"baseline-sweep-{func_name}"
    mlflow.set_experiment(experiment_name)

    # --- Optuna study (persisted in SQLite) ---
    if storage is None:
        db_path = output_dir / f"sweep_{func_name}.db"
        storage = f"sqlite:///{db_path}"

    study_name = experiment_name
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,  # let 5 trials finish before pruning
            n_warmup_steps=50,  # don't prune in the first 50 reports
            interval_steps=1,
        ),
        load_if_exists=True,
    )

    logger.info(
        f"Optuna study '{study_name}' — {len(study.trials)} existing trials, "
        f"scheduling {n_trials} more"
    )

    objective = Objective(ctx)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # --- Summary ---
    _print_summary(study, func_name, experiment_name)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_summary(study: optuna.Study, func_name: str, experiment_name: str) -> None:
    """Pretty-print the top results after the sweep finishes."""
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    logger.info("=" * 60)
    logger.info(f"Sweep finished for {func_name}")
    logger.info(f"  Completed : {len(complete)}")
    logger.info(f"  Pruned    : {len(pruned)}")
    logger.info(f"  Total     : {len(study.trials)}")
    logger.info("=" * 60)

    if not complete:
        logger.warning("No trials completed — consider raising patience or epochs.")
        return

    top = sorted(complete, key=lambda t: t.value if t.value is not None else float("inf"))[:5]
    logger.info(f"TOP 5 configurations for {func_name}:")
    for rank, t in enumerate(top, 1):
        p = t.params
        arch = p.get("model_arch", "mlp")
        detail = f"h={p['hidden_dim']}  arch={arch}"
        if "num_blocks" in p:
            detail += f"  blocks={p['num_blocks']}"
        if "activation" in p:
            detail += f"  act={p['activation']}"
        if "omega_0" in p:
            detail += f"  ω₀={p['omega_0']:.1f}"
        if "num_fourier" in p:
            detail += f"  fourier={p['num_fourier']}"
        detail += f"  lr={p['lr']:.2e}  bs={p['batch_size']}"
        logger.info(f"  #{rank}  trial={t.number}  val_loss={t.value:.6f}  |  {detail}")

    best = study.best_trial
    logger.success(f"Best trial: #{best.number}  val_loss={best.value:.6f}  params={best.params}")
    logger.info(f"View results: mlflow ui -> experiment '{experiment_name}'")


if __name__ == "__main__":
    app()
