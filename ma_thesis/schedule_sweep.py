"""Optuna sweep over curriculum weight schedules with frozen model hyperparameters.

This experiment isolates one question:
Does curriculum weighting help final performance on the hardest target?

All model/training hyperparameters are fixed.
Optuna searches only schedule parameters controlling how loss weights evolve over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
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

from ma_thesis.common import fit_hidden_dim_to_param_budget
from ma_thesis.config import FIGURES_DIR, PROCESSED_DATA_DIR
from ma_thesis.models import build_model

app = typer.Typer()


def _select_sigma_columns(df: pl.DataFrame, num_losses: int) -> list[str]:
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
    cols = [sigma_cols[i] for i in idx]
    return cols


def _schedule_weights(epoch: int, total_epochs: int, a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
    t = epoch / max(1, total_epochs - 1)
    logits = a + b * t
    return torch.softmax(logits / tau, dim=0)


def _serialize(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _log_weight_plot(history: list[dict[str, float]], trial_number: int, output_dir: Path) -> Path:
    if not history:
        raise ValueError("Cannot plot empty history.")
    keys = [k for k in history[0].keys() if k.startswith("weight_")]
    keys.sort(key=lambda k: int(k.split("_")[-1]))
    epochs = [int(row["epoch"]) for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    for key in keys:
        values = [float(row[key]) for row in history]
        ax.plot(epochs, values, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight")
    ax.set_title(f"Schedule weights over time (trial {trial_number})")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend(loc="best")
    plot_path = output_dir / "weights" / f"trial_{trial_number:03d}_weights.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return plot_path


@dataclass
class SweepContext:
    X_train: torch.Tensor
    Y_train: torch.Tensor
    X_val: torch.Tensor
    Y_val: torch.Tensor
    level_cols: list[str]
    func_name: str
    device: torch.device
    epochs: int
    batch_size: int
    patience: int
    min_delta: float
    lr: float
    # frozen model hyperparameters
    hp: dict[str, Any]
    # metadata/logging
    output_dir: Path
    note: str


class ScheduleObjective:
    def __init__(self, ctx: SweepContext) -> None:
        self.ctx = ctx

    def _suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        n = len(self.ctx.level_cols)
        params: dict[str, Any] = {
            "tau": trial.suggest_float("tau", 0.5, 2.5),
        }
        for i in range(n):
            params[f"a_{i}"] = trial.suggest_float(f"a_{i}", -5.0, 5.0)
            params[f"b_{i}"] = trial.suggest_float(f"b_{i}", -10.0, 10.0)
        return params

    def __call__(self, trial: optuna.Trial) -> float:
        ctx = self.ctx
        schedule = self._suggest(trial)
        n_levels = len(ctx.level_cols)
        a = torch.tensor([schedule[f"a_{i}"] for i in range(n_levels)], device=ctx.device)
        b = torch.tensor([schedule[f"b_{i}"] for i in range(n_levels)], device=ctx.device)
        tau = float(schedule["tau"])

        model = build_model(ctx.hp, ctx.device)
        optimizer = optim.AdamW(model.parameters(), lr=ctx.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=ctx.epochs,
            eta_min=ctx.lr * 0.01,
        )
        mse = nn.MSELoss()

        n_train = ctx.X_train.shape[0]
        steps_per_epoch = max(1, n_train // ctx.batch_size)

        best_hard = float("inf")
        best_state = None
        patience_counter = 0
        history: list[dict[str, float]] = []

        with mlflow.start_run(run_name=f"schedule_trial_{trial.number:03d}", nested=True):
            mlflow.set_tags(
                {
                    "experiment_type": "curriculum_schedule_search",
                    "objective": "minimize_hard_val_loss",
                    "run_role": "trial",
                    "function": ctx.func_name,
                    "n_levels": str(n_levels),
                }
            )
            mlflow.set_tag(
                "mlflow.note.content",
                (
                    f"Schedule-only trial for {ctx.func_name}. "
                    f"Model hyperparameters are frozen; only schedule params are tuned."
                ),
            )
            mlflow.log_params(
                {
                    **{f"fixed_{k}": v for k, v in ctx.hp.items()},
                    "epochs": ctx.epochs,
                    "batch_size": ctx.batch_size,
                    "lr": ctx.lr,
                    "levels": ",".join(ctx.level_cols),
                    **schedule,
                }
            )

            for epoch in range(ctx.epochs):
                weights = _schedule_weights(epoch, ctx.epochs, a, b, tau)
                model.train()
                perm = torch.randperm(n_train, device=ctx.device)
                x_perm = ctx.X_train[perm]
                y_perm = ctx.Y_train[perm]
                epoch_loss = 0.0

                for i in range(steps_per_epoch):
                    s = i * ctx.batch_size
                    e = s + ctx.batch_size
                    xb = x_perm[s:e]
                    yb = y_perm[s:e]  # (batch, levels)
                    optimizer.zero_grad(set_to_none=True)
                    preds = model(xb)
                    losses = torch.stack([mse(preds, yb[:, j : j + 1]) for j in range(n_levels)])
                    loss = torch.sum(weights * losses)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                scheduler.step()
                train_loss = epoch_loss / steps_per_epoch

                model.eval()
                with torch.no_grad():
                    pred_val = model(ctx.X_val)
                    val_losses = torch.stack(
                        [mse(pred_val, ctx.Y_val[:, j : j + 1]) for j in range(n_levels)]
                    )
                    weighted_val = torch.sum(weights * val_losses).item()
                    hard_val = val_losses[-1].item()

                metrics = {
                    "train_weighted_loss": train_loss,
                    "val_weighted_loss": weighted_val,
                    "val_hard_loss": hard_val,
                }
                for j in range(n_levels):
                    metrics[f"weight_{j}"] = float(weights[j].item())
                    metrics[f"val_loss_level_{j}"] = float(val_losses[j].item())
                mlflow.log_metrics(metrics, step=epoch)
                history.append({"epoch": epoch, **metrics})

                trial.report(hard_val, epoch)
                if trial.should_prune():
                    mlflow.set_tag("pruned", "true")
                    mlflow.log_metric("best_hard_val_loss", best_hard)
                    raise optuna.TrialPruned()

                if hard_val < best_hard - ctx.min_delta:
                    best_hard = hard_val
                    patience_counter = 0
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= ctx.patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)
            mlflow.log_metric("best_hard_val_loss", best_hard)
            mlflow.log_metric("epochs_trained", len(history))

            # Persist weight trajectory for easy inspection.
            hist_df = pl.DataFrame(history)
            hist_path = ctx.output_dir / "weights" / f"trial_{trial.number:03d}_trajectory.parquet"
            hist_path.parent.mkdir(parents=True, exist_ok=True)
            hist_df.write_parquet(hist_path)
            mlflow.log_artifact(str(hist_path), artifact_path="schedule")
            weight_plot = _log_weight_plot(history, trial.number, ctx.output_dir)
            mlflow.log_artifact(str(weight_plot), artifact_path="schedule")

        return best_hard


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "ackley.parquet",
    output_dir: Path = FIGURES_DIR / "schedule_sweep",
    experiment_name: str = "curriculum-schedule-sweep",
    study_name: str | None = None,
    study_note: str = (
        "Optuna schedule-only sweep: frozen model/training hyperparameters, "
        "optimize hardest-level validation loss."
    ),
    storage: str | None = None,
    n_trials: int = 80,
    seed: int = 42,
    # fixed model hyperparameters
    model_arch: str = "mlp",
    hidden_dim: int = 16,
    num_blocks: int = 6,
    activation: str = "gelu",
    num_layers: int = 4,
    omega_0: float = 30.0,
    num_fourier: int = 128,
    fourier_sigma: float = 1.0,
    # fixed training hyperparameters
    epochs: int = 400,
    batch_size: int = 64,
    lr: float = 3e-4,
    patience: int = 40,
    min_delta: float = 1e-6,
    min_train_per_param: float = 10.0,
    # curriculum controls
    num_losses: int = 4,
) -> None:
    """Run Optuna over schedule parameters only (frozen model/training settings)."""
    if num_losses < 2:
        raise typer.BadParameter("num_losses must be >= 2.")
    if min_train_per_param <= 0:
        raise typer.BadParameter("min_train_per_param must be > 0.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pl.read_parquet(input_path)
    func_name = input_path.stem
    levels = _select_sigma_columns(df, num_losses)
    logger.info(f"Selected levels for schedule search: {levels}")

    X_np = df.select(["x1", "x2"]).to_numpy()
    x_min = X_np.min(axis=0)
    x_max = X_np.max(axis=0)
    X_scaled = 2.0 * (X_np - x_min) / (x_max - x_min) - 1.0
    X = torch.from_numpy(X_scaled).float().to(device)
    Y = torch.from_numpy(np.column_stack([df[c].to_numpy() for c in levels])).float().to(device)

    n = X.shape[0]
    n_val = int(0.2 * n)
    n_train = n - n_val
    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    perm = torch.randperm(n, device=device, generator=gen)

    X_train, Y_train = X[perm[:n_train]], Y[perm[:n_train]]
    X_val, Y_val = X[perm[n_train:]], Y[perm[n_train:]]

    hp_requested = {
        "model_arch": model_arch,
        "hidden_dim": hidden_dim,
        "num_blocks": num_blocks,
        "activation": activation,
        "num_layers": num_layers,
        "omega_0": omega_0,
        "num_fourier": num_fourier,
        "sigma": fourier_sigma,
    }
    hp, budget = fit_hidden_dim_to_param_budget(
        hp_requested,
        n_train=int(n_train),
        min_train_per_param=min_train_per_param,
    )
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

    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(experiment_name)

    if storage is None:
        storage = f"sqlite:///{output_dir / f'schedule_sweep_{func_name}.db'}"
    if study_name is None:
        study_name = f"schedule-sweep-{func_name}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=40),
        load_if_exists=True,
    )

    # Encourage an explicit "hard-only" candidate early for interpretability.
    baseline = {"tau": 1.0}
    for i in range(len(levels)):
        baseline[f"a_{i}"] = 4.0 if i == len(levels) - 1 else -4.0
        baseline[f"b_{i}"] = 0.0
    study.enqueue_trial(baseline)

    ctx = SweepContext(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        level_cols=levels,
        func_name=func_name,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        min_delta=min_delta,
        lr=lr,
        hp=hp,
        output_dir=output_dir,
        note=study_note,
    )
    objective = ScheduleObjective(ctx)

    with mlflow.start_run(run_name=f"{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.set_tags(
            {
                "experiment_type": "curriculum_schedule_search",
                "run_role": "study",
                "function": func_name,
                "study_name": study_name,
                "n_trials_requested": str(n_trials),
            }
        )
        mlflow.set_tag("mlflow.note.content", study_note)
        mlflow.log_params(
            {
                "input_path": str(input_path),
                "storage": storage,
                "seed": seed,
                "num_losses": num_losses,
                "levels": ",".join(levels),
                "frozen_model_hp": json.dumps(hp, sort_keys=True),
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "patience": patience,
                "min_delta": min_delta,
                "requested_hidden_dim": int(budget["requested_hidden_dim"]),
                "effective_hidden_dim": int(budget["effective_hidden_dim"]),
                "target_max_params": int(budget["target_max_params"]),
                "effective_train_per_param": float(budget["effective_train_per_param"]),
                "min_train_per_param": float(min_train_per_param),
            }
        )
        _serialize(
            output_dir / "study_config.json",
            {
                "input_path": str(input_path),
                "experiment_name": experiment_name,
                "study_name": study_name,
                "study_note": study_note,
                "storage": storage,
                "n_trials": n_trials,
                "seed": seed,
                "num_losses": num_losses,
                "levels": levels,
                "frozen_hp": hp,
                "budget": {
                    "requested_hidden_dim": int(budget["requested_hidden_dim"]),
                    "effective_hidden_dim": int(budget["effective_hidden_dim"]),
                    "target_max_params": int(budget["target_max_params"]),
                    "effective_train_per_param": float(budget["effective_train_per_param"]),
                    "min_train_per_param": float(min_train_per_param),
                },
                "training": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "patience": patience,
                    "min_delta": min_delta,
                },
            },
        )
        mlflow.log_artifact(str(output_dir / "study_config.json"), artifact_path="config")

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_trial
        mlflow.log_metric("best_hard_val_loss", float(best.value))
        mlflow.log_param("best_trial_number", int(best.number))
        mlflow.log_param("best_schedule_params", json.dumps(best.params, sort_keys=True))

        logger.success(
            f"Study finished: best trial #{best.number}, "
            f"best_hard_val_loss={best.value:.6f}, params={best.params}"
        )


if __name__ == "__main__":
    app()
