"""Train with a pluggable schedule policy over multiple sigma losses."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import typer

from ma_thesis.common import fit_hidden_dim_to_param_budget
from ma_thesis.config import FIGURES_DIR, PROCESSED_DATA_DIR
from ma_thesis.io import load_dataset_bundle, select_sigma_columns, split_train_val_indices
from ma_thesis.mlflow_utils import log_dataset_reference, log_run_config
from ma_thesis.models import build_model
from ma_thesis.schedule_api import (
    ScheduleContext,
    ScheduleInitContext,
    load_schedule_policy,
    normalize_weights,
)
from ma_thesis.training_core import (
    build_adamw_cosine,
    compute_steps_per_epoch,
    iter_minibatches,
    shuffle_pair,
)

app = typer.Typer()


@dataclass
class PolicyTrainState:
    current_train_losses: tuple[float, ...] | None = None
    current_val_losses: tuple[float, ...] | None = None
    prev_train_losses: tuple[float, ...] | None = None
    prev_val_losses: tuple[float, ...] | None = None
    ema_train_losses: tuple[float, ...] | None = None
    ema_val_losses: tuple[float, ...] | None = None
    best_val_losses: tuple[float, ...] | None = None
    prev_weights: tuple[float, ...] | None = None
    best_hard_val_loss: float = float("inf")


def _ema_update(
    prev_ema: tuple[float, ...] | None,
    current: tuple[float, ...],
    *,
    alpha: float,
) -> tuple[float, ...]:
    if prev_ema is None:
        return current
    return tuple(alpha * cur + (1.0 - alpha) * prev for prev, cur in zip(prev_ema, current))


def _min_update(
    prev_best: tuple[float, ...] | None,
    current: tuple[float, ...],
) -> tuple[float, ...]:
    if prev_best is None:
        return current
    return tuple(min(prev, cur) for prev, cur in zip(prev_best, current))


def _sigma_indices(sigma_cols: list[str]) -> tuple[int, ...]:
    return tuple(int(col.split("_")[-1]) for col in sigma_cols)


def _build_context(
    *,
    epoch: int,
    total_epochs: int,
    sigma_cols: list[str],
    state: PolicyTrainState,
    recent_train_losses: deque[tuple[float, ...]],
    recent_val_losses: deque[tuple[float, ...]],
) -> ScheduleContext:
    sigma_indices = _sigma_indices(sigma_cols)
    return ScheduleContext(
        epoch=epoch,
        total_epochs=total_epochs,
        sigma_cols=tuple(sigma_cols),
        sigma_indices=sigma_indices,
        hard_index=len(sigma_cols) - 1,
        current_train_losses=state.current_train_losses,
        current_val_losses=state.current_val_losses,
        prev_train_losses=state.prev_train_losses,
        prev_val_losses=state.prev_val_losses,
        ema_train_losses=state.ema_train_losses,
        ema_val_losses=state.ema_val_losses,
        best_val_losses=state.best_val_losses,
        prev_weights=state.prev_weights,
        best_hard_val_loss=state.best_hard_val_loss,
        recent_train_losses=tuple(recent_train_losses),
        recent_val_losses=tuple(recent_val_losses),
    )


def _log_weight_plot(history: list[dict[str, float]], output_dir: Path, run_name: str) -> Path:
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
    ax.set_title(f"Policy weights over time ({run_name})")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend(loc="best")
    plot_path = output_dir / "weights" / f"{run_name}_weights.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return plot_path


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "ackley.parquet",
    output_dir: Path = FIGURES_DIR / "policy_train",
    experiment_name: str = "curriculum-policy",
    run_name: str | None = None,
    schedule_module: str = "llm_schedules.baselines",
    schedule_object: str | None = None,
    schedule_fn: str | None = None,
    schedule_note: str = "Pluggable schedule policy over multiple sigma losses.",
    candidate_id: str | None = None,
    llm_note: str | None = None,
    num_losses: int = 4,
    history_window: int = 5,
    ema_alpha: float = 0.3,
    seed: int = 42,
    model_arch: str = "mlp",
    hidden_dim: int = 16,
    num_blocks: int = 6,
    activation: str = "gelu",
    num_layers: int = 4,
    omega_0: float = 30.0,
    num_fourier: int = 128,
    fourier_sigma: float = 1.0,
    epochs: int = 400,
    batch_size: int = 64,
    lr: float = 3e-4,
    patience: int = 40,
    min_delta: float = 1e-6,
    min_train_per_param: float = 10.0,
    log_dataset_artifact: bool = False,
) -> dict[str, object]:
    """Train using a user-provided weighting policy over multiple sigma levels."""
    if num_losses < 2:
        raise typer.BadParameter("num_losses must be >= 2.")
    if history_window < 1:
        raise typer.BadParameter("history_window must be >= 1.")
    if not (0.0 < ema_alpha <= 1.0):
        raise typer.BadParameter("ema_alpha must be in (0, 1].")
    if min_train_per_param <= 0:
        raise typer.BadParameter("min_train_per_param must be > 0.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset_bundle(input_path, device)
    df = ds.df
    func_name = input_path.stem
    sigma_cols = select_sigma_columns(df, num_losses=num_losses)
    logger.info(f"Selected levels for policy training: {sigma_cols}")

    X = ds.X
    Y = torch.from_numpy(np.column_stack([df[c].to_numpy() for c in sigma_cols])).float().to(device)

    train_indices, val_indices = split_train_val_indices(X.shape[0], device, val_split=0.2, seed=seed)
    X_train, Y_train = X[train_indices], Y[train_indices]
    X_val, Y_val = X[val_indices], Y[val_indices]
    n_train = X_train.shape[0]

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
    model = build_model(hp, device)
    optimizer, scheduler = build_adamw_cosine(model, lr=lr, epochs=epochs)
    mse = nn.MSELoss()
    steps_per_epoch = compute_steps_per_epoch(n_train, batch_size)

    loaded_policy = load_schedule_policy(
        schedule_module,
        object_name=schedule_object,
        function_name=schedule_fn,
    )
    actual_run_name = run_name or f"{func_name}_policy_{loaded_policy.entry_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(experiment_name)

    state = PolicyTrainState()
    recent_train_losses: deque[tuple[float, ...]] = deque(maxlen=history_window)
    recent_val_losses: deque[tuple[float, ...]] = deque(maxlen=history_window)
    history: list[dict[str, float]] = []
    best_state = None
    best_hard_val = float("inf")
    patience_counter = 0

    init_ctx = ScheduleInitContext(
        sigma_cols=tuple(sigma_cols),
        sigma_indices=_sigma_indices(sigma_cols),
        hard_index=len(sigma_cols) - 1,
        total_epochs=epochs,
        history_window=history_window,
        seed=seed,
        run_name=actual_run_name,
    )
    loaded_policy.policy.reset(init_ctx)

    with mlflow.start_run(run_name=actual_run_name):
        mlflow.set_tags(
            {
                "experiment_type": "curriculum_policy",
                "function": func_name,
                "schedule_module": loaded_policy.module_path,
                "schedule_entry": loaded_policy.entry_name,
                "n_levels": str(len(sigma_cols)),
                "candidate_id": candidate_id or "",
                "llm_note": llm_note or "",
            }
        )
        note_parts = [schedule_note]
        if llm_note:
            note_parts.append(f"LLM note: {llm_note}")
        if candidate_id:
            note_parts.append(f"Candidate id: {candidate_id}")
        mlflow.set_tag("mlflow.note.content", "\n\n".join(note_parts))
        mlflow.log_params(
            {
                "input_path": str(input_path),
                "seed": seed,
                "num_losses": num_losses,
                "levels": ",".join(sigma_cols),
                "schedule_module": loaded_policy.module_path,
                "schedule_entry": loaded_policy.entry_name,
                "history_window": history_window,
                "ema_alpha": ema_alpha,
                "candidate_id": candidate_id or "",
                "llm_note": llm_note or "",
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
                **hp,
            }
        )
        log_dataset_reference(
            input_path,
            key="input_dataset_path",
            log_artifact=log_dataset_artifact,
        )
        if ds.test_path.exists():
            log_dataset_reference(
                ds.test_path,
                key="test_dataset_path",
                log_artifact=log_dataset_artifact,
            )
        log_run_config(
            output_dir / "config",
            {
                "input_path": str(input_path),
                "experiment_name": experiment_name,
                "run_name": actual_run_name,
                "schedule": {
                    "module": loaded_policy.module_path,
                    "entry": loaded_policy.entry_name,
                    "history_window": history_window,
                    "ema_alpha": ema_alpha,
                    "candidate_id": candidate_id,
                    "llm_note": llm_note,
                },
                "levels": sigma_cols,
                "training": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "patience": patience,
                    "min_delta": min_delta,
                },
                "model": hp,
                "budget": {
                    "requested_hidden_dim": int(budget["requested_hidden_dim"]),
                    "effective_hidden_dim": int(budget["effective_hidden_dim"]),
                    "target_max_params": int(budget["target_max_params"]),
                    "effective_train_per_param": float(budget["effective_train_per_param"]),
                    "min_train_per_param": float(min_train_per_param),
                },
            },
            filename="policy_config.json",
        )

        for epoch in range(epochs):
            policy_ctx = _build_context(
                epoch=epoch,
                total_epochs=epochs,
                sigma_cols=sigma_cols,
                state=state,
                recent_train_losses=recent_train_losses,
                recent_val_losses=recent_val_losses,
            )
            weights = normalize_weights(
                loaded_policy.policy.get_weights(policy_ctx),
                n_levels=len(sigma_cols),
            )
            weights_tensor = torch.tensor(weights, device=device)

            model.train()
            x_perm, y_perm = shuffle_pair(X_train, Y_train, device=device)
            epoch_weighted_loss = 0.0
            epoch_level_loss_sums = torch.zeros(len(sigma_cols), device=device)

            for xb, yb in iter_minibatches(x_perm, y_perm, batch_size=batch_size):
                optimizer.zero_grad(set_to_none=True)
                preds = model(xb)
                losses = torch.stack([mse(preds, yb[:, j : j + 1]) for j in range(len(sigma_cols))])
                loss = torch.sum(weights_tensor * losses)
                loss.backward()
                optimizer.step()
                epoch_weighted_loss += loss.item()
                epoch_level_loss_sums += losses.detach()

            scheduler.step()
            train_level_losses = tuple(
                float(v.item()) for v in (epoch_level_loss_sums / steps_per_epoch)
            )

            model.eval()
            with torch.no_grad():
                pred_val = model(X_val)
                val_losses_tensor = torch.stack(
                    [mse(pred_val, Y_val[:, j : j + 1]) for j in range(len(sigma_cols))]
                )
                val_level_losses = tuple(float(v.item()) for v in val_losses_tensor)
                weighted_val = float(torch.sum(weights_tensor * val_losses_tensor).item())
                hard_val = val_level_losses[-1]

            state.prev_train_losses = state.current_train_losses
            state.prev_val_losses = state.current_val_losses
            state.current_train_losses = train_level_losses
            state.current_val_losses = val_level_losses
            state.ema_train_losses = _ema_update(state.ema_train_losses, train_level_losses, alpha=ema_alpha)
            state.ema_val_losses = _ema_update(state.ema_val_losses, val_level_losses, alpha=ema_alpha)
            state.best_val_losses = _min_update(state.best_val_losses, val_level_losses)
            state.prev_weights = weights
            state.best_hard_val_loss = min(state.best_hard_val_loss, hard_val)
            recent_train_losses.append(train_level_losses)
            recent_val_losses.append(val_level_losses)

            metrics = {
                "train_weighted_loss": epoch_weighted_loss / steps_per_epoch,
                "val_weighted_loss": weighted_val,
                "val_hard_loss": hard_val,
                "best_hard_val_loss": state.best_hard_val_loss,
            }
            for j, weight in enumerate(weights):
                metrics[f"weight_{j}"] = float(weight)
            for j, loss in enumerate(train_level_losses):
                metrics[f"train_loss_level_{j}"] = float(loss)
            for j, loss in enumerate(val_level_losses):
                metrics[f"val_loss_level_{j}"] = float(loss)
            mlflow.log_metrics(metrics, step=epoch)
            history.append({"epoch": epoch, **metrics})

            logger.info(
                f"Epoch {epoch} | hard_val={hard_val:.6f} | weighted_val={weighted_val:.6f} "
                f"| weights={[round(w, 3) for w in weights]}"
            )

            if hard_val < best_hard_val - min_delta:
                best_hard_val = hard_val
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        mlflow.log_metric("best_hard_val_loss", float(best_hard_val))
        mlflow.log_metric("epochs_trained", len(history))

        test_loss: float | None = None
        if ds.X_test is not None and ds.y_hard_test is not None:
            model.eval()
            with torch.no_grad():
                test_loss = mse(model(ds.X_test), ds.y_hard_test).item()
            mlflow.log_metric("test_hard_loss", float(test_loss))

        hist_df = pl.DataFrame(history)
        hist_path = output_dir / "weights" / f"{actual_run_name}_trajectory.parquet"
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        hist_df.write_parquet(hist_path)
        mlflow.log_artifact(str(hist_path), artifact_path="schedule")
        weight_plot = _log_weight_plot(history, output_dir, actual_run_name)
        mlflow.log_artifact(str(weight_plot), artifact_path="schedule")

        summary = {
            "run_name": actual_run_name,
            "input_path": str(input_path),
            "schedule_module": loaded_policy.module_path,
            "schedule_entry": loaded_policy.entry_name,
            "candidate_id": candidate_id,
            "llm_note": llm_note,
            "seed": seed,
            "levels": sigma_cols,
            "best_hard_val_loss": float(best_hard_val),
            "final_hard_val_loss": float(history[-1]["val_hard_loss"]),
            "epochs_trained": len(history),
            "test_hard_loss": None if test_loss is None else float(test_loss),
            "trajectory_path": str(hist_path),
            "weight_plot_path": str(weight_plot),
        }
        summary_path = output_dir / "policy_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        mlflow.log_artifact(str(summary_path), artifact_path="summary")
        return summary


if __name__ == "__main__":
    app()
