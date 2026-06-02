"""Train with a fixed parametric schedule over multiple sigma losses."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import torch
import torch.nn as nn

from ma_thesis.common import fit_hidden_dim_to_param_budget
from ma_thesis.config import FIGURES_DIR, PROCESSED_DATA_DIR
from ma_thesis.io import load_dataset_bundle, select_sigma_columns, split_train_val_indices
from ma_thesis.mlflow_utils import log_dataset_reference, log_run_config
from ma_thesis.models import build_model
from ma_thesis.policy_train import (
    PolicyTrainState,
    _ema_update,
    _log_weight_plot,
    _min_update,
    _sigma_indices,
)
from ma_thesis.training_core import (
    build_adamw_cosine,
    compute_steps_per_epoch,
    iter_minibatches,
    shuffle_pair,
)


def _schedule_weights(
    epoch: int,
    total_epochs: int,
    a: torch.Tensor,
    b: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    t = epoch / max(1, total_epochs - 1)
    logits = a + b * t
    return torch.softmax(logits / tau, dim=0)


def main(
    *,
    schedule_params: dict[str, float],
    input_path: Path = PROCESSED_DATA_DIR / "ackley.parquet",
    output_dir: Path = FIGURES_DIR / "param_schedule_train",
    experiment_name: str = "curriculum-optuna-policy",
    run_name: str | None = None,
    schedule_note: str = "Fixed parametric schedule over multiple sigma losses.",
    candidate_id: str | None = None,
    llm_note: str | None = None,
    num_losses: int = 4,
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
    if num_losses < 2:
        raise ValueError("num_losses must be >= 2.")
    if not (0.0 < ema_alpha <= 1.0):
        raise ValueError("ema_alpha must be in (0, 1].")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset_bundle(input_path, device)
    df = ds.df
    func_name = input_path.stem
    sigma_cols = select_sigma_columns(df, num_losses=num_losses)

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

    actual_run_name = run_name or f"{func_name}_param_schedule"
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(experiment_name)

    n_levels = len(sigma_cols)
    tau = float(schedule_params["tau"])
    a = torch.tensor([schedule_params[f"a_{i}"] for i in range(n_levels)], device=device)
    b = torch.tensor([schedule_params[f"b_{i}"] for i in range(n_levels)], device=device)

    state = PolicyTrainState()
    history: list[dict[str, float]] = []
    best_state = None
    best_hard_val = float("inf")
    patience_counter = 0

    with mlflow.start_run(run_name=actual_run_name):
        mlflow.set_tags(
            {
                "experiment_type": "curriculum_param_schedule",
                "function": func_name,
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
                "ema_alpha": ema_alpha,
                "candidate_id": candidate_id or "",
                "llm_note": llm_note or "",
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "patience": patience,
                "min_delta": min_delta,
                **schedule_params,
                **hp,
            }
        )
        log_dataset_reference(input_path, key="input_dataset_path", log_artifact=log_dataset_artifact)
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
                    "params": schedule_params,
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
            },
            filename="param_schedule_config.json",
        )

        for epoch in range(epochs):
            weights_tensor = _schedule_weights(epoch, epochs, a, b, tau)
            weights = tuple(float(v.item()) for v in weights_tensor)

            model.train()
            x_perm, y_perm = shuffle_pair(X_train, Y_train, device=device)
            epoch_weighted_loss = 0.0
            epoch_level_loss_sums = torch.zeros(n_levels, device=device)

            for xb, yb in iter_minibatches(x_perm, y_perm, batch_size=batch_size):
                optimizer.zero_grad(set_to_none=True)
                preds = model(xb)
                losses = torch.stack([mse(preds, yb[:, j : j + 1]) for j in range(n_levels)])
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
                    [mse(pred_val, Y_val[:, j : j + 1]) for j in range(n_levels)]
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

            if hard_val < best_hard_val - min_delta:
                best_hard_val = hard_val
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
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
            "schedule_entry": "param_schedule",
            "candidate_id": candidate_id,
            "llm_note": llm_note,
            "seed": seed,
            "levels": sigma_cols,
            "sigma_indices": list(_sigma_indices(sigma_cols)),
            "best_hard_val_loss": float(best_hard_val),
            "final_hard_val_loss": float(history[-1]["val_hard_loss"]),
            "epochs_trained": len(history),
            "test_hard_loss": None if test_loss is None else float(test_loss),
            "trajectory_path": str(hist_path),
            "weight_plot_path": str(weight_plot),
        }
        summary_path = output_dir / "param_schedule_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        mlflow.log_artifact(str(summary_path), artifact_path="summary")
        return summary
