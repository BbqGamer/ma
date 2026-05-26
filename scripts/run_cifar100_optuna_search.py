#!/usr/bin/env python3
"""Run a simple Optuna search for CIFAR-100 easy-vs-hard weighting."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import optuna
import typer

from ma_thesis.cifar100_lightning import (
    CIFAR100_REPORTS,
    ParametricEasyHardPolicy,
    run_weighted_cifar100_training,
)

app = typer.Typer(add_completion=False)
SEARCH_ROOT = CIFAR100_REPORTS / "optuna"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_csv_int(values: str) -> list[int]:
    return [int(v.strip()) for v in values.split(",") if v.strip()]


def _safe_mean(values: list[float]) -> float | None:
    return None if not values else float(mean(values))


def _safe_std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(pstdev(values))


@app.command()
def main(
    study_name: str = typer.Option(f"cifar100_optuna_{_timestamp()}"),
    eval_seeds: str = typer.Option("42"),
    n_trials: int = 10,
    search_seed: int = 42,
    batch_size: int = 256,
    num_workers: int = 8,
    max_epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    ema_alpha: float = 0.3,
    history_window: int = 5,
    val_fraction: float = 0.1,
    patience: int = 8,
    min_delta: float = 1e-4,
    use_early_stopping: bool = typer.Option(
        False,
        help="Enable early stopping. Leave off for fixed-budget fair search.",
    ),
    accelerator: str = "auto",
    devices: str = "1",
    precision: str = "16-mixed",
) -> None:
    seeds = _parse_csv_int(eval_seeds)
    if not seeds:
        raise typer.BadParameter("eval_seeds must contain at least one seed.")

    study_dir = SEARCH_ROOT / study_name
    trials_dir = study_dir / "trials"
    study_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{study_dir / 'study.db'}"
    trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        policy = ParametricEasyHardPolicy(
            tau=trial.suggest_float("tau", 0.2, 2.0),
            a=trial.suggest_float("a", -4.0, 4.0),
            b=trial.suggest_float("b", -6.0, 6.0),
            c=trial.suggest_float("c", -6.0, 10.0),
            d=trial.suggest_float("d", -6.0, 6.0),
        )
        trial_id = f"trial_{trial.number:03d}"
        trial_dir = trials_dir / trial_id
        seed_rows: list[dict[str, Any]] = []
        for seed in seeds:
            summary = run_weighted_cifar100_training(
                output_dir=trial_dir / f"seed_{seed}",
                run_name=f"{study_name}_{trial_id}_seed{seed}",
                schedule_policy=policy,
                seed=seed,
                batch_size=batch_size,
                num_workers=num_workers,
                max_epochs=max_epochs,
                lr=lr,
                weight_decay=weight_decay,
                ema_alpha=ema_alpha,
                history_window=history_window,
                val_fraction=val_fraction,
                patience=patience,
                min_delta=min_delta,
                use_early_stopping=use_early_stopping,
                accelerator=accelerator,
                devices=int(devices) if devices.isdigit() else devices,
                precision=precision,
            )
            seed_rows.append({"status": "ok", **summary})

        (trial_dir / "per_seed_results.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in seed_rows),
            encoding="utf-8",
        )
        best_vals = [float(row["best_hard_val_loss"]) for row in seed_rows]
        final_vals = [float(row["final_hard_val_loss"]) for row in seed_rows]
        aggregate = {
            "trial_number": int(trial.number),
            "candidate_id": trial_id,
            "schedule_params": trial.params,
            "mean_best_hard_val_loss": _safe_mean(best_vals),
            "std_best_hard_val_loss": _safe_std(best_vals),
            "mean_final_hard_val_loss": _safe_mean(final_vals),
            "std_final_hard_val_loss": _safe_std(final_vals),
            "status": "ok",
        }
        (trial_dir / "aggregate.json").write_text(
            json.dumps(aggregate, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        trial_rows.append(aggregate)
        trial.set_user_attr("aggregate_path", str(trial_dir / "aggregate.json"))
        return float(aggregate["mean_best_hard_val_loss"])

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=search_seed),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    best_aggregate = json.loads(Path(str(best.user_attrs["aggregate_path"])).read_text(encoding="utf-8"))
    overall = {
        "study_name": study_name,
        "n_trials": n_trials,
        "eval_seeds": seeds,
        "use_early_stopping": bool(use_early_stopping),
        "max_epochs": int(max_epochs),
        "best_trial_number": int(best.number),
        "best_schedule_params": best.params,
        **best_aggregate,
    }
    (study_dir / "aggregate.json").write_text(
        json.dumps(overall, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (study_dir / "all_trials.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in trial_rows),
        encoding="utf-8",
    )
    print(json.dumps(overall, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
