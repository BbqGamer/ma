#!/usr/bin/env python3
"""Run a multi-seed Optuna search over parametric schedules for fair comparison with LLM sweeps."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import optuna
import typer

from ma_thesis.config import PROCESSED_DATA_DIR, REPORTS_DIR
from ma_thesis.dataset import main as dataset_main
from ma_thesis.param_schedule_train import main as param_schedule_main

app = typer.Typer(add_completion=False)

BENCHMARK_ROOT = REPORTS_DIR / "benchmarks" / "optuna_schedule"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _float_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _dataset_train_filename(
    function: str,
    num_samples: int,
    num_sigmas: int,
    sigma_scale: float,
    noise_ratio: float,
    seed: int,
) -> str:
    return (
        f"{function}_n{num_samples}_k{num_sigmas}"
        f"_ss{_float_token(sigma_scale)}"
        f"_nr{_float_token(noise_ratio)}"
        f"_seed{seed}_train.parquet"
    )


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


def _trial_schedule_params(trial: optuna.Trial, n_levels: int) -> dict[str, float]:
    params: dict[str, float] = {"tau": trial.suggest_float("tau", 0.5, 2.5)}
    for i in range(n_levels):
        params[f"a_{i}"] = trial.suggest_float(f"a_{i}", -5.0, 5.0)
        params[f"b_{i}"] = trial.suggest_float(f"b_{i}", -10.0, 10.0)
    return params


@app.command()
def main(
    function: str = typer.Option("eggholder"),
    benchmark_id: str = typer.Option(
        f"optuna_schedule_{_timestamp()}",
        help="Benchmark id used for output files and run names.",
    ),
    experiment_name: str = typer.Option(
        "optuna-schedule-search-benchmark",
        help="MLflow experiment name.",
    ),
    input_path: Path | None = typer.Option(None, help="Optional dataset path."),
    regenerate_data: bool = typer.Option(False, help="Regenerate dataset before search."),
    num_samples: int = 20000,
    num_sigmas: int = 5,
    sigma_scale: float = 5.0,
    train_samples: int | None = None,
    noise_ratio: float = 0.02,
    data_seed: int = 42,
    eval_seeds: str = typer.Option("42,43,44", help="Comma-separated seed list."),
    n_trials: int = 20,
    search_seed: int = 42,
    num_losses: int = 4,
    ema_alpha: float = 0.3,
    model_arch: str = "fourier",
    hidden_dim: int = 16,
    num_blocks: int = 4,
    activation: str = "silu",
    num_layers: int = 4,
    omega_0: float = 30.0,
    num_fourier: int = 128,
    fourier_sigma: float = 10.0,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 20,
    min_delta: float = 1e-5,
    min_train_per_param: float = 10.0,
    log_dataset_artifact: bool = False,
) -> None:
    eval_seed_values = _parse_csv_int(eval_seeds)
    if not eval_seed_values:
        raise typer.BadParameter("eval_seeds must contain at least one seed.")

    effective_train_samples = num_samples if train_samples is None else train_samples
    default_dataset = PROCESSED_DATA_DIR / _dataset_train_filename(
        function=function,
        num_samples=num_samples,
        num_sigmas=num_sigmas,
        sigma_scale=sigma_scale,
        noise_ratio=noise_ratio,
        seed=data_seed,
    )
    resolved_input = input_path or default_dataset
    if regenerate_data or not resolved_input.exists():
        dataset_main(
            function=function,
            output_dir=resolved_input.parent,
            num_samples=num_samples,
            num_sigmas=num_sigmas,
            sigma_scale=sigma_scale,
            train_samples=effective_train_samples,
            noise_ratio=noise_ratio,
            seed=data_seed,
            output_name=resolved_input.name,
        )

    benchmark_dir = BENCHMARK_ROOT / benchmark_id
    trials_dir = benchmark_dir / "trials"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{benchmark_dir / 'study.db'}"

    manifest = {
        "benchmark_id": benchmark_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "optuna_param_schedule",
        "function": function,
        "input_path": str(resolved_input),
        "data_seed": data_seed,
        "eval_seeds": eval_seed_values,
        "n_trials": n_trials,
        "search_seed": search_seed,
        "num_losses": num_losses,
        "model_arch": model_arch,
        "hidden_dim": hidden_dim,
        "num_blocks": num_blocks,
        "activation": activation,
        "num_layers": num_layers,
        "omega_0": omega_0,
        "num_fourier": num_fourier,
        "fourier_sigma": fourier_sigma,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "patience": patience,
        "min_delta": min_delta,
        "min_train_per_param": min_train_per_param,
        "storage": storage,
    }
    (benchmark_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        schedule_params = _trial_schedule_params(trial, num_losses)
        trial_id = f"trial_{trial.number:03d}"
        trial_dir = trials_dir / trial_id
        seed_rows: list[dict[str, Any]] = []
        for seed in eval_seed_values:
            seed_dir = trial_dir / f"seed_{seed}"
            summary = param_schedule_main(
                schedule_params=schedule_params,
                input_path=resolved_input,
                output_dir=seed_dir,
                experiment_name=experiment_name,
                run_name=f"{benchmark_id}_{trial_id}_{function}_seed{seed}",
                schedule_note="Optuna parametric schedule trial.",
                candidate_id=trial_id,
                llm_note=f"Optuna trial {trial.number}",
                num_losses=num_losses,
                ema_alpha=ema_alpha,
                seed=seed,
                model_arch=model_arch,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                activation=activation,
                num_layers=num_layers,
                omega_0=omega_0,
                num_fourier=num_fourier,
                fourier_sigma=fourier_sigma,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
                min_delta=min_delta,
                min_train_per_param=min_train_per_param,
                log_dataset_artifact=log_dataset_artifact,
            )
            seed_rows.append({"seed": seed, "status": "ok", **summary})

        per_seed_path = trial_dir / "per_seed_results.jsonl"
        per_seed_path.parent.mkdir(parents=True, exist_ok=True)
        per_seed_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in seed_rows),
            encoding="utf-8",
        )

        best_vals = [float(row["best_hard_val_loss"]) for row in seed_rows]
        final_vals = [float(row["final_hard_val_loss"]) for row in seed_rows]
        epochs_vals = [int(row["epochs_trained"]) for row in seed_rows]
        aggregate = {
            "trial_number": trial.number,
            "candidate_id": trial_id,
            "schedule_params": schedule_params,
            "mean_best_hard_val_loss": _safe_mean(best_vals),
            "std_best_hard_val_loss": _safe_std(best_vals),
            "mean_final_hard_val_loss": _safe_mean(final_vals),
            "std_final_hard_val_loss": _safe_std(final_vals),
            "mean_epochs_trained": _safe_mean([float(v) for v in epochs_vals]),
            "num_successful_runs": len(seed_rows),
            "num_failed_runs": 0,
            "status": "ok",
        }
        (trial_dir / "aggregate.json").write_text(
            json.dumps(aggregate, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        trial.set_user_attr("aggregate_path", str(trial_dir / "aggregate.json"))
        trial.set_user_attr("mean_best_hard_val_loss", aggregate["mean_best_hard_val_loss"])
        trial_rows.append({**aggregate, **manifest})
        return float(aggregate["mean_best_hard_val_loss"])

    study = optuna.create_study(
        study_name=benchmark_id,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=search_seed),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    best_aggregate_path = Path(str(best.user_attrs["aggregate_path"]))
    best_aggregate = json.loads(best_aggregate_path.read_text(encoding="utf-8"))

    overall = {
        **manifest,
        "status": "ok",
        "best_trial_number": int(best.number),
        "best_schedule_params": best.params,
        **best_aggregate,
    }
    (benchmark_dir / "aggregate.json").write_text(
        json.dumps(overall, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (benchmark_dir / "all_trials.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in trial_rows),
        encoding="utf-8",
    )
    print(json.dumps(overall, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
