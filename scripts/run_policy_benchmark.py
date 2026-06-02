#!/usr/bin/env python3
"""Evaluate one schedule policy over multiple seeds and aggregate results."""

from __future__ import annotations

from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import typer

from ma_thesis.config import PROCESSED_DATA_DIR, REPORTS_DIR
from ma_thesis.dataset import main as dataset_main
from ma_thesis.policy_train import main as policy_main

app = typer.Typer(add_completion=False)

BENCHMARK_ROOT = REPORTS_DIR / "benchmarks" / "policy"


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


def _resolve_policy_source(schedule_module: str) -> Path | None:
    spec = importlib.util.find_spec(schedule_module)
    if spec is None or spec.origin is None:
        return None
    origin = Path(spec.origin)
    return origin if origin.exists() else None


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    function: str = typer.Option("ackley"),
    benchmark_id: str = typer.Option(
        f"policy_{_timestamp()}",
        help="Benchmark id used for output files and run names.",
    ),
    experiment_name: str = typer.Option("benchmark-policy", help="MLflow experiment name."),
    input_path: Path | None = typer.Option(None, help="Optional dataset path."),
    regenerate_data: bool = typer.Option(False, help="Regenerate dataset before evaluation."),
    num_samples: int = 20000,
    num_sigmas: int = 3,
    sigma_scale: float = 5.0,
    train_samples: int | None = None,
    noise_ratio: float = 0.02,
    data_seed: int = 42,
    eval_seeds: str = typer.Option("42,43,44,45,46", help="Comma-separated evaluation seeds."),
    schedule_module: str = typer.Option("llm_schedules.baselines"),
    schedule_object: str | None = typer.Option(None),
    schedule_fn: str | None = typer.Option(None),
    schedule_note: str = typer.Option("Multi-seed policy benchmark."),
    candidate_id: str | None = typer.Option(None),
    llm_note: str | None = typer.Option(None),
    num_losses: int = 4,
    history_window: int = 5,
    ema_alpha: float = 0.3,
    model_arch: str = "fourier",
    hidden_dim: int = 16,
    num_blocks: int = 4,
    activation: str = "silu",
    num_layers: int = 4,
    omega_0: float = 30.0,
    num_fourier: int = 128,
    fourier_sigma: float = 10.0,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 30,
    min_delta: float = 1e-5,
    min_train_per_param: float = 10.0,
    log_dataset_artifact: bool = False,
    continue_on_error: bool = typer.Option(True, help="Continue after failed seed runs."),
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
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    source_path = _resolve_policy_source(schedule_module)
    manifest = {
        "benchmark_id": benchmark_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "function": function,
        "input_path": str(resolved_input),
        "data_seed": data_seed,
        "eval_seeds": eval_seed_values,
        "schedule_module": schedule_module,
        "schedule_object": schedule_object,
        "schedule_fn": schedule_fn,
        "schedule_source_path": None if source_path is None else str(source_path),
        "schedule_sha256": _sha256(source_path),
        "candidate_id": candidate_id,
        "llm_note": llm_note,
        "num_losses": num_losses,
        "history_window": history_window,
        "ema_alpha": ema_alpha,
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
    }
    (benchmark_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    seed_rows: list[dict[str, Any]] = []
    for seed in eval_seed_values:
        run_name = f"{benchmark_id}_{function}_seed{seed}"
        seed_dir = benchmark_dir / f"seed_{seed}"
        try:
            summary = policy_main(
                input_path=resolved_input,
                output_dir=seed_dir,
                experiment_name=experiment_name,
                run_name=run_name,
                schedule_module=schedule_module,
                schedule_object=schedule_object,
                schedule_fn=schedule_fn,
                schedule_note=schedule_note,
                candidate_id=candidate_id,
                llm_note=llm_note,
                num_losses=num_losses,
                history_window=history_window,
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
        except Exception as exc:
            error_row = {"seed": seed, "status": "failed", "error": repr(exc)}
            seed_rows.append(error_row)
            (seed_dir / "error.txt").parent.mkdir(parents=True, exist_ok=True)
            (seed_dir / "error.txt").write_text(repr(exc), encoding="utf-8")
            if not continue_on_error:
                raise

    per_seed_path = benchmark_dir / "per_seed_results.jsonl"
    per_seed_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in seed_rows),
        encoding="utf-8",
    )

    ok_rows = [row for row in seed_rows if row["status"] == "ok"]
    best_vals = [float(row["best_hard_val_loss"]) for row in ok_rows]
    final_vals = [float(row["final_hard_val_loss"]) for row in ok_rows]
    test_vals = [float(row["test_hard_loss"]) for row in ok_rows if row["test_hard_loss"] is not None]
    epochs_vals = [int(row["epochs_trained"]) for row in ok_rows]

    aggregate_status = "ok" if ok_rows else "failed"
    aggregate = {
        **manifest,
        "status": aggregate_status,
        "num_successful_runs": len(ok_rows),
        "num_failed_runs": len(seed_rows) - len(ok_rows),
        "mean_best_hard_val_loss": _safe_mean(best_vals),
        "std_best_hard_val_loss": _safe_std(best_vals),
        "mean_final_hard_val_loss": _safe_mean(final_vals),
        "std_final_hard_val_loss": _safe_std(final_vals),
        "mean_test_hard_loss": _safe_mean(test_vals),
        "std_test_hard_loss": _safe_std(test_vals),
        "mean_epochs_trained": _safe_mean([float(v) for v in epochs_vals]),
        "best_seed": None,
    }
    if ok_rows:
        best_row = min(ok_rows, key=lambda row: float(row["best_hard_val_loss"]))
        aggregate["best_seed"] = int(best_row["seed"])
        aggregate["best_run_name"] = str(best_row["run_name"])

    aggregate_path = benchmark_dir / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
