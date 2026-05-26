#!/usr/bin/env python3
"""Evaluate one CIFAR-100 easy-vs-hard schedule policy over one or more seeds."""

from __future__ import annotations

from datetime import datetime
import json
from statistics import mean, pstdev
from typing import Any

import typer

from ma_thesis.cifar100_lightning import CIFAR100_REPORTS, run_weighted_cifar100_training
from ma_thesis.cifar100_schedule import load_task_schedule_policy

app = typer.Typer(add_completion=False)
BENCHMARK_ROOT = CIFAR100_REPORTS / "policy"


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
    benchmark_id: str = typer.Option(f"cifar100_policy_{_timestamp()}"),
    schedule_module: str = typer.Option(...),
    schedule_object: str | None = typer.Option(None),
    schedule_fn: str | None = typer.Option(None),
    eval_seeds: str = typer.Option("42"),
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
        help="Enable early stopping. Leave off for fixed-budget fair comparisons.",
    ),
    accelerator: str = "auto",
    devices: str = "1",
    precision: str = "16-mixed",
) -> None:
    seeds = _parse_csv_int(eval_seeds)
    if not seeds:
        raise typer.BadParameter("eval_seeds must contain at least one seed.")

    loaded = load_task_schedule_policy(
        schedule_module,
        object_name=schedule_object,
        function_name=schedule_fn,
    )
    benchmark_dir = BENCHMARK_ROOT / benchmark_id
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        run_dir = benchmark_dir / f"seed_{seed}"
        summary = run_weighted_cifar100_training(
            output_dir=run_dir,
            run_name=f"{benchmark_id}_seed{seed}",
            schedule_policy=loaded.policy,
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
        rows.append({"status": "ok", **summary})

    per_seed_path = benchmark_dir / "per_seed_results.jsonl"
    per_seed_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    best_vals = [float(row["best_hard_val_loss"]) for row in rows]
    final_vals = [float(row["final_hard_val_loss"]) for row in rows]
    best_accs = [float(row["best_hard_val_acc"]) for row in rows]
    aggregate = {
        "benchmark_id": benchmark_id,
        "schedule_module": loaded.module_path,
        "schedule_entry": loaded.entry_name,
        "eval_seeds": seeds,
        "status": "ok",
        "num_successful_runs": len(rows),
        "mean_best_hard_val_loss": _safe_mean(best_vals),
        "std_best_hard_val_loss": _safe_std(best_vals),
        "mean_final_hard_val_loss": _safe_mean(final_vals),
        "std_final_hard_val_loss": _safe_std(final_vals),
        "mean_best_hard_val_acc": _safe_mean(best_accs),
        "std_best_hard_val_acc": _safe_std(best_accs),
        "best_seed": int(min(rows, key=lambda row: row["best_hard_val_loss"])["seed"]),
        "use_early_stopping": bool(use_early_stopping),
        "max_epochs": int(max_epochs),
    }
    aggregate_path = benchmark_dir / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
