#!/usr/bin/env python3
"""Run curriculum vs single benchmark matrix using sweep-derived hyperparameters.

This script launches paired runs for each function and seed in two regimes:
- equal_epochs: both methods run with the same epoch budget from sweep config
- equal_time: both methods run with the same wall-clock timeout cap

It records an execution manifest and per-run outcomes under reports/benchmarks/.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any

import typer
import yaml

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "configs" / "sweeps"
REPORTS_DIR = ROOT / "reports"
BENCHMARK_DIR = REPORTS_DIR / "benchmarks"
LOGS_DIR = REPORTS_DIR / "logs"
PYTHON_BIN = ROOT / ".venv" / "bin" / "python"

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class SweepSpec:
    function: str
    num_samples: int
    num_sigmas: int
    sigma_scale: float
    sweep_storage_path: str
    study_name: str
    epochs: int
    patience: int
    min_delta: float
    grid_res: int


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object in {path}")
    return payload


def _storage_to_path(storage: str) -> str:
    prefix = "sqlite:///"
    if storage.startswith(prefix):
        return storage[len(prefix) :]
    return storage


def load_sweep_specs() -> dict[str, SweepSpec]:
    specs: dict[str, SweepSpec] = {}
    for path in sorted(SWEEP_DIR.glob("*_sweep.yaml")):
        cfg = _load_yaml(path)
        run = cfg.get("run", cfg)
        if not isinstance(run, dict):
            raise ValueError(f"Config must contain object at 'run': {path}")

        function = str(run["function"])
        study_name = str(run.get("experiment_name", f"sweep-{function.replace('_', '-')}-v1"))
        storage = str(run["sweep_storage"])

        specs[function] = SweepSpec(
            function=function,
            num_samples=int(run["num_samples"]),
            num_sigmas=int(run["num_sigmas"]),
            sigma_scale=float(run["sigma_scale"]),
            sweep_storage_path=_storage_to_path(storage),
            study_name=study_name,
            epochs=int(run["epochs"]),
            patience=int(run["patience"]),
            min_delta=float(run.get("min_delta", 1e-5)),
            grid_res=int(run.get("grid_res", 100)),
        )

    if not specs:
        raise RuntimeError(f"No sweep configs found in {SWEEP_DIR}")
    return specs


def _build_command(
    *,
    method: str,
    spec: SweepSpec,
    seed: int,
    experiment_name: str,
    run_name: str,
    epochs: int,
) -> list[str]:
    cmd = [
        str(PYTHON_BIN),
        "-m",
        "ma_thesis.experiment",
        "run",
        "--method",
        method,
        "--function",
        spec.function,
        "--num-samples",
        str(spec.num_samples),
        "--num-sigmas",
        str(spec.num_sigmas),
        "--sigma-scale",
        str(spec.sigma_scale),
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--patience",
        str(spec.patience),
        "--min-delta",
        str(spec.min_delta),
        "--grid-res",
        str(spec.grid_res),
        "--from-sweep",
        spec.sweep_storage_path,
        "--study-name",
        spec.study_name,
        "--experiment-name",
        experiment_name,
        "--run-name",
        run_name,
    ]
    if method == "single":
        cmd.extend(["--sigma-level", "-1"])
    return cmd


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "benchmark_id",
        "regime",
        "method",
        "function",
        "seed",
        "run_name",
        "status",
        "return_code",
        "elapsed_sec",
        "timeout_sec",
        "started_at",
        "finished_at",
        "command",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@app.command()
def main(
    benchmark_id: str = typer.Option(
        f"curr_vs_single_{_timestamp()}",
        help="Benchmark id used in run names and output files.",
    ),
    experiment_name: str = typer.Option(
        "benchmark-curriculum-vs-single-v1",
        help="MLflow experiment name for benchmark runs.",
    ),
    seeds: str = typer.Option("42,43,44", help="Comma-separated seed list."),
    regimes: str = typer.Option(
        "equal_epochs,equal_time",
        help="Comma-separated regimes: equal_epochs,equal_time",
    ),
    methods: str = typer.Option(
        "single,curriculum",
        help="Comma-separated methods to compare.",
    ),
    equal_time_cap_seconds: int = typer.Option(
        1800,
        help="Wall-clock cap (seconds) used for equal_time regime.",
    ),
    epoch_scale_for_equal_time: float = typer.Option(
        1.0,
        help="Optional multiplier for epochs under equal_time regime.",
    ),
    dry_run: bool = typer.Option(False, help="Print planned commands only."),
) -> None:
    if not PYTHON_BIN.exists():
        raise typer.BadParameter(f"Missing Python interpreter: {PYTHON_BIN}")

    seed_values = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    regime_values = [r.strip() for r in regimes.split(",") if r.strip()]
    method_values = [m.strip() for m in methods.split(",") if m.strip()]

    allowed_regimes = {"equal_epochs", "equal_time"}
    allowed_methods = {"single", "curriculum"}
    unknown_regimes = [r for r in regime_values if r not in allowed_regimes]
    unknown_methods = [m for m in method_values if m not in allowed_methods]
    if unknown_regimes:
        raise typer.BadParameter(f"Unsupported regimes: {unknown_regimes}")
    if unknown_methods:
        raise typer.BadParameter(f"Unsupported methods: {unknown_methods}")

    specs = load_sweep_specs()
    functions = list(specs.keys())

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    run_log_path = LOGS_DIR / f"benchmark_{benchmark_id}.log"
    manifest_path = BENCHMARK_DIR / f"{benchmark_id}_manifest.json"
    runs_path = BENCHMARK_DIR / f"{benchmark_id}_runs.csv"

    manifest = {
        "benchmark_id": benchmark_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_name": experiment_name,
        "functions": functions,
        "seeds": seed_values,
        "regimes": regime_values,
        "methods": method_values,
        "equal_time_cap_seconds": equal_time_cap_seconds,
        "epoch_scale_for_equal_time": epoch_scale_for_equal_time,
        "python_bin": str(PYTHON_BIN),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    rows: list[dict[str, Any]] = []

    total = len(functions) * len(seed_values) * len(regime_values) * len(method_values)
    idx = 0

    with run_log_path.open("a", encoding="utf-8") as run_log:
        run_log.write(f"[{datetime.now().isoformat()}] benchmark {benchmark_id} started\n")

        for seed in seed_values:
            for regime in regime_values:
                for function in functions:
                    spec = specs[function]

                    if regime == "equal_time":
                        regime_epochs = max(1, int(spec.epochs * epoch_scale_for_equal_time))
                        timeout_sec: int | None = equal_time_cap_seconds
                    else:
                        regime_epochs = spec.epochs
                        timeout_sec = None

                    for method in method_values:
                        idx += 1
                        run_name = (
                            f"{benchmark_id}__{regime}__{method}__{function}__seed{seed}"
                        )
                        cmd = _build_command(
                            method=method,
                            spec=spec,
                            seed=seed,
                            experiment_name=experiment_name,
                            run_name=run_name,
                            epochs=regime_epochs,
                        )
                        cmd_text = shlex.join(cmd)
                        print(f"[{idx}/{total}] {cmd_text}")
                        run_log.write(f"[{datetime.now().isoformat()}] CMD {cmd_text}\n")

                        status = "DRY_RUN" if dry_run else "OK"
                        return_code = 0
                        start_dt = datetime.now().astimezone()
                        t0 = time.perf_counter()

                        if not dry_run:
                            try:
                                completed = subprocess.run(
                                    cmd,
                                    cwd=ROOT,
                                    timeout=timeout_sec,
                                    stdout=run_log,
                                    stderr=run_log,
                                    check=False,
                                )
                                return_code = int(completed.returncode)
                                if completed.returncode != 0:
                                    status = "FAILED"
                            except subprocess.TimeoutExpired:
                                status = "TIMEOUT"
                                return_code = 124

                        elapsed_sec = time.perf_counter() - t0
                        finish_dt = datetime.now().astimezone()

                        rows.append(
                            {
                                "benchmark_id": benchmark_id,
                                "regime": regime,
                                "method": method,
                                "function": function,
                                "seed": seed,
                                "run_name": run_name,
                                "status": status,
                                "return_code": return_code,
                                "elapsed_sec": round(elapsed_sec, 3),
                                "timeout_sec": timeout_sec if timeout_sec is not None else "",
                                "started_at": start_dt.isoformat(timespec="seconds"),
                                "finished_at": finish_dt.isoformat(timespec="seconds"),
                                "command": cmd_text,
                            }
                        )
                        _write_csv(runs_path, rows)

        run_log.write(f"[{datetime.now().isoformat()}] benchmark {benchmark_id} finished\n")

    print(f"Manifest: {manifest_path}")
    print(f"Run log: {run_log_path}")
    print(f"Run table: {runs_path}")


if __name__ == "__main__":
    app()
