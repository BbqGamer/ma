#!/usr/bin/env python3
"""Run a simple meta-weighting experiment matrix.

Matrix:
- functions: levy, ackley, eggholder
- numbers of weighted partial losses: 1, 2, 3, 5
- learning rates: 3e-4, 1e-3, 3e-3, 1e-2, 3e-2
- noise ratios: 0.0, 0.1, 0.2, 0.3, 0.5

Workflow:
1. Prepare unique datasets sequentially for every (function, noise_ratio) pair.
2. Launch training runs in parallel against the pre-generated datasets.

Dataset sizing for this benchmark:
- 12k examples stored in the training parquet
- exact internal split of 10k train + 2k val during meta training

All runs use ma_thesis.meta_train through the unified experiment CLI with:
- Fourier model architecture
- parameter budget around 0.1× training samples (min_train_per_param=10)
- SGD for model and meta weights
- momentum=0.9
- mild exponential LR decay
- 100 epochs
- no early stopping
- conservative process/thread limits so the machine stays usable
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import os
import shlex
import subprocess
import time
from typing import Iterable

import typer

ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = ROOT / ".venv" / "bin" / "python"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = REPORTS_DIR / "logs"
STATUS_DIR = REPORTS_DIR / "status"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"

THREAD_LIMIT_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "POLARS_MAX_THREADS",
)

app = typer.Typer(add_completion=False)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _float_token(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p")


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


def _parse_csv_str(values: str) -> list[str]:
    return [v.strip() for v in values.split(",") if v.strip()]


def _parse_csv_int(values: str) -> list[int]:
    return [int(v.strip()) for v in values.split(",") if v.strip()]


def _parse_csv_float(values: str) -> list[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def _available_cpu_ids() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return list(range(os.cpu_count() or 1))


def _cpu_ids_for_slot(slot: int, *, cpus_per_run: int, available_cpu_ids: list[int]) -> list[int]:
    if cpus_per_run <= 0:
        raise ValueError("cpus_per_run must be >= 1")
    if not available_cpu_ids:
        raise ValueError("No CPUs available for affinity assignment")
    start = (slot * cpus_per_run) % len(available_cpu_ids)
    return [available_cpu_ids[(start + i) % len(available_cpu_ids)] for i in range(cpus_per_run)]


def _affinity_preexec_fn(cpu_ids: list[int]):
    def _set_affinity() -> None:
        os.sched_setaffinity(0, cpu_ids)

    return _set_affinity


def _subprocess_env(
    *,
    status_path: Path | None = None,
    num_threads: int,
    num_interop_threads: int,
) -> dict[str, str]:
    env = os.environ.copy()
    for name in THREAD_LIMIT_VARS:
        env[name] = str(num_threads)
    env["MA_TORCH_NUM_THREADS"] = str(num_threads)
    env["MA_TORCH_NUM_INTEROP_THREADS"] = str(num_interop_threads)
    if status_path is not None:
        env["MA_META_STATUS_PATH"] = str(status_path)
    return env


def _build_prepare_command(
    *,
    function: str,
    noise_ratio: float,
    num_samples: int,
    num_sigmas: int,
    sigma_scale: float,
    train_samples: int,
    seed: int,
    dataset_path: Path,
) -> list[str]:
    return [
        str(PYTHON_BIN),
        "-m",
        "ma_thesis.experiment",
        "prepare-data",
        "--function",
        function,
        "--output-dir",
        str(dataset_path.parent),
        "--num-samples",
        str(num_samples),
        "--num-sigmas",
        str(num_sigmas),
        "--sigma-scale",
        str(sigma_scale),
        "--train-samples",
        str(train_samples),
        "--noise-ratio",
        str(noise_ratio),
        "--seed",
        str(seed),
        "--versioned-name",
    ]


def _build_train_command(
    *,
    function: str,
    input_path: Path,
    num_losses: int,
    lr_model: float,
    lr_meta: float,
    noise_ratio: float,
    experiment_name: str,
    run_name: str,
    num_sigmas: int,
    epochs: int,
    batch_size: int,
    model_arch: str,
    min_train_per_param: float,
    inner_steps: int,
    momentum: float,
    lr_decay_gamma: float,
    grad_clip_norm: float | None,
    seed: int,
    meta_val_samples: int,
) -> list[str]:
    return [
        str(PYTHON_BIN),
        "-m",
        "ma_thesis.experiment",
        "run",
        "--method",
        "meta",
        "--function",
        function,
        "--input-path",
        str(input_path),
        "--num-sigmas",
        str(num_sigmas),
        "--noise-ratio",
        str(noise_ratio),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--model-arch",
        model_arch,
        "--min-train-per-param",
        str(min_train_per_param),
        "--lr-model",
        str(lr_model),
        "--lr-meta",
        str(lr_meta),
        "--momentum",
        str(momentum),
        "--lr-decay-gamma",
        str(lr_decay_gamma),
        "--grad-clip-norm",
        str(grad_clip_norm),
        "--inner-steps",
        str(inner_steps),
        "--meta-num-losses",
        str(num_losses),
        "--meta-val-samples",
        str(meta_val_samples),
        "--seed",
        str(seed),
        "--experiment-name",
        experiment_name,
        "--run-name",
        run_name,
    ]


def _function_experiment_name(experiment_name: str, function: str) -> str:
    return f"{experiment_name}-{function}"


def _iter_matrix(
    seeds: Iterable[int],
    functions: Iterable[str],
    num_losses_values: Iterable[int],
    lrs: Iterable[float],
    noise_ratios: Iterable[float],
) -> Iterable[tuple[int, str, int, float, float]]:
    for seed in seeds:
        for function in functions:
            for num_losses in num_losses_values:
                for lr in lrs:
                    for noise_ratio in noise_ratios:
                        yield seed, function, num_losses, lr, noise_ratio


def _iter_dataset_jobs(
    seeds: Iterable[int],
    functions: Iterable[str],
    noise_ratios: Iterable[float],
    *,
    num_samples: int,
    num_sigmas: int,
    sigma_scale: float,
) -> Iterable[tuple[int, str, float, Path]]:
    for seed in seeds:
        for function in functions:
            for noise_ratio in noise_ratios:
                yield (
                    seed,
                    function,
                    noise_ratio,
                    PROCESSED_DATA_DIR
                    / _dataset_train_filename(
                        function=function,
                        num_samples=num_samples,
                        num_sigmas=num_sigmas,
                        sigma_scale=sigma_scale,
                        noise_ratio=noise_ratio,
                        seed=seed,
                    ),
                )


def _run_sequential_job(
    *,
    label: str,
    cmd: list[str],
    log_path: Path,
    dry_run: bool,
) -> int:
    cmd_text = shlex.join(cmd)
    print(f"{label}: {cmd_text}")
    if dry_run:
        return 0

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[{datetime.now().isoformat()}] START {cmd_text}\n")
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            env=_subprocess_env(num_threads=1, num_interop_threads=1),
            stdout=log_file,
            stderr=log_file,
            check=False,
        )
        log_file.write(f"[{datetime.now().isoformat()}] END rc={completed.returncode}\n")
        return int(completed.returncode)


def _load_status(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _progress_bar(progress: float, width: int = 18) -> str:
    clamped = max(0.0, min(1.0, progress))
    filled = int(round(clamped * width))
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def _short_run_name(run_name: str, limit: int = 42) -> str:
    if len(run_name) <= limit:
        return run_name
    return f"...{run_name[-(limit - 3):]}"


def _render_dashboard(active: list[dict[str, object]], total_jobs: int, completed_jobs: int) -> None:
    lines = [
        "\n=== meta-weighting progress ===",
        f"completed: {completed_jobs}/{total_jobs} | active: {len(active)}",
    ]
    if not active:
        if completed_jobs >= total_jobs:
            lines.append("all runs finished")
        else:
            lines.append("waiting for runs to launch...")
    for entry in active:
        run_name = str(entry["run_name"])
        status_path = entry["status_path"]
        assert isinstance(status_path, Path)
        status = _load_status(status_path)
        if status is None:
            lines.append(f"{_short_run_name(run_name)} | starting...")
            continue

        progress = float(status.get("progress", 0.0))
        epoch = int(status.get("epoch", 0))
        epochs_total = int(status.get("epochs_total", 0))
        stage = str(status.get("stage", "starting"))
        val_loss = status.get("val_loss")
        train_loss = status.get("train_loss")
        lr_model = status.get("lr_model")
        bar = _progress_bar(progress)
        metric_bits = []
        if isinstance(train_loss, int | float):
            metric_bits.append(f"train={train_loss:.4f}")
        if isinstance(val_loss, int | float):
            metric_bits.append(f"val={val_loss:.4f}")
        if isinstance(lr_model, int | float):
            metric_bits.append(f"lr={lr_model:.2e}")
        metrics = " | ".join(metric_bits)
        lines.append(
            f"{_short_run_name(run_name)} | {bar} {epoch:>3}/{epochs_total:<3} | {stage:<10}"
            + (f" | {metrics}" if metrics else "")
        )

    print("\033[2J\033[H" + "\n".join(lines), end="", flush=True)


def _launch_parallel_jobs(
    *,
    jobs: list[tuple[int, str, list[str], str]],
    max_parallel: int,
    cpus_per_run: int,
    num_interop_threads: int,
    benchmark_log_file,
) -> tuple[int, list[tuple[int, str, int]]]:
    active: list[dict[str, object]] = []
    next_job = 0
    completed_jobs = 0
    failures: list[tuple[int, str, int]] = []
    available_cpu_ids = _available_cpu_ids()

    while next_job < len(jobs) or active:
        while next_job < len(jobs) and len(active) < max_parallel:
            idx, run_name, cmd, cmd_text = jobs[next_job]
            run_log_path = LOGS_DIR / f"{run_name}.log"
            status_path = STATUS_DIR / f"{run_name}.json"
            if status_path.exists():
                status_path.unlink()
            run_log_handle = run_log_path.open("w", encoding="utf-8")
            run_log_handle.write(f"[{datetime.now().isoformat()}] START {cmd_text}\n")
            cpu_ids = _cpu_ids_for_slot(
                len(active),
                cpus_per_run=cpus_per_run,
                available_cpu_ids=available_cpu_ids,
            )
            process = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=_subprocess_env(
                    status_path=status_path,
                    num_threads=cpus_per_run,
                    num_interop_threads=num_interop_threads,
                ),
                stdout=run_log_handle,
                stderr=run_log_handle,
                preexec_fn=_affinity_preexec_fn(cpu_ids),
            )
            active.append(
                {
                    "slot": slot,
                    "idx": idx,
                    "run_name": run_name,
                    "process": process,
                    "log_handle": run_log_handle,
                    "log_path": run_log_path,
                    "status_path": status_path,
                    "cpu_ids": cpu_ids,
                }
            )
            benchmark_log_file.write(
                f"[{datetime.now().isoformat()}] LAUNCHED [{idx}/{len(jobs)}] "
                f"pid={process.pid} run={run_name} cpus={cpu_ids} log={run_log_path}\n"
            )
            print(
                f"launched [{idx}/{len(jobs)}] pid={process.pid} run={run_name} "
                f"slot={slot} cpus={cpu_ids} ({len(active)}/{max_parallel} active)"
            )
            next_job += 1

        _render_dashboard(active, len(jobs), completed_jobs)
        time.sleep(1.0)
        still_active: list[dict[str, object]] = []
        for entry in active:
            process = entry["process"]
            assert isinstance(process, subprocess.Popen)
            rc = process.poll()
            if rc is None:
                still_active.append(entry)
                continue

            log_handle = entry["log_handle"]
            assert hasattr(log_handle, "write")
            log_handle.write(f"[{datetime.now().isoformat()}] END rc={rc}\n")
            log_handle.close()

            run_name = entry["run_name"]
            idx = entry["idx"]
            benchmark_log_file.write(
                f"[{datetime.now().isoformat()}] FINISHED [{idx}/{len(jobs)}] "
                f"rc={rc} run={run_name} log={entry['log_path']}\n"
            )
            completed_jobs += 1
            status_path = entry["status_path"]
            assert isinstance(status_path, Path)
            if rc != 0:
                failures.append((int(idx), str(run_name), int(rc)))
                status_path.write_text(
                    json.dumps(
                        {
                            "stage": "failed",
                            "run_name": run_name,
                            "progress": 0.0,
                            "return_code": int(rc),
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                benchmark_log_file.write(
                    f"[{datetime.now().isoformat()}] FAILURE recorded; continuing benchmark "
                    f"run={run_name} rc={rc}\n"
                )
            print(f"finished [{idx}/{len(jobs)}] rc={rc} run={run_name}")

        active = still_active

    _render_dashboard(active, len(jobs), completed_jobs)
    print()
    return (0 if not failures else 1), failures


@app.command()
def main(
    benchmark_id: str = typer.Option(
        f"meta_weighting_{_timestamp()}",
        help="Benchmark id used in run names and logs.",
    ),
    experiment_name: str = typer.Option(
        "meta-weighting-v1",
        help="MLflow experiment name.",
    ),
    functions: str = typer.Option(
        "levy,ackley,eggholder",
        help="Comma-separated function list.",
    ),
    num_losses_values: str = typer.Option(
        "1,2,3,5",
        help="Comma-separated numbers of weighted partial losses. Use 1 for the baseline final-loss-only run.",
    ),
    learning_rates: str = typer.Option(
        "3e-4,1e-3,3e-3,1e-2,3e-2",
        help="Comma-separated SGD learning rates used for both model and meta updates.",
    ),
    noise_ratios: str = typer.Option(
        "0.0,0.02,0.05,0.1,0.2",
        help="Comma-separated dataset noise ratios applied before smoothing.",
    ),
    num_samples: int = typer.Option(
        12000,
        help="Number of samples generated per dataset. Use 12000 for 10k train + 2k val inside meta training.",
    ),
    train_samples: int = typer.Option(
        12000,
        help="Samples stored in the *_train split consumed by meta training.",
    ),
    meta_val_samples: int = typer.Option(
        2000,
        help="Exact validation set size carved out inside meta training.",
    ),
    num_sigmas: int = typer.Option(
        5,
        help="Number of smoothing levels to generate. Must be >= max(num_losses_values).",
    ),
    sigma_scale: float = typer.Option(5.0, help="Maximum sigma scale for dataset smoothing."),
    epochs: int = typer.Option(200, help="Fixed epoch budget for every run."),
    batch_size: int = typer.Option(128, help="Batch size."),
    model_arch: str = typer.Option("fourier", help="Model architecture for all runs."),
    min_train_per_param: float = typer.Option(
        10.0,
        help="Parameter budget as train-samples-per-parameter (10 => ~0.1 params per train sample).",
    ),
    inner_steps: int = typer.Option(10, help="Inner-loop updates per epoch."),
    momentum: float = typer.Option(0.9, help="SGD momentum."),
    lr_decay_gamma: float = typer.Option(0.999, help="Mild exponential LR decay."),
    grad_clip_norm: float | None = typer.Option(1.0, help="Gradient clipping norm for model and meta updates."),
    seeds: str = typer.Option("42,666,777,888,999", help="Comma-separated dataset/train split seeds."),
    cpus_per_run: int = typer.Option(2, help="CPU cores reserved per training subprocess."),
    num_interop_threads: int = typer.Option(1, help="Torch inter-op threads per subprocess."),
    max_parallel: int = typer.Option(
        6,
        help="Maximum number of concurrent training runs. Safe default is 2 for laptop use.",
    ),
    force_prepare: bool = typer.Option(
        False,
        help="Regenerate datasets even if the target parquet file already exists.",
    ),
    dry_run: bool = typer.Option(False, help="Print commands without executing them."),
) -> None:
    if not PYTHON_BIN.exists():
        raise typer.BadParameter(f"Missing Python interpreter: {PYTHON_BIN}")

    function_values = _parse_csv_str(functions)
    num_loss_values = _parse_csv_int(num_losses_values)
    lr_values = _parse_csv_float(learning_rates)
    noise_ratio_values = _parse_csv_float(noise_ratios)
    seed_values = _parse_csv_int(seeds)

    if not function_values:
        raise typer.BadParameter("At least one function is required.")
    if not num_loss_values:
        raise typer.BadParameter("At least one num_losses value is required.")
    if not lr_values:
        raise typer.BadParameter("At least one learning rate is required.")
    if not noise_ratio_values:
        raise typer.BadParameter("At least one noise ratio is required.")
    if not seed_values:
        raise typer.BadParameter("At least one seed is required.")
    if max(num_loss_values) > num_sigmas:
        raise typer.BadParameter("num_sigmas must be >= max(num_losses_values).")
    if max_parallel < 1:
        raise typer.BadParameter("max_parallel must be >= 1.")
    if cpus_per_run < 1:
        raise typer.BadParameter("cpus_per_run must be >= 1.")
    if num_interop_threads < 1:
        raise typer.BadParameter("num_interop_threads must be >= 1.")
    if train_samples < 1 or train_samples > num_samples:
        raise typer.BadParameter("train_samples must satisfy 1 <= train_samples <= num_samples.")
    if meta_val_samples < 1 or meta_val_samples >= train_samples:
        raise typer.BadParameter(
            "meta_val_samples must satisfy 1 <= meta_val_samples < train_samples."
        )

    available_cpu_ids = _available_cpu_ids()
    requested_cpus = max_parallel * cpus_per_run
    if requested_cpus > len(available_cpu_ids):
        print(
            "Warning: requested CPU budget "
            f"({requested_cpus} = {max_parallel}x{cpus_per_run}) exceeds available CPUs "
            f"({len(available_cpu_ids)}). Affinity sets will wrap and cores will be shared."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{benchmark_id}.log"

    dataset_jobs = list(
        _iter_dataset_jobs(
            seed_values,
            function_values,
            noise_ratio_values,
            num_samples=num_samples,
            num_sigmas=num_sigmas,
            sigma_scale=sigma_scale,
        )
    )
    matrix = list(
        _iter_matrix(seed_values, function_values, num_loss_values, lr_values, noise_ratio_values)
    )

    train_jobs: list[tuple[int, str, list[str], str]] = []
    for idx, (seed, function, num_losses, lr, noise_ratio) in enumerate(matrix, start=1):
        dataset_path = PROCESSED_DATA_DIR / _dataset_train_filename(
            function=function,
            num_samples=num_samples,
            num_sigmas=num_sigmas,
            sigma_scale=sigma_scale,
            noise_ratio=noise_ratio,
            seed=seed,
        )
        run_name = (
            f"{benchmark_id}__seed{seed}__{function}__losses{num_losses}"
            f"__lr{lr:g}__noise{noise_ratio:g}"
        )
        cmd = _build_train_command(
            function=function,
            input_path=dataset_path,
            num_losses=num_losses,
            lr_model=lr,
            lr_meta=lr,
            noise_ratio=noise_ratio,
            experiment_name=experiment_name,
            run_name=run_name,
            num_sigmas=num_sigmas,
            epochs=epochs,
            batch_size=batch_size,
            model_arch=model_arch,
            min_train_per_param=min_train_per_param,
            inner_steps=inner_steps,
            momentum=momentum,
            lr_decay_gamma=lr_decay_gamma,
            grad_clip_norm=grad_clip_norm,
            seed=seed,
            meta_val_samples=meta_val_samples,
        )
        train_jobs.append((idx, run_name, cmd, shlex.join(cmd)))

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"[{datetime.now().isoformat()}] benchmark {benchmark_id} started "
            f"(max_parallel={max_parallel}, cpus_per_run={cpus_per_run}, "
            f"num_interop_threads={num_interop_threads})\n"
        )

        log_file.write(f"[{datetime.now().isoformat()}] dataset preparation phase started\n")
        for seed, function, noise_ratio, dataset_path in dataset_jobs:
            prepare_cmd = _build_prepare_command(
                function=function,
                noise_ratio=noise_ratio,
                num_samples=num_samples,
                num_sigmas=num_sigmas,
                sigma_scale=sigma_scale,
                train_samples=train_samples,
                seed=seed,
                dataset_path=dataset_path,
            )
            dataset_log_path = LOGS_DIR / (
                f"{benchmark_id}__prepare__seed{seed}__{function}__noise{noise_ratio:g}.log"
            )
            if dataset_path.exists() and not force_prepare:
                msg = f"SKIP prepare existing dataset seed={seed} {dataset_path}"
                print(msg)
                log_file.write(f"[{datetime.now().isoformat()}] {msg}\n")
                continue

            rc = _run_sequential_job(
                label=f"prepare seed={seed} {function} noise={noise_ratio:g}",
                cmd=prepare_cmd,
                log_path=dataset_log_path,
                dry_run=dry_run,
            )
            log_file.write(
                f"[{datetime.now().isoformat()}] PREPARE seed={seed} function={function} "
                f"noise={noise_ratio:g} rc={rc} log={dataset_log_path}\n"
            )
            if rc != 0:
                log_file.write(
                    f"[{datetime.now().isoformat()}] benchmark {benchmark_id} failed during dataset preparation rc={rc}\n"
                )
                raise typer.Exit(code=rc)

        log_file.write(f"[{datetime.now().isoformat()}] dataset preparation phase finished\n")
        log_file.write(f"[{datetime.now().isoformat()}] training phase started\n")

        for idx, _, _, cmd_text in train_jobs:
            print(f"[{idx}/{len(train_jobs)}] {cmd_text}")
            log_file.write(f"[{datetime.now().isoformat()}] CMD {cmd_text}\n")

        if dry_run:
            log_file.write(f"[{datetime.now().isoformat()}] benchmark {benchmark_id} dry-run\n")
            print(f"Log: {log_path}")
            return

        rc, failures = _launch_parallel_jobs(
            jobs=train_jobs,
            max_parallel=max_parallel,
            cpus_per_run=cpus_per_run,
            num_interop_threads=num_interop_threads,
            benchmark_log_file=log_file,
        )
        if failures:
            log_file.write(
                f"[{datetime.now().isoformat()}] benchmark {benchmark_id} finished with "
                f"{len(failures)} failed runs\n"
            )
            for idx, run_name, fail_rc in failures:
                log_file.write(
                    f"[{datetime.now().isoformat()}] FAILED [{idx}/{len(train_jobs)}] "
                    f"rc={fail_rc} run={run_name}\n"
                )
            print("Failed runs:")
            for idx, run_name, fail_rc in failures:
                print(f"- [{idx}/{len(train_jobs)}] rc={fail_rc} {run_name}")
            raise typer.Exit(code=rc)

        log_file.write(f"[{datetime.now().isoformat()}] benchmark {benchmark_id} finished\n")

    print(f"Log: {log_path}")


if __name__ == "__main__":
    app()
