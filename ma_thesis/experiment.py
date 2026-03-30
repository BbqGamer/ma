"""Unified experiment CLI.

Single entrypoint for:
- dataset generation
- baseline training (single/curriculum)
- meta-curriculum training
- Optuna sweeps
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shlex
import sys
from typing import Any

from loguru import logger
import typer
import yaml

from ma_thesis.config import PROCESSED_DATA_DIR, REPORTS_DIR
from ma_thesis.dataset import FUNCTIONS
from ma_thesis.dataset import main as dataset_main
from ma_thesis.meta_train import main as meta_main
from ma_thesis.schedule_sweep import main as schedule_sweep_main
from ma_thesis.sweep import main as sweep_main
from ma_thesis.train import main as train_main

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _normalize_optioninfo(value: Any) -> Any:
    if isinstance(value, typer.models.OptionInfo):
        return None
    return value


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _stringify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _stringify(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_stringify(v) for v in value]
    return value


def _write_manifest(manifest_dir: Path, payload: dict[str, Any]) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_path = manifest_dir / f"{payload['run_id']}.json"
    out_path.write_text(
        json.dumps(_stringify(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    return out_path


def _float_token(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p")


def _dataset_train_filename(
    function: str,
    num_samples: int,
    num_sigmas: int,
    sigma_scale: float,
    seed: int,
) -> str:
    return (
        f"{function}_n{num_samples}_k{num_sigmas}_ss{_float_token(sigma_scale)}_seed{seed}_train.parquet"
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"Config at {path} must be a YAML mapping/object.")
    return payload


def _as_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


@app.command()
def prepare_data(
    function: str = typer.Option("all", help="Function name (ackley/levy/eggholder) or all."),
    output_dir: Path = PROCESSED_DATA_DIR,
    num_samples: int = 20000,
    num_sigmas: int = 3,
    sigma_scale: float = 5.0,
    train_samples: int = 10000,
    noise_ratio: float = 0.02,
    seed: int = 42,
    versioned_name: bool = typer.Option(
        True,
        help="Write versioned dataset filename (recommended).",
    ),
) -> None:
    """Generate processed datasets without starting training."""
    output_name = None
    if function != "all" and versioned_name:
        output_name = _dataset_train_filename(function, num_samples, num_sigmas, sigma_scale, seed)
    dataset_main(
        function=function,
        output_dir=output_dir,
        num_samples=num_samples,
        num_sigmas=num_sigmas,
        sigma_scale=sigma_scale,
        train_samples=train_samples,
        noise_ratio=noise_ratio,
        seed=seed,
        output_name=output_name,
    )


@app.command()
def run(
    method: str = typer.Option(
        "curriculum",
        help="Experiment method: single, curriculum, meta, sweep, or schedule_sweep.",
    ),
    function: str = typer.Option(
        "ackley",
        help="Function name: ackley, levy, eggholder.",
    ),
    input_path: Path | None = typer.Option(
        None,
        help="Optional dataset path. If omitted, defaults to data/processed/{function}.parquet.",
    ),
    regenerate_data: bool = typer.Option(
        False,
        help="Regenerate dataset before running the experiment.",
    ),
    num_samples: int = 20000,
    num_sigmas: int = 3,
    sigma_scale: float = 5.0,
    train_samples: int = 10000,
    noise_ratio: float = 0.02,
    seed: int = 42,
    experiment_name: str | None = None,
    run_name: str | None = None,
    # Shared model/training options
    model_arch: str = "mlp",
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
    grid_res: int = 100,
    snapshot_interval: int = 50,
    # single mode options
    sigma_level: int | None = typer.Option(None, help="Used only for method=single."),
    # curriculum options
    from_sweep: str | None = typer.Option(
        None,
        help="Path to Optuna sqlite DB; if set, best params are loaded for train methods.",
    ),
    study_name: str | None = typer.Option(
        None,
        help="Optuna study name in from_sweep DB.",
    ),
    # meta options
    lr_model: float = 3e-4,
    lr_meta: float = 1e-3,
    inner_steps: int = 10,
    lambda_reg: float = 0.1,
    num_crude: int | None = None,
    # sweep options
    n_trials: int = 40,
    report_interval: int = 10,
    model_archs: str = "mlp,siren,fourier",
    sweep_storage: str | None = None,
    min_train_per_param: float = 10.0,
    max_train_per_param: float = 20.0,
    step_metrics_interval: int = 50,
    log_dataset_artifact: bool = False,
    # schedule-sweep options
    schedule_num_losses: int = 4,
    study_note: str = (
        "Optuna schedule-only sweep: frozen model/training hyperparameters, "
        "optimize hardest-level validation loss."
    ),
) -> None:
    """Run a fully logged experiment from one command."""
    method = method.strip().lower()
    sigma_level = _normalize_optioninfo(sigma_level)
    from_sweep = _normalize_optioninfo(from_sweep)
    study_name = _normalize_optioninfo(study_name)

    if function not in FUNCTIONS:
        raise typer.BadParameter(
            f"Unknown function '{function}'. Allowed: {', '.join(sorted(FUNCTIONS))}."
        )
    if method not in {"single", "curriculum", "meta", "sweep", "schedule_sweep"}:
        raise typer.BadParameter(
            "method must be one of: single, curriculum, meta, sweep, schedule_sweep"
        )
    if method == "single" and sigma_level is None:
        sigma_level = -1

    default_dataset = PROCESSED_DATA_DIR / _dataset_train_filename(
        function=function,
        num_samples=num_samples,
        num_sigmas=num_sigmas,
        sigma_scale=sigma_scale,
        seed=seed,
    )
    resolved_input = input_path or default_dataset

    if regenerate_data or not resolved_input.exists():
        logger.info(f"Generating dataset for {function} -> {resolved_input}")
        dataset_main(
            function=function,
            output_dir=resolved_input.parent,
            num_samples=num_samples,
            num_sigmas=num_sigmas,
            sigma_scale=sigma_scale,
            train_samples=train_samples,
            noise_ratio=noise_ratio,
            seed=seed,
            output_name=resolved_input.name,
        )
    else:
        logger.info(f"Using existing dataset: {resolved_input}")

    run_id = f"{_timestamp()}_{function}_{method}"
    run_output_dir = REPORTS_DIR / "figures" / run_id
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "entrypoint": "python -m ma_thesis.experiment run",
        "argv": " ".join(shlex.quote(a) for a in sys.argv),
        "method": method,
        "function": function,
        "dataset_path": resolved_input,
        "output_dir": run_output_dir,
        "params": {
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
            "grid_res": grid_res,
            "snapshot_interval": snapshot_interval,
            "sigma_level": sigma_level,
            "from_sweep": from_sweep,
            "study_name": study_name,
            "lr_model": lr_model,
            "lr_meta": lr_meta,
            "inner_steps": inner_steps,
            "lambda_reg": lambda_reg,
            "num_crude": num_crude,
            "n_trials": n_trials,
            "report_interval": report_interval,
            "model_archs": model_archs,
            "sweep_storage": sweep_storage,
            "schedule_num_losses": schedule_num_losses,
            "study_note": study_note,
            "min_train_per_param": min_train_per_param,
            "max_train_per_param": max_train_per_param,
            "step_metrics_interval": step_metrics_interval,
            "log_dataset_artifact": log_dataset_artifact,
            "train_samples": train_samples,
            "noise_ratio": noise_ratio,
            "seed": seed,
        },
    }
    manifest_path = _write_manifest(REPORTS_DIR / "runs", manifest)
    logger.info(f"Run manifest saved: {manifest_path}")

    if method in {"single", "curriculum"}:
        train_main(
            input_path=resolved_input,
            output_dir=run_output_dir,
            patience=patience,
            min_delta=min_delta,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            grid_res=grid_res,
            snapshot_interval=snapshot_interval,
            experiment_name=experiment_name or "gaussian-continuation",
            mode=method,
            sigma_level=sigma_level,
            run_name=run_name or run_id,
            model_arch=model_arch,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            activation=activation,
            num_layers=num_layers,
            omega_0=omega_0,
            num_fourier=num_fourier,
            fourier_sigma=fourier_sigma,
            from_sweep=from_sweep,
            study_name=study_name,
            min_train_per_param=min_train_per_param,
            step_metrics_interval=step_metrics_interval,
            log_dataset_artifact=log_dataset_artifact,
        )
        return

    if method == "meta":
        meta_main(
            input_path=resolved_input,
            output_dir=run_output_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr_model=lr_model,
            lr_meta=lr_meta,
            inner_steps=inner_steps,
            lambda_reg=lambda_reg,
            model_arch=model_arch,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            activation=activation,
            num_fourier=num_fourier,
            fourier_sigma=fourier_sigma,
            num_crude=num_crude,
            grid_res=grid_res,
            snapshot_interval=snapshot_interval,
            experiment_name=experiment_name or "meta-curriculum",
            run_name=run_name or run_id,
            min_train_per_param=min_train_per_param,
            log_dataset_artifact=log_dataset_artifact,
        )
        return

    if method == "schedule_sweep":
        schedule_sweep_main(
            input_path=resolved_input,
            output_dir=run_output_dir,
            experiment_name=experiment_name or "curriculum-schedule-sweep",
            study_name=run_name or f"schedule-sweep-{function}",
            study_note=study_note,
            storage=sweep_storage,
            n_trials=n_trials,
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
            num_losses=schedule_num_losses,
        )
        return

    sweep_main(
        input_path=resolved_input,
        output_dir=run_output_dir,
        n_trials=n_trials,
        epochs=epochs,
        patience=patience,
        min_delta=min_delta,
        report_interval=report_interval,
        grid_res=min(grid_res, 100),
        seed=seed,
        storage=sweep_storage,
        experiment_name=experiment_name or f"baseline-sweep-{function}",
        model_archs=model_archs,
        step_metrics_interval=step_metrics_interval,
        min_train_per_param=min_train_per_param,
        max_train_per_param=max_train_per_param,
    )


@app.command()
def run_config(
    config_path: Path = typer.Argument(..., help="Path to YAML config."),
) -> None:
    """Run an experiment from a YAML config file."""
    cfg = _load_yaml(config_path)
    params = cfg.get("run", cfg)
    if not isinstance(params, dict):
        raise typer.BadParameter("Config key 'run' must be a mapping/object.")

    kwargs = dict(params)
    kwargs["input_path"] = _as_path(kwargs.get("input_path"))
    run(**kwargs)


if __name__ == "__main__":
    app()
