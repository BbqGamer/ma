#!/usr/bin/env python3
"""Run the CIFAR-100 Optuna search, LLM search, and result bundling in one command."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import typer

from ma_thesis.config import PROJ_ROOT, REPORTS_DIR

app = typer.Typer(add_completion=False)
ROOT = PROJ_ROOT
PYTHON_BIN = Path(sys.executable)
OPTUNA_SCRIPT = ROOT / "scripts" / "run_cifar100_optuna_search.py"
LLM_SCRIPT = ROOT / "scripts" / "run_cifar100_llm_search.py"
BUNDLE_SCRIPT = ROOT / "scripts" / "bundle_cifar100_results.py"
SUITE_ROOT = REPORTS_DIR / "cifar100_easy_hard" / "suite_runs"
OPTUNA_ROOT = REPORTS_DIR / "cifar100_easy_hard" / "optuna"
LLM_ROOT = REPORTS_DIR / "cifar100_easy_hard" / "llm_search"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_phase(name: str, cmd: list[str], *, log_dir: Path) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}_stdout.txt"
    stderr_path = log_dir / f"{name}_stderr.txt"
    started_at = datetime.now().isoformat(timespec="seconds")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")
    return {
        "name": name,
        "command": cmd,
        "returncode": int(result.returncode),
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "status": "ok" if result.returncode == 0 else "failed",
    }


def _maybe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@app.command()
def main(
    suite_name: str = typer.Option(f"cifar100_suite_{_timestamp()}"),
    optuna_trials: int = typer.Option(10),
    llm_iterations: int = typer.Option(10),
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
        help="Enable early stopping for both searches. Off by default for fair fixed-budget runs.",
    ),
    accelerator: str = "gpu",
    devices: str = "1",
    precision: str = "16-mixed",
    openai_model: str = "gpt-5.4-mini",
    openai_api_key_env: str = "OPENAI_API_KEY",
    openai_base_url: str = "https://api.openai.com/v1",
    bundle_results: bool = typer.Option(True),
) -> None:
    suite_dir = SUITE_ROOT / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    log_dir = suite_dir / "logs"

    optuna_study_name = f"{suite_name}_optuna"
    llm_study_name = f"{suite_name}_llm"
    bundle_name = f"{suite_name}_bundle"

    summary: dict[str, Any] = {
        "suite_name": suite_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "optuna_study_name": optuna_study_name,
        "llm_study_name": llm_study_name,
        "bundle_name": bundle_name,
        "phases": [],
    }
    summary_path = suite_dir / "summary.json"

    common_args = [
        "--eval-seeds",
        eval_seeds,
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--max-epochs",
        str(max_epochs),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--ema-alpha",
        str(ema_alpha),
        "--history-window",
        str(history_window),
        "--val-fraction",
        str(val_fraction),
        "--patience",
        str(patience),
        "--min-delta",
        str(min_delta),
        "--accelerator",
        accelerator,
        "--devices",
        devices,
        "--precision",
        precision,
    ]
    if use_early_stopping:
        common_args.append("--use-early-stopping")

    optuna_cmd = [
        str(PYTHON_BIN),
        str(OPTUNA_SCRIPT),
        "--study-name",
        optuna_study_name,
        "--n-trials",
        str(optuna_trials),
        *common_args,
    ]
    llm_cmd = [
        str(PYTHON_BIN),
        str(LLM_SCRIPT),
        "--study-name",
        llm_study_name,
        "--iterations",
        str(llm_iterations),
        "--openai-model",
        openai_model,
        "--openai-api-key-env",
        openai_api_key_env,
        "--openai-base-url",
        openai_base_url,
        *common_args,
    ]

    try:
        optuna_phase = _run_phase("optuna", optuna_cmd, log_dir=log_dir)
        optuna_phase["aggregate_path"] = str(OPTUNA_ROOT / optuna_study_name / "aggregate.json")
        optuna_phase["aggregate"] = _maybe_load_json(Path(optuna_phase["aggregate_path"]))
        summary["phases"].append(optuna_phase)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        if optuna_phase["status"] != "ok":
            raise RuntimeError(f"Optuna phase failed. See {optuna_phase['stderr_path']}")

        llm_phase = _run_phase("llm", llm_cmd, log_dir=log_dir)
        llm_phase["history_path"] = str(LLM_ROOT / llm_study_name / "history.jsonl")
        llm_phase["token_summary_path"] = str(LLM_ROOT / llm_study_name / "token_summary.json")
        llm_phase["token_summary"] = _maybe_load_json(Path(llm_phase["token_summary_path"]))
        summary["phases"].append(llm_phase)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        if llm_phase["status"] != "ok":
            raise RuntimeError(f"LLM phase failed. See {llm_phase['stderr_path']}")
    finally:
        if bundle_results:
            bundle_phase = _run_phase(
                "bundle",
                [
                    str(PYTHON_BIN),
                    str(BUNDLE_SCRIPT),
                    "--bundle-name",
                    bundle_name,
                    "--llm-study-name",
                    llm_study_name,
                    "--optuna-study-name",
                    optuna_study_name,
                ],
                log_dir=log_dir,
            )
            bundle_phase["bundle_path"] = str(REPORTS_DIR / "artifacts" / f"{bundle_name}.tar.gz")
            summary["phases"].append(bundle_phase)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
