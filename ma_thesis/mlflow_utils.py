"""Shared MLflow helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow


def log_run_config(
    output_dir: Path,
    payload: dict[str, Any],
    *,
    filename: str = "run_config.json",
    artifact_path: str = "config",
) -> Path:
    """Write JSON config payload and log it as an MLflow artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / filename
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    mlflow.log_artifact(str(config_path), artifact_path=artifact_path)
    return config_path


def log_dataset_reference(
    path: Path,
    *,
    key: str,
    artifact_path: str = "data",
    log_artifact: bool = False,
) -> None:
    """Log dataset path param and optionally upload dataset file as artifact."""
    mlflow.log_param(key, str(path))
    if log_artifact:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
