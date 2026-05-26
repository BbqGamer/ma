#!/usr/bin/env python3
"""Bundle CIFAR-100 easy-vs-hard experiment results for download."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import tarfile
from typing import Any

import typer

from ma_thesis.config import PROJ_ROOT, REPORTS_DIR

app = typer.Typer(add_completion=False)
ARTIFACT_ROOT = REPORTS_DIR / "artifacts"
CIFAR_ROOT = REPORTS_DIR / "cifar100_easy_hard"
GENERATED_ROOT = PROJ_ROOT / "llm_schedules" / "generated"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _should_include(path: Path, *, include_checkpoints: bool) -> bool:
    if include_checkpoints:
        return True
    if any(part == "checkpoints" for part in path.parts):
        return False
    if path.suffix == ".ckpt":
        return False
    return True


def _add_tree(
    tar: tarfile.TarFile,
    path: Path,
    *,
    arcname: str,
    include_checkpoints: bool,
) -> None:
    if not path.exists():
        return
    if path.is_file():
        if _should_include(path, include_checkpoints=include_checkpoints):
            tar.add(path, arcname=arcname)
        return
    for child in sorted(path.rglob("*")):
        if child.is_dir():
            continue
        if not _should_include(child, include_checkpoints=include_checkpoints):
            continue
        child_arcname = str(Path(arcname) / child.relative_to(path))
        tar.add(child, arcname=child_arcname)


@app.command()
def main(
    bundle_name: str = typer.Option(f"cifar100_easy_hard_bundle_{_timestamp()}"),
    llm_study_name: str | None = typer.Option(None, help="Optional LLM study directory to include."),
    optuna_study_name: str | None = typer.Option(None, help="Optional Optuna study directory to include."),
    suite_name: str | None = typer.Option(None, help="Optional suite run directory to include."),
    include_all_reports: bool = typer.Option(False, help="Include the entire reports/cifar100_easy_hard tree."),
    include_checkpoints: bool = typer.Option(False, help="Include Lightning checkpoint files."),
) -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    bundle_path = ARTIFACT_ROOT / f"{bundle_name}.tar.gz"

    manifest: dict[str, Any] = {
        "bundle_name": bundle_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "include_all_reports": include_all_reports,
        "llm_study_name": llm_study_name,
        "optuna_study_name": optuna_study_name,
        "items": [],
    }

    with tarfile.open(bundle_path, "w:gz") as tar:
        _add_tree(
            tar,
            PROJ_ROOT / "pyproject.toml",
            arcname="pyproject.toml",
            include_checkpoints=include_checkpoints,
        )
        manifest["items"].append("pyproject.toml")

        if include_all_reports:
            print("Adding reports/cifar100_easy_hard ...", flush=True)
            _add_tree(
                tar,
                CIFAR_ROOT,
                arcname="reports/cifar100_easy_hard",
                include_checkpoints=include_checkpoints,
            )
            manifest["items"].append("reports/cifar100_easy_hard")
        else:
            if llm_study_name:
                llm_report_dir = CIFAR_ROOT / "llm_search" / llm_study_name
                print(f"Adding {llm_report_dir} ...", flush=True)
                _add_tree(
                    tar,
                    llm_report_dir,
                    arcname=f"reports/cifar100_easy_hard/llm_search/{llm_study_name}",
                    include_checkpoints=include_checkpoints,
                )
                manifest["items"].append(f"reports/cifar100_easy_hard/llm_search/{llm_study_name}")

            if optuna_study_name:
                optuna_dir = CIFAR_ROOT / "optuna" / optuna_study_name
                print(f"Adding {optuna_dir} ...", flush=True)
                _add_tree(
                    tar,
                    optuna_dir,
                    arcname=f"reports/cifar100_easy_hard/optuna/{optuna_study_name}",
                    include_checkpoints=include_checkpoints,
                )
                manifest["items"].append(f"reports/cifar100_easy_hard/optuna/{optuna_study_name}")

            if suite_name:
                suite_dir = CIFAR_ROOT / "suite_runs" / suite_name
                print(f"Adding {suite_dir} ...", flush=True)
                _add_tree(
                    tar,
                    suite_dir,
                    arcname=f"reports/cifar100_easy_hard/suite_runs/{suite_name}",
                    include_checkpoints=include_checkpoints,
                )
                manifest["items"].append(f"reports/cifar100_easy_hard/suite_runs/{suite_name}")

        if llm_study_name:
            llm_gen_dir = GENERATED_ROOT / llm_study_name
            print(f"Adding {llm_gen_dir} ...", flush=True)
            _add_tree(
                tar,
                llm_gen_dir,
                arcname=f"llm_schedules/generated/{llm_study_name}",
                include_checkpoints=include_checkpoints,
            )
            manifest["items"].append(f"llm_schedules/generated/{llm_study_name}")

        manifest_path = ARTIFACT_ROOT / f"{bundle_name}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        tar.add(manifest_path, arcname=f"{bundle_name}_manifest.json")

    print(json.dumps({"bundle_path": str(bundle_path), **manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
