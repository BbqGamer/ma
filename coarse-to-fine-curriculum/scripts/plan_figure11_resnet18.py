#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


CURRICULUM_LENGTHS = [5, 10, 20, 30, 40, 50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Figure-11-style CIFAR-100 ResNet-18 curriculum-length sweep"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--data-dir", default="/workspace/data")
    parser.add_argument("--output-dir", default="/workspace/runs")
    parser.add_argument("--python", default="python train_coarse_to_fine.py")
    parser.add_argument("--run-prefix", default="fig11-resnet18-cifar100")
    parser.add_argument("--output", type=Path, default=Path("figure11_resnet18_commands.sh"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("figure11_resnet18_manifest.csv"),
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, mode: str, curriculum_epochs: int | None) -> tuple[str, dict[str, object]]:
    suffix = "baseline" if mode == "baseline" else f"curr{curriculum_epochs}"
    run_id = f"{args.run_prefix}-seed{args.seed}-{suffix}"
    parts = [
        args.python,
        "--mode", mode,
        "--dataset", "cifar100",
        "--model", "resnet18",
        "--epochs", str(args.epochs),
        "--val_ratio", str(args.val_ratio),
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
        "--run_id", run_id,
        "--seed", str(args.seed),
    ]
    reference_run_dir = ""
    if mode == "curriculum":
        reference_run_id = f"{args.run_prefix}-seed{args.seed}-baseline"
        reference_run_dir = (
            f"{args.output_dir}/{reference_run_id}/cifar100_resnet18_baseline"
        )
        parts.extend(["--curriculum_epochs", str(curriculum_epochs)])
        parts.extend(["--reference_run_dir", reference_run_dir])

    command = " ".join(parts)
    row = {
        "dataset": "cifar100",
        "model": "resnet18",
        "seed": args.seed,
        "epochs": args.epochs,
        "val_ratio": args.val_ratio,
        "mode": mode,
        "curriculum_epochs": curriculum_epochs if curriculum_epochs is not None else "",
        "run_id": run_id,
        "reference_run_dir": reference_run_dir,
        "command": command,
    }
    return command, row


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    commands: list[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]

    command, row = build_command(args, "baseline", None)
    commands.append(command)
    rows.append(row)

    for curriculum_epochs in CURRICULUM_LENGTHS:
        command, row = build_command(args, "curriculum", curriculum_epochs)
        commands.append(command)
        rows.append(row)

    args.output.write_text("\n".join(commands) + "\n")
    args.output.chmod(0o755)

    with args.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote commands to {args.output}")
    print(f"Wrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()
