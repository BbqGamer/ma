#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


CURRICULUM_LENGTHS = [5, 10, 20, 30, 40, 50]
DEFAULT_OPTIMIZER = "adam"
DEFAULT_SCHEDULER = "none"
DEFAULT_LR = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Figure-11-style CIFAR-100 curriculum-length sweep"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--dataset",
        choices=[
            "cifar10",
            "cifar100",
            "mnist",
            "fashion-mnist",
            "kmnist",
            "svhn",
            "stl10",
            "tiny-imagenet",
        ],
        default="cifar100",
    )
    parser.add_argument("--model", choices=["cnn", "resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--optimizer", default=DEFAULT_OPTIMIZER)
    parser.add_argument("--scheduler", default=DEFAULT_SCHEDULER)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--roughness-probes", action="store_true")
    parser.add_argument("--roughness-epochs", default="1,5,10,11,20,50,100")
    parser.add_argument("--roughness-batches", type=int, default=2)
    parser.add_argument("--sharpness-rho", type=float, default=0.05)
    parser.add_argument("--hessian-iters", type=int, default=10)
    parser.add_argument("--hessian-samples", type=int, default=2)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="coarse-to-fine-curriculum")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-tags", default="runpod,figure11")
    parser.add_argument("--data-dir", default="/workspace/data")
    parser.add_argument("--output-dir", default="/workspace/runs")
    parser.add_argument("--python", default="python train_coarse_to_fine.py")
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--output", type=Path, default=Path("figure11_resnet18_commands.sh"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("figure11_resnet18_manifest.csv"),
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, mode: str, curriculum_epochs: int | None) -> tuple[str, dict[str, object]]:
    run_prefix = args.run_prefix or f"fig11-{args.model}-{args.dataset}"
    suffix = "baseline" if mode == "baseline" else f"curr{curriculum_epochs}"
    run_id = f"{run_prefix}-seed{args.seed}-{suffix}"
    parts = [
        args.python,
        "--mode", mode,
        "--dataset", args.dataset,
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--val_ratio", str(args.val_ratio),
        "--optimizer", args.optimizer,
        "--scheduler", args.scheduler,
        "--lr", str(args.lr),
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
        "--run_id", run_id,
        "--seed", str(args.seed),
    ]
    if args.batch_size is not None:
        parts.extend(["--batch_size", str(args.batch_size)])
    if args.roughness_probes:
        parts.extend(["--roughness-probes"])
        parts.extend(["--roughness-epochs", args.roughness_epochs])
        parts.extend(["--roughness-batches", str(args.roughness_batches)])
        parts.extend(["--sharpness-rho", str(args.sharpness_rho)])
        parts.extend(["--hessian-iters", str(args.hessian_iters)])
        parts.extend(["--hessian-samples", str(args.hessian_samples)])
    if args.wandb:
        group = args.wandb_group or f"{run_prefix}-seed{args.seed}"
        parts.extend(["--wandb", "--wandb-project", args.wandb_project, "--wandb-group", group])
        parts.extend(["--wandb-tags", args.wandb_tags])
        if args.wandb_entity:
            parts.extend(["--wandb-entity", args.wandb_entity])

    reference_run_dir = ""
    if mode == "curriculum":
        reference_run_id = f"{run_prefix}-seed{args.seed}-baseline"
        reference_run_dir = f"{args.output_dir}/{reference_run_id}/{args.dataset}_{args.model}_baseline"
        parts.extend(["--curriculum_epochs", str(curriculum_epochs)])
        parts.extend(["--reference_run_dir", reference_run_dir])

    command = " ".join(parts)
    row = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "val_ratio": args.val_ratio,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "lr": args.lr,
        "batch_size": args.batch_size if args.batch_size is not None else "",
        "roughness_probes": args.roughness_probes,
        "roughness_epochs": args.roughness_epochs if args.roughness_probes else "",
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
