#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a reproducible teacher-hierarchy comparison suite for a known CIFAR-100 setup."
    )
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100", "cifar10", "tiny-imagenet"])
    parser.add_argument(
        "--model",
        default="cnn",
        choices=["cnn", "cifar_resnet8", "cifar_resnet14", "cifar_resnet20", "resnet18"],
    )
    parser.add_argument("--cnn-width-multiplier", type=float, default=0.5)
    parser.add_argument("--cifar-resnet-width-multiplier", type=float, default=1.0)
    parser.add_argument("--curriculum-epochs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--random-seeds", default="1001,1002,1003")
    parser.add_argument("--teacher-run-dir", default="/runpod-volume/teachers/teacher-cifar100-resnet18-imagenet30")
    parser.add_argument("--teacher-checkpoint-path", default="")
    parser.add_argument("--teacher-model", default="")
    parser.add_argument("--teacher-embedding-split", default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--teacher-pretrained-source",
        default="none",
        choices=["none", "torchvision_imagenet"],
    )
    parser.add_argument("--wandb-project", default="coarse-to-fine-curriculum")
    parser.add_argument("--wandb-group", default="teacher-hierarchy-cifar100-cnn-w0.5")
    parser.add_argument("--wandb-tags", default="runpod,teacher-hierarchy,cifar100,cnn-w0.5,anti-curriculum")
    parser.add_argument("--data-dir", default="/runpod-volume/data")
    parser.add_argument("--output-dir", default="/runpod-volume/runs")
    parser.add_argument("--entrypoint", default="./entrypoint.sh")
    parser.add_argument("--output", type=Path, default=Path("teacher_hierarchy_suite_w0_5.sh"))
    parser.add_argument("--manifest", type=Path, default=Path("teacher_hierarchy_suite_w0_5_manifest.csv"))
    return parser.parse_args(argv)


def model_token(args: argparse.Namespace) -> str:
    if args.model == "cnn":
        return f"cnn-w{args.cnn_width_multiplier:g}" if args.cnn_width_multiplier != 1.0 else "cnn"
    if args.model.startswith("cifar_resnet") and args.cifar_resnet_width_multiplier != 1.0:
        return f"{args.model}-w{args.cifar_resnet_width_multiplier:g}"
    return args.model


def build_plan(args: argparse.Namespace) -> tuple[list[str], list[dict[str, object]]]:
    spec = (
        f"{args.dataset}:{args.model}:{args.cnn_width_multiplier}:"
        f"{args.cifar_resnet_width_multiplier}:{args.curriculum_epochs}:{args.epochs}"
    )
    env_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated teacher-hierarchy comparison suite",
        "export EXPERIMENT=teacher_hierarchy_suite",
        f"export DATA_DIR={args.data_dir}",
        f"export OUTPUT_DIR={args.output_dir}",
        f"export TEACHER_HIERARCHY_SPECS={spec}",
        f"export TEACHER_HIERARCHY_SEEDS={args.seeds}",
        f"export TEACHER_HIERARCHY_RANDOM_SEEDS={args.random_seeds}",
        f"export TEACHER_EMBEDDING_SPLIT={args.teacher_embedding_split}",
        f"export TEACHER_PRETRAINED_SOURCE={args.teacher_pretrained_source}",
        "export WANDB=1",
        f"export WANDB_PROJECT={args.wandb_project}",
        f"export WANDB_GROUP={args.wandb_group}",
        f"export WANDB_TAGS={args.wandb_tags}",
        "export OPTIMIZER=adam",
        "export SCHEDULER=none",
        "export LR=0.001",
        "export BATCH_SIZE=128",
        "export VAL_RATIO=0.1",
        "export DETERMINISTIC=1",
        "export AMP=1",
        "export DOWNLOAD=1",
    ]
    if args.teacher_run_dir:
        env_lines.append(f"export TEACHER_RUN_DIR={args.teacher_run_dir}")
    if args.teacher_checkpoint_path:
        env_lines.append(f"export TEACHER_CHECKPOINT_PATH={args.teacher_checkpoint_path}")
    if args.teacher_model:
        env_lines.append(f"export TEACHER_MODEL={args.teacher_model}")
    env_lines.extend(["", args.entrypoint])

    manifest_rows: list[dict[str, object]] = []
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    random_seeds = [int(item.strip()) for item in args.random_seeds.split(",") if item.strip()]
    token = model_token(args)
    for seed in seeds:
        prefix = f"teacher-{args.dataset}-{token}-seed{seed}"
        manifest_rows.extend(
            [
                {"seed": seed, "condition": "baseline", "run_id": f"{prefix}-baseline"},
                {"seed": seed, "condition": "self", "run_id": f"{prefix}-self-curr{args.curriculum_epochs}"},
                {"seed": seed, "condition": "teacher", "run_id": f"{prefix}-teacher-curr{args.curriculum_epochs}"},
                {"seed": seed, "condition": "teacher_anti", "run_id": f"{prefix}-teacher-anti-curr{args.curriculum_epochs}"},
            ]
        )
        for random_seed in random_seeds:
            manifest_rows.append(
                {
                    "seed": seed,
                    "condition": "random",
                    "random_hierarchy_seed": random_seed,
                    "run_id": f"{prefix}-random{random_seed}-curr{args.curriculum_epochs}",
                }
            )
    for row in manifest_rows:
        row.update(
            {
                "dataset": args.dataset,
                "model": args.model,
                "model_token": token,
                "cnn_width_multiplier": args.cnn_width_multiplier,
                "cifar_resnet_width_multiplier": args.cifar_resnet_width_multiplier,
                "curriculum_epochs": args.curriculum_epochs,
                "epochs": args.epochs,
            }
        )
    return env_lines, manifest_rows


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    script_lines, manifest_rows = build_plan(args)
    args.output.write_text("\n".join(script_lines) + "\n")
    args.output.chmod(0o755)
    fieldnames: list[str] = []
    for row in manifest_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with args.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote teacher-hierarchy suite script to {args.output}")
    print(f"Wrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()
