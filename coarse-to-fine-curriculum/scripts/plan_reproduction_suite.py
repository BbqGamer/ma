#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate recommended reproduction commands")
    parser.add_argument("--output", type=Path, default=Path("suite_commands.sh"))
    parser.add_argument("--manifest", type=Path, default=Path("suite_manifest.csv"))
    parser.add_argument("--suite", choices=["smoke", "core", "extended"], default="core")
    parser.add_argument("--data-dir", default="/workspace/data")
    parser.add_argument("--output-dir", default="/workspace/runs")
    parser.add_argument("--python", default="python train_coarse_to_fine.py")
    parser.add_argument("--base-seed", type=int, default=42)
    return parser.parse_args()


def build_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    suites = {
        "smoke": {
            "datasets": ["cifar100"],
            "models": ["cnn"],
            "lengths": [10, 20],
            "seeds": [args.base_seed],
        },
        "core": {
            "datasets": ["cifar100"],
            "models": ["cnn", "resnet18", "resnet50"],
            "lengths": [5, 10, 20, 30, 40, 50],
            "seeds": [args.base_seed + offset for offset in range(3)],
        },
        "extended": {
            "datasets": ["cifar10", "cifar100", "tiny-imagenet"],
            "models": ["cnn", "resnet18"],
            "lengths": [5, 10, 20, 30, 40, 50],
            "seeds": [args.base_seed + offset for offset in range(3)],
        },
    }

    spec = suites[args.suite]
    for dataset in spec["datasets"]:
        for model in spec["models"]:
            default_epochs = 400 if model == "cnn" else 200
            val_ratio = 0.2 if model == "cnn" else 0.1
            for seed in spec["seeds"]:
                run_root = f"{dataset}-{model}-seed{seed}"
                rows.append(
                    {
                        "suite": args.suite,
                        "dataset": dataset,
                        "model": model,
                        "mode": "baseline",
                        "curriculum_epochs": "",
                        "seed": seed,
                        "epochs": default_epochs,
                        "val_ratio": val_ratio,
                        "run_id": f"{run_root}-baseline",
                    }
                )
                for curriculum_epochs in spec["lengths"]:
                    rows.append(
                        {
                            "suite": args.suite,
                            "dataset": dataset,
                            "model": model,
                            "mode": "curriculum",
                            "curriculum_epochs": curriculum_epochs,
                            "seed": seed,
                            "epochs": default_epochs,
                            "val_ratio": val_ratio,
                            "run_id": f"{run_root}-curr{curriculum_epochs}",
                        }
                    )
    return rows


def build_command(row: dict[str, object], args: argparse.Namespace) -> str:
    parts = [
        args.python,
        f"--mode {row['mode']}",
        f"--dataset {row['dataset']}",
        f"--model {row['model']}",
        f"--epochs {row['epochs']}",
        f"--val_ratio {row['val_ratio']}",
        f"--data_dir {args.data_dir}",
        f"--output_dir {args.output_dir}",
        f"--run_id {row['run_id']}",
        f"--seed {row['seed']}",
    ]
    if row["mode"] == "curriculum":
        root = f"{row['dataset']}-{row['model']}-seed{row['seed']}"
        reference_run_id = f"{root}-baseline"
        reference_run_dir = (
            f"{args.output_dir}/{reference_run_id}/{row['dataset']}_{row['model']}_baseline"
        )
        parts.append(f"--curriculum_epochs {row['curriculum_epochs']}")
        parts.append(f"--reference_run_dir {reference_run_dir}")
    return " ".join(str(part) for part in parts)


def main() -> None:
    args = parse_args()
    rows = build_rows(args)

    args.output.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(build_command(row, args) for row in rows) + "\n")
    args.output.chmod(0o755)

    with args.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) + ["command"])
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["command"] = build_command(row, args)
            writer.writerow(payload)

    print(f"Wrote {len(rows)} commands to {args.output}")
    print(f"Wrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()
