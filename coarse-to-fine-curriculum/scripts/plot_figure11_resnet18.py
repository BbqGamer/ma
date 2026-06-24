#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


CURRICULUM_LENGTHS = [5, 10, 20, 30, 40, 50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a Figure-11-style CIFAR-100 curriculum-length sweep"
    )
    parser.add_argument(
        "runs_root",
        type=Path,
        help="Directory containing the generated run_id directories",
    )
    parser.add_argument("--run-prefix", default="fig11-resnet18-cifar100")
    parser.add_argument(
        "--model",
        choices=[
            "cnn",
            "cifar_resnet8",
            "cifar_resnet14",
            "cifar_resnet20",
            "cifar_resnet32",
            "cifar_resnet44",
            "cifar_resnet56",
            "resnet18",
            "resnet50",
        ],
        default="resnet18",
    )
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", choices=["test_acc", "val_acc"], default="test_acc")
    return parser.parse_args()


def run_dir(root: Path, run_id: str, suffix: str) -> Path:
    return root / run_id / suffix


def load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path / "history.csv")


def load_summary(path: Path) -> pd.Series:
    return pd.read_csv(path / "summary.csv").iloc[0]


def plot_metric(
    baseline: pd.DataFrame,
    curricula: dict[int, pd.DataFrame],
    metric: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        baseline["epoch"],
        baseline[metric],
        label="baseline",
        color="black",
        linewidth=2.0,
    )

    cmap = plt.get_cmap("viridis", len(curricula))
    for idx, (curriculum_epochs, history) in enumerate(sorted(curricula.items())):
        ax.plot(
            history["epoch"],
            history[metric],
            label=f"curriculum {curriculum_epochs}",
            color=cmap(idx),
            linewidth=1.8,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} across curriculum lengths")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary_table(
    baseline_summary: pd.Series,
    curriculum_summaries: dict[int, pd.Series],
    output_path: Path,
) -> None:
    rows = [
        {
            "label": "baseline",
            "curriculum_epochs": "",
            **baseline_summary.to_dict(),
        }
    ]
    for curriculum_epochs, summary in sorted(curriculum_summaries.items()):
        rows.append(
            {
                "label": f"curriculum_{curriculum_epochs}",
                "curriculum_epochs": curriculum_epochs,
                **summary.to_dict(),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    seed = args.seed
    run_prefix = args.run_prefix
    root = args.runs_root
    analysis_dir = root / f"{run_prefix}-seed{seed}-figure11-analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    baseline_run_id = f"{run_prefix}-seed{seed}-baseline"
    baseline_dir = run_dir(root, baseline_run_id, f"{args.dataset}_{args.model}_baseline")
    baseline_history = load_history(baseline_dir)
    baseline_summary = load_summary(baseline_dir)

    curriculum_histories: dict[int, pd.DataFrame] = {}
    curriculum_summaries: dict[int, pd.Series] = {}
    for curriculum_epochs in CURRICULUM_LENGTHS:
        run_id = f"{run_prefix}-seed{seed}-curr{curriculum_epochs}"
        path = run_dir(root, run_id, f"{args.dataset}_{args.model}_curriculum")
        curriculum_histories[curriculum_epochs] = load_history(path)
        curriculum_summaries[curriculum_epochs] = load_summary(path)

    plot_metric(
        baseline_history,
        curriculum_histories,
        args.metric,
        analysis_dir / f"figure11_{args.metric}.png",
    )
    plot_metric(
        baseline_history,
        curriculum_histories,
        "val_acc",
        analysis_dir / "figure11_val_acc.png",
    )
    write_summary_table(
        baseline_summary,
        curriculum_summaries,
        analysis_dir / "figure11_summary.csv",
    )

    print(f"Wrote Figure-11-style analysis to {analysis_dir}")


if __name__ == "__main__":
    main()
