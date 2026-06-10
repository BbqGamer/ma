#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyze_run import analyze_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze one or more coarse-to-fine result directories"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Either a single run directory or a directory containing many run directories",
    )
    return parser.parse_args()


def is_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_baseline = any(child.is_dir() and child.name.endswith("_baseline") for child in path.iterdir())
    has_curriculum = any(child.is_dir() and child.name.endswith("_curriculum") for child in path.iterdir())
    return has_baseline and has_curriculum


def discover_run_dirs(path: Path) -> list[Path]:
    if is_run_dir(path):
        return [path]
    return sorted(child for child in path.iterdir() if is_run_dir(child))


def summarize_run(run_dir: Path) -> dict[str, object]:
    analysis_path = run_dir / "analysis" / "comparison_summary.csv"
    if not analysis_path.exists():
        raise FileNotFoundError(f"Missing analysis summary: {analysis_path}")

    df = pd.read_csv(analysis_path)
    if set(df["run"]) != {"baseline", "curriculum"}:
        raise ValueError(f"Unexpected run labels in {analysis_path}")

    baseline_dir = next(path for path in run_dir.iterdir() if path.is_dir() and path.name.endswith("_baseline"))
    curriculum_dir = next(path for path in run_dir.iterdir() if path.is_dir() and path.name.endswith("_curriculum"))
    curriculum_config = pd.read_json(curriculum_dir / "config.json", typ="series")
    curriculum_result = pd.read_json(curriculum_dir / "results.json", typ="series")

    baseline = df[df["run"] == "baseline"].iloc[0]
    curriculum = df[df["run"] == "curriculum"].iloc[0]
    return {
        "run_dir": run_dir.name,
        "dataset": curriculum_config.get("dataset"),
        "model": curriculum_config.get("model"),
        "seed": curriculum_config.get("seed"),
        "distance_source": curriculum_config.get("distance_source"),
        "epochs": curriculum_config.get("epochs"),
        "curriculum_epochs": curriculum_result.get("curriculum_epochs"),
        "baseline_best_test_acc": float(baseline["best_test_acc"]),
        "baseline_best_val_acc": float(baseline["best_val_acc"]),
        "baseline_best_test_epoch": int(baseline["best_test_epoch"]),
        "curriculum_best_test_acc": float(curriculum["best_test_acc"]),
        "curriculum_best_val_acc": float(curriculum["best_val_acc"]),
        "curriculum_best_test_epoch": int(curriculum["best_test_epoch"]),
        "gain_best_test_acc": float(curriculum["best_test_acc"] - baseline["best_test_acc"]),
        "gain_best_val_acc": float(curriculum["best_val_acc"] - baseline["best_val_acc"]),
    }


def plot_best_test_accuracy(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(summary))
    width = 0.38
    ax.bar(
        [i - width / 2 for i in x],
        summary["baseline_best_test_acc"],
        width=width,
        label="baseline",
        color="#1f77b4",
    )
    ax.bar(
        [i + width / 2 for i in x],
        summary["curriculum_best_test_acc"],
        width=width,
        label="curriculum",
        color="#d62728",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["run_dir"], rotation=25, ha="right")
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Best test accuracy by run")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gain(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ca02c" if gain >= 0 else "#d62728" for gain in summary["gain_best_test_acc"]]
    ax.bar(summary["run_dir"], summary["gain_best_test_acc"], color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Curriculum - baseline best test accuracy")
    ax.set_title("Best test accuracy gain by run")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gain_vs_curriculum_epochs(summary: pd.DataFrame, output_path: Path) -> bool:
    required = {"model", "curriculum_epochs", "gain_best_test_acc"}
    if not required.issubset(summary.columns):
        return False
    subset = summary.dropna(subset=["curriculum_epochs", "model"])
    if subset.empty:
        return False

    grouped = (
        subset.groupby(["model", "curriculum_epochs"], as_index=False)["gain_best_test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model, frame in grouped.groupby("model"):
        frame = frame.sort_values("curriculum_epochs")
        ax.plot(frame["curriculum_epochs"], frame["mean"], marker="o", label=str(model))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Curriculum epochs")
    ax.set_ylabel("Mean best-test gain")
    ax.set_title("Mean curriculum gain vs curriculum length")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    grouped.to_csv(output_path.with_suffix(".csv"), index=False)
    return True


def write_overview(summary: pd.DataFrame, output_path: Path) -> None:
    mean_gain = summary["gain_best_test_acc"].mean()
    lines = [
        "# Aggregate analysis",
        "",
        f"- Number of analyzed runs: **{len(summary)}**",
        f"- Mean best-test gain (curriculum - baseline): **{mean_gain:.4f}** ({mean_gain * 100:.2f} pp)",
        "",
    ]
    if {"model", "curriculum_epochs", "gain_best_test_acc"}.issubset(summary.columns):
        lines.extend(["## Mean gain by model and curriculum length", ""])
        grouped = (
            summary.groupby(["model", "curriculum_epochs"], as_index=False)["gain_best_test_acc"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        for _, row in grouped.iterrows():
            lines.append(
                f"- model={row['model']}, curriculum_epochs={int(row['curriculum_epochs'])}: "
                f"mean_gain={row['mean']:.4f}, std={0.0 if pd.isna(row['std']) else row['std']:.4f}, n={int(row['count'])}"
            )
        lines.extend(["", "## Runs", ""])
    else:
        lines.extend(["## Runs", ""])
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['run_dir']}: baseline={row['baseline_best_test_acc']:.4f}, "
            f"curriculum={row['curriculum_best_test_acc']:.4f}, gain={row['gain_best_test_acc']:.4f}"
        )
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    run_dirs = discover_run_dirs(args.path)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {args.path}")

    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        analyze_run(run_dir)
        rows.append(summarize_run(run_dir))

    if len(run_dirs) == 1:
        print(f"Analyzed single run: {run_dirs[0]}")
        return

    output_dir = args.path / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows).sort_values("run_dir")
    summary.to_csv(output_dir / "aggregate_summary.csv", index=False)
    plot_best_test_accuracy(summary, output_dir / "aggregate_best_test_accuracy.png")
    plot_gain(summary, output_dir / "aggregate_gain.png")
    plot_gain_vs_curriculum_epochs(summary, output_dir / "gain_vs_curriculum_epochs.png")
    write_overview(summary, output_dir / "aggregate_report.md")
    print(f"Wrote aggregate analysis to {output_dir}")


if __name__ == "__main__":
    main()
