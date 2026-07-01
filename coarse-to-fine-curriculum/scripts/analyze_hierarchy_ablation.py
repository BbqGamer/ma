#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze learned-vs-random hierarchy ablation runs exported from W&B."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
        help="Path to wandb_runs_summary.csv from scripts/export_wandb_results.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("wandb_analysis_hierarchy_ablation"),
    )
    return parser.parse_args()


def first_existing(row: pd.Series, names: list[str], default: Any = np.nan) -> Any:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def model_label(row: pd.Series) -> str:
    model = str(first_existing(row, ["config_model", "summary_model"], "unknown"))
    if model == "cnn":
        width = float(
            first_existing(
                row,
                ["config_cnn_width_multiplier", "summary_cnn_width_multiplier"],
                1.0,
            )
        )
        return f"cnn_w{width:g}"
    if model.startswith("cifar_resnet"):
        width = float(
            first_existing(
                row,
                ["config_cifar_resnet_width_multiplier", "summary_cifar_resnet_width_multiplier"],
                1.0,
            )
        )
        return f"{model}_w{width:g}" if width != 1.0 else model
    return model


def hierarchy_condition(row: pd.Series) -> str:
    mode = str(first_existing(row, ["config_mode", "summary_mode"], ""))
    if mode == "baseline":
        return "baseline"
    source = str(first_existing(row, ["summary_distance_source", "config_distance_source"], ""))
    if source == "random_permutation":
        return "random"
    if source == "classifier_weights":
        return "learned"
    return source or mode


def prepare_runs(summary: pd.DataFrame) -> pd.DataFrame:
    if "state" in summary.columns:
        summary = summary[summary["state"].fillna("finished") == "finished"].copy()
    rows = []
    for _, row in summary.iterrows():
        best = first_existing(row, ["summary_best_test_acc", "summary_test_acc"], np.nan)
        if pd.isna(best):
            continue
        rows.append(
            {
                "wandb_id": row.get("wandb_id", ""),
                "name": row.get("name", ""),
                "url": row.get("url", ""),
                "dataset": first_existing(row, ["config_dataset", "summary_dataset"], "unknown"),
                "model_label": model_label(row),
                "seed": int(first_existing(row, ["config_seed"], -1)),
                "condition": hierarchy_condition(row),
                "distance_source": first_existing(
                    row,
                    ["summary_distance_source", "config_distance_source"],
                    "",
                ),
                "random_hierarchy_seed": first_existing(
                    row,
                    ["summary_random_hierarchy_seed", "config_random_hierarchy_seed"],
                    np.nan,
                ),
                "curriculum_epochs": first_existing(
                    row,
                    ["summary_curriculum_epochs", "config_curriculum_epochs"],
                    np.nan,
                ),
                "best_test_acc": float(best),
                "final_test_acc": float(
                    first_existing(row, ["summary_final_test_acc", "summary_test_acc"], best)
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baselines = runs[runs["condition"] == "baseline"]
    baseline_map = {
        (row.dataset, row.model_label, int(row.seed)): float(row.best_test_acc)
        for row in baselines.itertuples(index=False)
    }

    curriculum = runs[runs["condition"].isin(["learned", "random"])].copy()
    curriculum["baseline_best_test_acc"] = [
        baseline_map.get((row.dataset, row.model_label, int(row.seed)), np.nan)
        for row in curriculum.itertuples(index=False)
    ]
    curriculum = curriculum.dropna(subset=["baseline_best_test_acc"])
    curriculum["best_acc_gain"] = (
        curriculum["best_test_acc"] - curriculum["baseline_best_test_acc"]
    )

    paired_rows = []
    keys = sorted(
        set(
            zip(
                curriculum["dataset"],
                curriculum["model_label"],
                curriculum["seed"],
                strict=False,
            )
        )
    )
    for dataset, model, seed in keys:
        subset = curriculum[
            (curriculum["dataset"] == dataset)
            & (curriculum["model_label"] == model)
            & (curriculum["seed"] == seed)
        ]
        learned = subset[subset["condition"] == "learned"]
        random_rows = subset[subset["condition"] == "random"]
        if learned.empty and random_rows.empty:
            continue
        baseline = baseline_map.get((dataset, model, int(seed)), np.nan)
        learned_best = float(learned["best_test_acc"].max()) if not learned.empty else np.nan
        learned_gain = learned_best - baseline if pd.notna(learned_best) else np.nan
        random_mean_best = (
            float(random_rows["best_test_acc"].mean()) if not random_rows.empty else np.nan
        )
        random_best_best = (
            float(random_rows["best_test_acc"].max()) if not random_rows.empty else np.nan
        )
        paired_rows.append(
            {
                "dataset": dataset,
                "model_label": model,
                "seed": int(seed),
                "baseline_best_test_acc": baseline,
                "learned_best_test_acc": learned_best,
                "learned_gain": learned_gain,
                "random_mean_best_test_acc": random_mean_best,
                "random_mean_gain": (
                    random_mean_best - baseline if pd.notna(random_mean_best) else np.nan
                ),
                "random_best_of_seeds_test_acc": random_best_best,
                "random_best_of_seeds_gain": (
                    random_best_best - baseline if pd.notna(random_best_best) else np.nan
                ),
                "random_best_test_acc_sd": float(random_rows["best_test_acc"].std(ddof=1))
                if len(random_rows) > 1
                else 0.0,
                "n_random_hierarchies": int(len(random_rows)),
            }
        )
    paired = pd.DataFrame(paired_rows)

    aggregate_rows = []
    for (dataset, model), group in paired.groupby(["dataset", "model_label"], dropna=False):
        aggregate_rows.append(
            {
                "dataset": dataset,
                "model_label": model,
                "n_seeds": int(group["seed"].nunique()),
                "mean_random_hierarchies_per_seed": float(group["n_random_hierarchies"].mean()),
                "baseline_best_mean": float(group["baseline_best_test_acc"].mean()),
                "learned_best_mean": float(group["learned_best_test_acc"].mean()),
                "learned_gain_mean": float(group["learned_gain"].mean()),
                "learned_gain_sd": (
                    float(group["learned_gain"].std(ddof=1)) if len(group) > 1 else 0.0
                ),
                "random_mean_best_mean": float(group["random_mean_best_test_acc"].mean()),
                "random_mean_gain_mean": float(group["random_mean_gain"].mean()),
                "random_mean_gain_sd": (
                    float(group["random_mean_gain"].std(ddof=1)) if len(group) > 1 else 0.0
                ),
                "random_best_gain_mean": float(group["random_best_of_seeds_gain"].mean()),
                "learned_minus_random_mean_gain": float(
                    (group["learned_gain"] - group["random_mean_gain"]).mean()
                ),
                "learned_beats_random_mean_seeds": int(
                    (group["learned_gain"] > group["random_mean_gain"]).sum()
                ),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    return curriculum, paired, aggregate


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, aggregate: pd.DataFrame, paired: pd.DataFrame) -> None:
    lines = ["# Hierarchy ablation analysis", ""]
    lines.append(
        "This compares the classifier-weight hierarchy against random permutations of the "
        "same hierarchy shape. The random condition keeps the number of levels and cluster "
        "sizes fixed, so the comparison is not just learned grouping versus a smaller label "
        "space."
    )
    lines.append("")
    if aggregate.empty:
        lines.append("No complete baseline/curriculum pairs were found.")
    else:
        table = aggregate.copy()
        for col in [
            "baseline_best_mean",
            "learned_best_mean",
            "learned_gain_mean",
            "random_mean_best_mean",
            "random_mean_gain_mean",
            "random_best_gain_mean",
            "learned_minus_random_mean_gain",
        ]:
            table[col] = (table[col] * 100).map(
                lambda value: f"{value:+.2f}" if "gain" in col else f"{value:.2f}"
            )
        lines.append(markdown_table(table))
        lines.append("")
        lines.append(
            "`random_best_gain_mean` is an optimistic diagnostic: it shows what happens if one "
            "could try several random hierarchies and keep the best one. The main comparison "
            "should be learned versus `random_mean_gain_mean`."
        )
    lines.append("")
    lines.append(f"Paired seed rows: {len(paired)}")
    (output_dir / "REPORT.md").write_text("\n".join(lines))


def maybe_plot(output_dir: Path, aggregate: pd.DataFrame) -> None:
    if aggregate.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    labels = [f"{row.dataset}\n{row.model_label}" for row in aggregate.itertuples(index=False)]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2.2), 4.5))
    ax.bar(x - width, aggregate["learned_gain_mean"] * 100, width, label="Learned")
    ax.bar(x, aggregate["random_mean_gain_mean"] * 100, width, label="Random mean")
    ax.bar(x + width, aggregate["random_best_gain_mean"] * 100, width, label="Random best")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Best test accuracy gain (pp)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    ax.set_title("Classifier-weight hierarchy vs random hierarchy")
    fig.tight_layout()
    fig.savefig(output_dir / "hierarchy_ablation_gain.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary_csv)
    runs = prepare_runs(summary)
    curriculum, paired, aggregate = summarize(runs)
    runs.to_csv(args.output_dir / "runs_normalized.csv", index=False)
    curriculum.to_csv(args.output_dir / "curriculum_with_baselines.csv", index=False)
    paired.to_csv(args.output_dir / "paired_hierarchy_deltas_by_seed.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_hierarchy_ablation.csv", index=False)
    write_report(args.output_dir, aggregate, paired)
    maybe_plot(args.output_dir, aggregate)
    print(f"Wrote hierarchy ablation analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
