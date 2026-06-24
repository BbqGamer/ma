#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Pareto-style speed/accuracy/roughness summaries from run histories."
    )
    parser.add_argument("runs_root", type=Path, help="Root containing run_id directories")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoints", default="10,20,50,100")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def discover_history_dirs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("history.csv"))


def parse_run_label(run_id: str, mode: str, config: dict[str, Any]) -> str:
    if mode == "baseline":
        return "baseline"
    if mode == "curriculum":
        curr = read_json(Path(config.get("_run_dir", "")) / "results.json").get("curriculum_epochs")
        if curr is None:
            match = re.search(r"curr(\d+)", run_id)
            curr = match.group(1) if match else "unknown"
        return f"curr{curr}"
    if mode == "multiloss":
        return f"multiloss_{config.get('multi_weighting', 'unknown')}"
    return mode


def normalized_auc(history: pd.DataFrame, metric: str) -> float:
    if metric not in history or history.empty:
        return np.nan
    frame = history[["epoch", metric]].dropna()
    if frame.empty:
        return np.nan
    if len(frame) == 1:
        return float(frame[metric].iloc[0])
    duration = float(frame["epoch"].iloc[-1] - frame["epoch"].iloc[0])
    if duration <= 0:
        return float(frame[metric].mean())
    return float(np.trapezoid(frame[metric], frame["epoch"]) / duration)


def epoch_to_threshold(history: pd.DataFrame, metric: str, threshold: float) -> float:
    if metric not in history or pd.isna(threshold):
        return np.nan
    reached = history[history[metric] >= threshold]
    if reached.empty:
        return np.nan
    return float(reached["epoch"].iloc[0])


def value_at_epoch(history: pd.DataFrame, metric: str, epoch: int) -> float:
    if metric not in history or history.empty:
        return np.nan
    frame = history[history["epoch"] <= epoch]
    if frame.empty:
        return np.nan
    return float(frame[metric].iloc[-1])


def curve_roughness(history: pd.DataFrame, metric: str) -> float:
    if metric not in history:
        return np.nan
    values = history[metric].dropna().to_numpy(dtype=np.float64)
    if values.size < 3:
        return np.nan
    return float(np.mean(np.abs(np.diff(values, n=2))))


def max_drop(history: pd.DataFrame, metric: str) -> float:
    if metric not in history:
        return np.nan
    values = history[metric].dropna().to_numpy(dtype=np.float64)
    if values.size < 2:
        return np.nan
    best_so_far = np.maximum.accumulate(values)
    return float(np.max(best_so_far - values))


def mean_probe_metric(history: pd.DataFrame, metric: str) -> float:
    if metric not in history:
        return np.nan
    values = history[metric].dropna()
    return float(values.mean()) if not values.empty else np.nan


def summarize_history(run_dir: Path, checkpoints: list[int]) -> dict[str, Any]:
    history = pd.read_csv(run_dir / "history.csv")
    config = read_json(run_dir / "config.json")
    results = read_json(run_dir / "results.json")
    config["_run_dir"] = str(run_dir)
    run_id = str(config.get("run_id", run_dir.parent.name))
    mode = str(config.get("mode", results.get("mode", run_dir.name.split("_")[-1])))
    row: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "label": parse_run_label(run_id, mode, config),
        "dataset": config.get("dataset", results.get("dataset")),
        "model": config.get("model", results.get("model")),
        "seed": config.get("seed"),
        "mode": mode,
        "curriculum_epochs": results.get("curriculum_epochs"),
        "multi_weighting": results.get("multi_weighting", config.get("multi_weighting")),
        "epochs_completed": len(history),
        "best_val_acc": float(history["val_acc"].max()) if "val_acc" in history else np.nan,
        "best_test_acc": float(history["test_acc"].max()) if "test_acc" in history else np.nan,
        "final_test_acc": float(history["test_acc"].iloc[-1]) if "test_acc" in history else np.nan,
        "auc_val_acc": normalized_auc(history, "val_acc"),
        "auc_test_acc": normalized_auc(history, "test_acc"),
        "auc_test_f1_macro": normalized_auc(history, "test_f1_macro"),
        "auc_hier_score_official": normalized_auc(history, "test_hier_score_official"),
        "auc_hier_score_learned": normalized_auc(history, "test_hier_score_learned"),
        "test_acc_roughness": curve_roughness(history, "test_acc"),
        "val_acc_roughness": curve_roughness(history, "val_acc"),
        "max_test_acc_drop": max_drop(history, "test_acc"),
        "mean_rough_hessian_top_eigenvalue": mean_probe_metric(history, "rough_hessian_top_eigenvalue"),
        "mean_rough_hessian_frobenius": mean_probe_metric(history, "rough_hessian_frobenius"),
        "mean_rough_critical_sharpness": mean_probe_metric(history, "rough_critical_sharpness"),
        "mean_rough_gradient_noise_scale": mean_probe_metric(history, "rough_gradient_noise_scale"),
        "mean_rough_grad_norm_skew": mean_probe_metric(history, "rough_grad_norm_skew"),
    }
    for epoch in checkpoints:
        row[f"test_acc_epoch_{epoch}"] = value_at_epoch(history, "test_acc", epoch)
        row[f"val_acc_epoch_{epoch}"] = value_at_epoch(history, "val_acc", epoch)
    return row


def add_baseline_thresholds(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    group_cols = ["dataset", "model", "seed"]
    baselines = summary[summary["mode"] == "baseline"]
    base_best = baselines.groupby(group_cols)["best_test_acc"].max().rename("baseline_best_test_acc")
    summary = summary.merge(base_best, on=group_cols, how="left")
    summary["gain_best_test_acc"] = summary["best_test_acc"] - summary["baseline_best_test_acc"]
    summary["threshold_90pct_baseline"] = 0.90 * summary["baseline_best_test_acc"]
    summary["threshold_95pct_baseline"] = 0.95 * summary["baseline_best_test_acc"]
    summary["threshold_100pct_baseline"] = summary["baseline_best_test_acc"]

    # Compute threshold epochs from histories after baseline thresholds are known.
    for idx, row in summary.iterrows():
        history = pd.read_csv(Path(row["run_dir"]) / "history.csv")
        for pct in [90, 95, 100]:
            threshold = row[f"threshold_{pct}pct_baseline"]
            summary.loc[idx, f"epoch_to_{pct}pct_baseline"] = epoch_to_threshold(
                history,
                "test_acc",
                threshold,
            )
    return summary


def pareto_front(df: pd.DataFrame, objectives: list[tuple[str, str]]) -> pd.Series:
    valid = df[[name for name, _ in objectives]].notna().all(axis=1)
    front = pd.Series(False, index=df.index)
    candidates = df[valid]
    for idx, row in candidates.iterrows():
        dominated = False
        for other_idx, other in candidates.iterrows():
            if idx == other_idx:
                continue
            better_or_equal = []
            strictly_better = []
            for name, direction in objectives:
                if direction == "max":
                    better_or_equal.append(other[name] >= row[name])
                    strictly_better.append(other[name] > row[name])
                else:
                    better_or_equal.append(other[name] <= row[name])
                    strictly_better.append(other[name] < row[name])
            if all(better_or_equal) and any(strictly_better):
                dominated = True
                break
        front.loc[idx] = not dominated
    return front


def add_pareto_columns(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["pareto_accuracy_speed"] = False
    summary["pareto_accuracy_roughness"] = False
    group_cols = ["dataset", "model", "seed"]
    for _, group in summary.groupby(group_cols, dropna=False):
        idx = group.index
        speed_objectives = [
            ("best_test_acc", "max"),
            ("auc_test_acc", "max"),
            ("epoch_to_95pct_baseline", "min"),
        ]
        summary.loc[idx, "pareto_accuracy_speed"] = pareto_front(group, speed_objectives).values
        rough_objectives = [
            ("best_test_acc", "max"),
            ("mean_rough_hessian_top_eigenvalue", "min"),
            ("mean_rough_gradient_noise_scale", "min"),
        ]
        summary.loc[idx, "pareto_accuracy_roughness"] = pareto_front(group, rough_objectives).values
    return summary


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    output_path: Path,
    x_label: str | None = None,
    y_label: str | None = None,
) -> None:
    subset = df.dropna(subset=[x, y])
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = sorted(str(label) for label in subset["label"].dropna().unique())
    cmap = plt.get_cmap("tab10", max(len(labels), 1))
    label_to_color = {label: cmap(i) for i, label in enumerate(labels)}
    for label, frame in subset.groupby("label"):
        ax.scatter(frame[x], frame[y], label=label, s=55, alpha=0.85, color=label_to_color[str(label)])
        for _, row in frame.iterrows():
            ax.annotate(str(row["seed"]), (row[x], row[y]), fontsize=7, alpha=0.7)
    ax.set_xlabel(x_label or x)
    ax.set_ylabel(y_label or y)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, output_path: Path) -> None:
    lines = ["# Pareto analysis", ""]
    lines.append(f"Analyzed **{len(summary)}** runs.")
    lines.append("")
    if "gain_best_test_acc" in summary:
        lines.append("## Mean best-test gain by label")
        lines.append("")
        grouped = summary.groupby("label")["gain_best_test_acc"].agg(["mean", "std", "count"]).reset_index()
        for _, row in grouped.iterrows():
            std = 0.0 if pd.isna(row["std"]) else row["std"]
            lines.append(f"- {row['label']}: mean={row['mean']:.4f}, std={std:.4f}, n={int(row['count'])}")
        lines.append("")
    lines.append("## Pareto-front frequency")
    lines.append("")
    for col in ["pareto_accuracy_speed", "pareto_accuracy_roughness"]:
        if col in summary:
            grouped = summary.groupby("label")[col].agg(["mean", "sum", "count"]).reset_index()
            lines.append(f"### {col}")
            for _, row in grouped.iterrows():
                lines.append(
                    f"- {row['label']}: {int(row['sum'])}/{int(row['count'])} "
                    f"({100 * row['mean']:.1f}%)"
                )
            lines.append("")
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.runs_root / "pareto_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = [int(item.strip()) for item in args.checkpoints.split(",") if item.strip()]

    history_dirs = discover_history_dirs(args.runs_root)
    if not history_dirs:
        raise FileNotFoundError(f"No history.csv files found under {args.runs_root}")

    summary = pd.DataFrame(summarize_history(path, checkpoints) for path in history_dirs)
    summary = add_baseline_thresholds(summary)
    summary = add_pareto_columns(summary)
    summary = summary.sort_values(["dataset", "model", "seed", "label"])
    summary.to_csv(output_dir / "pareto_summary.csv", index=False)

    plot_scatter(
        summary,
        "auc_test_acc",
        "best_test_acc",
        output_dir / "accuracy_vs_auc.png",
        x_label="AUC test accuracy (higher is better)",
        y_label="Best test accuracy (higher is better)",
    )
    plot_scatter(
        summary,
        "epoch_to_95pct_baseline",
        "best_test_acc",
        output_dir / "accuracy_vs_epoch_to_95pct_baseline.png",
        x_label="Epoch to 95% of baseline best (lower is better)",
        y_label="Best test accuracy (higher is better)",
    )
    plot_scatter(
        summary,
        "mean_rough_hessian_top_eigenvalue",
        "best_test_acc",
        output_dir / "accuracy_vs_hessian_top_eigenvalue.png",
        x_label="Mean top Hessian eigenvalue (lower is better)",
        y_label="Best test accuracy (higher is better)",
    )
    plot_scatter(
        summary,
        "mean_rough_gradient_noise_scale",
        "best_test_acc",
        output_dir / "accuracy_vs_gradient_noise_scale.png",
        x_label="Mean gradient noise scale (lower is better)",
        y_label="Best test accuracy (higher is better)",
    )
    if "auc_hier_score_learned" in summary:
        plot_scatter(
            summary,
            "auc_hier_score_learned",
            "best_test_acc",
            output_dir / "accuracy_vs_hier_score_learned_auc.png",
            x_label="AUC learned hierarchy score (higher is better)",
            y_label="Best test accuracy (higher is better)",
        )
    write_report(summary, output_dir / "pareto_report.md")
    print(f"Wrote Pareto analysis to {output_dir}")


if __name__ == "__main__":
    main()
