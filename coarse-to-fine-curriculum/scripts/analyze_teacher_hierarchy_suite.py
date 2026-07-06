#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_pareto import normalized_auc  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze local teacher-hierarchy suite runs and write thesis-friendly tables."
    )
    parser.add_argument("runs_root", type=Path, help="Root containing run_id directories")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for analysis files (default: <runs_root>/analysis/teacher_hierarchy_suite).",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def model_label(config: dict[str, Any]) -> str:
    model = str(config.get("model", "unknown"))
    if model == "cnn":
        width = float(config.get("cnn_width_multiplier", 1.0))
        return f"cnn_w{width:g}"
    if model.startswith("cifar_resnet"):
        width = float(config.get("cifar_resnet_width_multiplier", 1.0))
        return f"{model}_w{width:g}" if width != 1.0 else model
    return model


def condition_label(run_id: str, config: dict[str, Any]) -> str:
    mode = str(config.get("mode", ""))
    if mode == "baseline" or run_id.endswith("-baseline"):
        return "baseline"
    if "-teacher-anti-" in run_id:
        return "teacher_anti"
    if "-teacher-curr" in run_id:
        return "teacher"
    if "-self-curr" in run_id:
        return "self"
    if "-random" in run_id:
        return "random"
    source = str(config.get("distance_source", ""))
    order = str(config.get("curriculum_order", "easy_to_hard"))
    if source == "teacher_embeddings" and order == "hard_to_easy":
        return "teacher_anti"
    if source == "teacher_embeddings":
        return "teacher"
    if source == "classifier_weights":
        return "self"
    if source == "random_permutation":
        return "random"
    return mode or "unknown"


def random_seed_from_run_id(run_id: str) -> float:
    marker = "-random"
    if marker not in run_id:
        return np.nan
    suffix = run_id.split(marker, 1)[1]
    digits = []
    for ch in suffix:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return float("".join(digits)) if digits else np.nan


def discover_run_dirs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("results.json") if path.parent.name.endswith(("_baseline", "_curriculum")))


def collect_runs(runs_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in discover_run_dirs(runs_root):
        config = read_json(run_dir / "config.json")
        results = read_json(run_dir / "results.json")
        history_path = run_dir / "history.csv"
        history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        run_id = str(config.get("run_id", run_dir.parent.name))
        rows.append(
            {
                "run_dir": str(run_dir),
                "run_id": run_id,
                "dataset": config.get("dataset", results.get("dataset")),
                "model_label": model_label(config),
                "seed": int(config.get("seed", -1)),
                "condition": condition_label(run_id, config),
                "distance_source": config.get("distance_source", results.get("distance_source", "")),
                "curriculum_order": config.get("curriculum_order", "easy_to_hard"),
                "curriculum_epochs": results.get("curriculum_epochs", config.get("curriculum_epochs", np.nan)),
                "random_hierarchy_seed": config.get("random_hierarchy_seed", random_seed_from_run_id(run_id)),
                "teacher_embedding_split": config.get("teacher_embedding_split", ""),
                "best_test_acc": float(results.get("best_test_acc", np.nan)),
                "final_test_acc": float(results.get("final_test_acc", np.nan)),
                "best_val_acc": float(results.get("best_val_acc", np.nan)),
                "auc_test_acc": normalized_auc(history, "test_acc"),
                "epochs_completed": float(results.get("epochs_completed", np.nan)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["dataset", "model_label", "seed", "condition", "run_id"]).reset_index(drop=True)


def summarize_suite(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baselines = runs[runs["condition"] == "baseline"].copy()
    baseline_map = {
        (row.dataset, row.model_label, int(row.seed)): row
        for row in baselines.itertuples(index=False)
    }

    curriculum = runs[runs["condition"].isin(["self", "teacher", "teacher_anti", "random"])].copy()
    baseline_rows = [
        baseline_map.get((row.dataset, row.model_label, int(row.seed)))
        for row in curriculum.itertuples(index=False)
    ]
    curriculum["baseline_best_test_acc"] = [row.best_test_acc if row is not None else np.nan for row in baseline_rows]
    curriculum["baseline_final_test_acc"] = [row.final_test_acc if row is not None else np.nan for row in baseline_rows]
    curriculum["baseline_auc_test_acc"] = [row.auc_test_acc if row is not None else np.nan for row in baseline_rows]
    curriculum = curriculum.dropna(subset=["baseline_best_test_acc"]).copy()
    curriculum["best_gain"] = curriculum["best_test_acc"] - curriculum["baseline_best_test_acc"]
    curriculum["final_gain"] = curriculum["final_test_acc"] - curriculum["baseline_final_test_acc"]
    curriculum["auc_gain"] = curriculum["auc_test_acc"] - curriculum["baseline_auc_test_acc"]

    paired_rows: list[dict[str, Any]] = []
    keys = sorted(
        set(zip(curriculum["dataset"], curriculum["model_label"], curriculum["seed"], strict=False))
    )
    for dataset, model_label_value, seed in keys:
        subset = curriculum[
            (curriculum["dataset"] == dataset)
            & (curriculum["model_label"] == model_label_value)
            & (curriculum["seed"] == seed)
        ]
        baseline = baseline_map.get((dataset, model_label_value, int(seed)))
        if baseline is None:
            continue

        def single(condition: str, metric: str) -> float:
            frame = subset[subset["condition"] == condition]
            if frame.empty:
                return np.nan
            return float(frame[metric].max())

        random_rows = subset[subset["condition"] == "random"]
        row = {
            "dataset": dataset,
            "model_label": model_label_value,
            "seed": int(seed),
            "curriculum_epochs": float(subset["curriculum_epochs"].dropna().iloc[0]) if subset["curriculum_epochs"].notna().any() else np.nan,
            "baseline_best_test_acc": float(baseline.best_test_acc),
            "baseline_final_test_acc": float(baseline.final_test_acc),
            "baseline_auc_test_acc": float(baseline.auc_test_acc),
        }
        for condition in ["self", "teacher", "teacher_anti"]:
            row[f"{condition}_best_test_acc"] = single(condition, "best_test_acc")
            row[f"{condition}_final_test_acc"] = single(condition, "final_test_acc")
            row[f"{condition}_auc_test_acc"] = single(condition, "auc_test_acc")
            row[f"{condition}_best_gain"] = single(condition, "best_gain")
            row[f"{condition}_final_gain"] = single(condition, "final_gain")
            row[f"{condition}_auc_gain"] = single(condition, "auc_gain")
        if not random_rows.empty:
            row["random_mean_best_test_acc"] = float(random_rows["best_test_acc"].mean())
            row["random_mean_final_test_acc"] = float(random_rows["final_test_acc"].mean())
            row["random_mean_auc_test_acc"] = float(random_rows["auc_test_acc"].mean())
            row["random_mean_best_gain"] = float(random_rows["best_gain"].mean())
            row["random_mean_final_gain"] = float(random_rows["final_gain"].mean())
            row["random_mean_auc_gain"] = float(random_rows["auc_gain"].mean())
            row["random_best_of_k_best_test_acc"] = float(random_rows["best_test_acc"].max())
            row["random_best_of_k_best_gain"] = float(random_rows["best_gain"].max())
            row["n_random_hierarchies"] = int(len(random_rows))
        else:
            row["random_mean_best_test_acc"] = np.nan
            row["random_mean_final_test_acc"] = np.nan
            row["random_mean_auc_test_acc"] = np.nan
            row["random_mean_best_gain"] = np.nan
            row["random_mean_final_gain"] = np.nan
            row["random_mean_auc_gain"] = np.nan
            row["random_best_of_k_best_test_acc"] = np.nan
            row["random_best_of_k_best_gain"] = np.nan
            row["n_random_hierarchies"] = 0
        row["teacher_minus_self_best_gain"] = row["teacher_best_gain"] - row["self_best_gain"]
        row["teacher_minus_random_mean_best_gain"] = row["teacher_best_gain"] - row["random_mean_best_gain"]
        row["teacher_minus_anti_best_gain"] = row["teacher_best_gain"] - row["teacher_anti_best_gain"]
        paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)

    aggregate_rows: list[dict[str, Any]] = []
    for (dataset, model_label_value, curriculum_epochs), group in paired.groupby(
        ["dataset", "model_label", "curriculum_epochs"], dropna=False
    ):
        method_map = {
            "Baseline": ("baseline_best_test_acc", "baseline_final_test_acc", "baseline_auc_test_acc", None, None, None),
            "Self hierarchy (weak)": ("self_best_test_acc", "self_final_test_acc", "self_auc_test_acc", "self_best_gain", "self_final_gain", "self_auc_gain"),
            "Teacher hierarchy": ("teacher_best_test_acc", "teacher_final_test_acc", "teacher_auc_test_acc", "teacher_best_gain", "teacher_final_gain", "teacher_auc_gain"),
            "Teacher anti-curriculum": ("teacher_anti_best_test_acc", "teacher_anti_final_test_acc", "teacher_anti_auc_test_acc", "teacher_anti_best_gain", "teacher_anti_final_gain", "teacher_anti_auc_gain"),
            "Random hierarchy mean": ("random_mean_best_test_acc", "random_mean_final_test_acc", "random_mean_auc_test_acc", "random_mean_best_gain", "random_mean_final_gain", "random_mean_auc_gain"),
            "Random hierarchy best-of-k": ("random_best_of_k_best_test_acc", None, None, "random_best_of_k_best_gain", None, None),
        }
        for method, columns in method_map.items():
            best_col, final_col, auc_col, best_gain_col, final_gain_col, auc_gain_col = columns
            row = {
                "dataset": dataset,
                "model_label": model_label_value,
                "curriculum_epochs": curriculum_epochs,
                "method": method,
                "n_seeds": int(group["seed"].nunique()),
                "best_test_acc_mean": float(group[best_col].mean()),
                "best_test_acc_sd": float(group[best_col].std(ddof=1)) if len(group) > 1 else 0.0,
                "final_test_acc_mean": float(group[final_col].mean()) if final_col else np.nan,
                "final_test_acc_sd": float(group[final_col].std(ddof=1)) if final_col and len(group) > 1 else (0.0 if final_col else np.nan),
                "auc_test_acc_mean": float(group[auc_col].mean()) if auc_col else np.nan,
                "auc_test_acc_sd": float(group[auc_col].std(ddof=1)) if auc_col and len(group) > 1 else (0.0 if auc_col else np.nan),
                "best_gain_mean": float(group[best_gain_col].mean()) if best_gain_col else np.nan,
                "best_gain_sd": float(group[best_gain_col].std(ddof=1)) if best_gain_col and len(group) > 1 else (0.0 if best_gain_col else np.nan),
                "final_gain_mean": float(group[final_gain_col].mean()) if final_gain_col else np.nan,
                "final_gain_sd": float(group[final_gain_col].std(ddof=1)) if final_gain_col and len(group) > 1 else (0.0 if final_gain_col else np.nan),
                "auc_gain_mean": float(group[auc_gain_col].mean()) if auc_gain_col else np.nan,
                "auc_gain_sd": float(group[auc_gain_col].std(ddof=1)) if auc_gain_col and len(group) > 1 else (0.0 if auc_gain_col else np.nan),
            }
            aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    return curriculum, paired, aggregate


def percent(value: float, signed: bool = False) -> str:
    if pd.isna(value):
        return "--"
    scaled = value * 100.0
    return f"{scaled:+.2f}" if signed else f"{scaled:.2f}"


def write_report(output_dir: Path, aggregate: pd.DataFrame, paired: pd.DataFrame) -> None:
    lines = ["# Teacher hierarchy suite", ""]
    if aggregate.empty:
        lines.append("No complete teacher hierarchy runs were found.")
        (output_dir / "REPORT.md").write_text("\n".join(lines))
        return
    for (dataset, model_label_value, curriculum_epochs), group in aggregate.groupby(
        ["dataset", "model_label", "curriculum_epochs"], dropna=False
    ):
        lines.append(f"## {dataset} / {model_label_value} / curr{int(curriculum_epochs)}")
        lines.append("")
        lines.append("| Method | Best acc. | Best gain | Final acc. | AUC gain |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in group.itertuples(index=False):
            lines.append(
                f"| {row.method} | {percent(row.best_test_acc_mean)} | {percent(row.best_gain_mean, signed=True)} | {percent(row.final_test_acc_mean)} | {percent(row.auc_gain_mean, signed=True)} |"
            )
        lines.append("")
        matched = paired[
            (paired["dataset"] == dataset)
            & (paired["model_label"] == model_label_value)
            & (paired["curriculum_epochs"] == curriculum_epochs)
        ]
        if not matched.empty:
            teacher_wins = int((matched["teacher_best_gain"] > matched["self_best_gain"]).sum())
            anti_wins = int((matched["teacher_best_gain"] > matched["teacher_anti_best_gain"]).sum())
            random_wins = int((matched["teacher_best_gain"] > matched["random_mean_best_gain"]).sum())
            lines.append(
                f"Teacher hierarchy beats self hierarchy on {teacher_wins}/{len(matched)} seeds, "
                f"beats anti-curriculum on {anti_wins}/{len(matched)} seeds, "
                f"and beats the random-hierarchy mean on {random_wins}/{len(matched)} seeds."
            )
            lines.append("")
    (output_dir / "REPORT.md").write_text("\n".join(lines))


def maybe_plot(output_dir: Path, aggregate: pd.DataFrame) -> None:
    if aggregate.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    preferred_order = [
        "Baseline",
        "Self hierarchy (weak)",
        "Teacher hierarchy",
        "Teacher anti-curriculum",
        "Random hierarchy mean",
        "Random hierarchy best-of-k",
    ]
    for (dataset, model_label_value, curriculum_epochs), group in aggregate.groupby(
        ["dataset", "model_label", "curriculum_epochs"], dropna=False
    ):
        group = group.set_index("method").reindex(preferred_order).dropna(how="all").reset_index()
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        x = np.arange(len(group))
        ax.bar(x, group["best_gain_mean"] * 100.0, color=["#4c78a8", "#72b7b2", "#54a24b", "#e45756", "#f58518", "#b279a2"][: len(group)])
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(group["method"], rotation=20, ha="right")
        ax.set_ylabel("Best test-accuracy gain (pp)")
        ax.set_title(f"Teacher hierarchy comparison: {dataset} {model_label_value} curr{int(curriculum_epochs)}")
        fig.tight_layout()
        out_name = f"teacher_hierarchy_gain_{dataset}_{model_label_value}_curr{int(curriculum_epochs)}.png".replace("/", "-")
        fig.savefig(output_dir / out_name, dpi=180)
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or (args.runs_root / "analysis" / "teacher_hierarchy_suite")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(args.runs_root)
    if runs.empty:
        raise FileNotFoundError(f"No teacher-hierarchy-compatible runs found under {args.runs_root}")

    curriculum, paired, aggregate = summarize_suite(runs)
    runs.to_csv(output_dir / "runs_normalized.csv", index=False)
    curriculum.to_csv(output_dir / "curriculum_with_baselines.csv", index=False)
    paired.to_csv(output_dir / "paired_by_seed.csv", index=False)
    aggregate.to_csv(output_dir / "comparison_table.csv", index=False)
    write_report(output_dir, aggregate, paired)
    maybe_plot(output_dir, aggregate)
    print(f"Wrote teacher hierarchy suite analysis to {output_dir}")


if __name__ == "__main__":
    main()
