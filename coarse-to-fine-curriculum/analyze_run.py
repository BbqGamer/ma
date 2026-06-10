from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one coarse-to-fine run directory")
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def find_mode_dir(run_dir: Path, suffix: str) -> Path:
    matches = sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.endswith(suffix))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '*{suffix}' directory inside {run_dir}, found {len(matches)}"
        )
    return matches[0]


def load_histories(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_dir = find_mode_dir(run_dir, "_baseline")
    curriculum_dir = find_mode_dir(run_dir, "_curriculum")
    baseline = pd.read_csv(baseline_dir / "history.csv")
    curriculum = pd.read_csv(curriculum_dir / "history.csv")
    return baseline, curriculum


def add_stage_boundaries(ax: plt.Axes, schedule: list[dict], max_epoch: int) -> None:
    start = 1
    for stage in schedule:
        epochs = int(stage["epochs"])
        if epochs <= 0:
            continue
        end = start + epochs - 1
        visible_end = min(end, max_epoch)
        if start > max_epoch:
            break
        ax.axvline(visible_end, color="#999999", linestyle=":", linewidth=1)
        label_x = (start + visible_end) / 2
        ax.text(
            label_x,
            1.01,
            stage["name"],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
        )
        start = end + 1


def plot_accuracy_curves(
    baseline: pd.DataFrame,
    curriculum: pd.DataFrame,
    schedule: list[dict],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    max_epoch = max(int(baseline["epoch"].max()), int(curriculum["epoch"].max()))

    if "train_acc" in baseline.columns:
        ax.plot(baseline["epoch"], baseline["train_acc"], label="baseline train", color="#9ecae1")
    ax.plot(baseline["epoch"], baseline["val_acc"], label="baseline val", color="#1f77b4")
    ax.plot(baseline["epoch"], baseline["test_acc"], label="baseline test", color="#1f77b4", linestyle="--")
    if "train_acc" in curriculum.columns:
        ax.plot(curriculum["epoch"], curriculum["train_acc"], label="curriculum train", color="#fcbba1")
    ax.plot(curriculum["epoch"], curriculum["val_acc"], label="curriculum val", color="#d62728")
    ax.plot(curriculum["epoch"], curriculum["test_acc"], label="curriculum test", color="#d62728", linestyle="--")
    add_stage_boundaries(ax, schedule, max_epoch)
    ax.set_xlim(1, max_epoch)
    ax.set_title("Accuracy per observed epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(
    baseline: pd.DataFrame,
    curriculum: pd.DataFrame,
    schedule: list[dict],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    max_epoch = max(int(baseline["epoch"].max()), int(curriculum["epoch"].max()))

    ax.plot(baseline["epoch"], baseline["train_loss"], label="baseline train loss", color="#1f77b4")
    ax.plot(baseline["epoch"], baseline["val_loss"], label="baseline val loss", color="#1f77b4", linestyle="--")
    ax.plot(curriculum["epoch"], curriculum["train_loss"], label="curriculum train loss", color="#d62728")
    ax.plot(curriculum["epoch"], curriculum["val_loss"], label="curriculum val loss", color="#d62728", linestyle="--")
    add_stage_boundaries(ax, schedule, max_epoch)
    ax.set_xlim(1, max_epoch)
    ax.set_title("Loss per observed epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    baseline_dir: Path,
    curriculum_dir: Path,
    output_path: Path,
) -> bool:
    baseline_path = baseline_dir / "confusion_test_normalized.csv"
    curriculum_path = curriculum_dir / "confusion_test_normalized.csv"
    if not baseline_path.exists() or not curriculum_path.exists():
        return False

    baseline_conf = np.loadtxt(baseline_path, delimiter=",")
    curriculum_conf = np.loadtxt(curriculum_path, delimiter=",")
    vmax = max(float(baseline_conf.max()), float(curriculum_conf.max()), 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, matrix, title in [
        (axes[0], baseline_conf, "Baseline test confusion"),
        (axes[1], curriculum_conf, "Curriculum test confusion"),
    ]:
        im = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_class_accuracy_gain(
    baseline_dir: Path,
    curriculum_dir: Path,
    output_path: Path,
) -> bool:
    baseline_path = baseline_dir / "class_metrics_test.csv"
    curriculum_path = curriculum_dir / "class_metrics_test.csv"
    if not baseline_path.exists() or not curriculum_path.exists():
        return False

    baseline = pd.read_csv(baseline_path)
    curriculum = pd.read_csv(curriculum_path)
    merged = baseline[["class_idx", "class_name", "accuracy"]].merge(
        curriculum[["class_idx", "accuracy"]],
        on="class_idx",
        suffixes=("_baseline", "_curriculum"),
    )
    merged["gain"] = merged["accuracy_curriculum"] - merged["accuracy_baseline"]
    merged = merged.sort_values("gain", ascending=False)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    colors = ["#2ca02c" if gain >= 0 else "#d62728" for gain in merged["gain"]]
    ax.bar(range(len(merged)), merged["gain"], color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Per-class test accuracy gain (curriculum - baseline)")
    ax.set_xlabel("Classes sorted by gain")
    ax.set_ylabel("Accuracy gain")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def build_difficulty_summary(baseline_dir: Path, curriculum_dir: Path, output_dir: Path) -> bool:
    baseline_path = baseline_dir / "difficulty_metrics_test.csv"
    curriculum_path = curriculum_dir / "difficulty_metrics_test.csv"
    if not baseline_path.exists() or not curriculum_path.exists():
        return False

    baseline = pd.read_csv(baseline_path).iloc[0].to_dict()
    curriculum = pd.read_csv(curriculum_path).iloc[0].to_dict()
    rows = []
    for key in baseline:
        if key not in curriculum:
            continue
        try:
            baseline_val = float(baseline[key])
            curriculum_val = float(curriculum[key])
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "metric": key,
                "baseline": baseline_val,
                "curriculum": curriculum_val,
                "difference": curriculum_val - baseline_val,
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "difficulty_summary_test.csv", index=False)
    return True


def summarize_history(name: str, df: pd.DataFrame) -> dict[str, float | int | str]:
    best_val_idx = df["val_acc"].idxmax()
    best_test_idx = df["test_acc"].idxmax()
    return {
        "run": name,
        "epochs_completed": int(len(df)),
        "best_val_epoch": int(df.loc[best_val_idx, "epoch"]),
        "best_val_acc": float(df.loc[best_val_idx, "val_acc"]),
        "test_acc_at_best_val": float(df.loc[best_val_idx, "test_acc"]),
        "best_test_epoch": int(df.loc[best_test_idx, "epoch"]),
        "best_test_acc": float(df.loc[best_test_idx, "test_acc"]),
        "final_epoch": int(df.iloc[-1]["epoch"]),
        "final_val_acc": float(df.iloc[-1]["val_acc"]),
        "final_test_acc": float(df.iloc[-1]["test_acc"]),
    }


def cluster_names(cluster: list[int], class_names: list[str]) -> list[str]:
    return [class_names[idx] for idx in cluster]


def write_report(
    output_path: Path,
    run_name: str,
    baseline_summary: dict,
    curriculum_summary: dict,
    schedule: list[dict],
    hierarchy: dict,
    first_curriculum_beat_baseline_test_epoch: int | None,
    first_curriculum_beat_baseline_val_epoch: int | None,
) -> None:
    abs_gain = curriculum_summary["best_test_acc"] - baseline_summary["best_test_acc"]
    rel_gain = abs_gain / baseline_summary["best_test_acc"] if baseline_summary["best_test_acc"] else 0.0

    coarse_levels = hierarchy["levels"][:3]
    class_names = hierarchy.get("class_names") or [str(i) for i in range(100)]
    lines = [
        f"# Analysis: {run_name}",
        "",
        "## Headline",
        "",
        f"- Baseline best test accuracy: **{baseline_summary['best_test_acc']:.4f}** at epoch {baseline_summary['best_test_epoch']}",
        f"- Curriculum best test accuracy: **{curriculum_summary['best_test_acc']:.4f}** at epoch {curriculum_summary['best_test_epoch']}",
        f"- Absolute gain: **{abs_gain:.4f}** ({abs_gain * 100:.2f} pp)",
        f"- Relative gain over baseline: **{rel_gain * 100:.2f}%**",
        "",
        "## Interpretation",
        "",
        "- This run is a clear positive result for the curriculum variant." if abs_gain > 0 else "- This run does not improve over the baseline.",
        "- The curriculum reaches baseline-level performance quickly after switching to fine labels and then surpasses it." if abs_gain > 0 else "- The curriculum does not clearly separate from the baseline in this run.",
        (
            f"- In this run, curriculum exceeds the baseline's best test accuracy by epoch {first_curriculum_beat_baseline_test_epoch} and the baseline's best validation accuracy by epoch {first_curriculum_beat_baseline_val_epoch}."
            if first_curriculum_beat_baseline_test_epoch is not None and first_curriculum_beat_baseline_val_epoch is not None
            else "- The curriculum never clearly exceeded the baseline peak during the recorded epochs."
        ),
        "- Both runs were early-stopped, so the `final_*` metrics in `results.json` come from the restored best checkpoint, not from the last row of `history.csv`.",
        "- The plots now stop at the last observed epoch, not the planned 400-epoch budget.",
        "- The accuracy plot shows validation and test accuracy only; train accuracy is not logged in the current training script.",
        "- The last row of each `history.csv` is worse than the best epoch, which suggests some overfitting / instability after the peak.",
        "- Curriculum train loss is not directly comparable to curriculum val loss during coarse stages, because train loss uses the stage-specific marginalized curriculum objective while val loss stays on the final fine-label task.",
        "- `level_4` already uses 100 singleton classes, so it is effectively an extra fine-label stage before `fine_tune`. That is worth keeping in mind when interpreting the nominal `curriculum_epochs=13`.",
        "",
        "## Schedule",
        "",
    ]
    for stage in schedule:
        lines.append(f"- {stage['name']}: {stage['epochs']} epochs, {len(stage['clusters'])} clusters")

    lines.extend([
        "",
        "## Coarsest hierarchy levels (class names)",
        "",
    ])

    for level_idx, level in enumerate(coarse_levels, start=1):
        lines.append(f"### Level {level_idx}: {len(level)} clusters")
        for cluster_idx, cluster in enumerate(level, start=1):
            names = ", ".join(cluster_names(cluster, class_names))
            lines.append(f"- Cluster {cluster_idx}: {names}")
        lines.append("")

    output_path.write_text("\n".join(lines))


def resolve_run_dir(run_dir: Path) -> Path:
    if any(child.is_dir() and child.name.endswith("_baseline") for child in run_dir.iterdir()):
        return run_dir
    nested = sorted(
        child
        for child in run_dir.iterdir()
        if child.is_dir() and any(grandchild.is_dir() and grandchild.name.endswith("_baseline") for grandchild in child.iterdir())
    )
    if len(nested) == 1:
        return nested[0]
    return run_dir


def analyze_run(run_dir: Path) -> Path:
    run_dir = resolve_run_dir(run_dir)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = find_mode_dir(run_dir, "_baseline")
    curriculum_dir = find_mode_dir(run_dir, "_curriculum")
    baseline, curriculum = load_histories(run_dir)
    schedule = json.loads((curriculum_dir / "schedule.json").read_text())
    hierarchy = json.loads((curriculum_dir / "hierarchy.json").read_text())

    baseline_summary = summarize_history("baseline", baseline)
    curriculum_summary = summarize_history("curriculum", curriculum)
    summary_df = pd.DataFrame([baseline_summary, curriculum_summary])
    summary_df.to_csv(analysis_dir / "comparison_summary.csv", index=False)

    first_curriculum_beat_baseline_test = curriculum[
        curriculum["test_acc"] >= baseline_summary["best_test_acc"]
    ].head(1)
    first_curriculum_beat_baseline_val = curriculum[
        curriculum["val_acc"] >= baseline_summary["best_val_acc"]
    ].head(1)
    beat_test_epoch = None if first_curriculum_beat_baseline_test.empty else int(first_curriculum_beat_baseline_test.iloc[0]["epoch"])
    beat_val_epoch = None if first_curriculum_beat_baseline_val.empty else int(first_curriculum_beat_baseline_val.iloc[0]["epoch"])

    plot_accuracy_curves(
        baseline,
        curriculum,
        schedule,
        analysis_dir / "accuracy_curves.png",
    )
    plot_loss_curves(
        baseline,
        curriculum,
        schedule,
        analysis_dir / "loss_curves.png",
    )
    plot_confusion_matrices(
        baseline_dir,
        curriculum_dir,
        analysis_dir / "confusion_matrices_test.png",
    )
    plot_class_accuracy_gain(
        baseline_dir,
        curriculum_dir,
        analysis_dir / "per_class_accuracy_gain_test.png",
    )
    build_difficulty_summary(
        baseline_dir,
        curriculum_dir,
        analysis_dir,
    )
    write_report(
        analysis_dir / "report.md",
        run_dir.name,
        baseline_summary,
        curriculum_summary,
        schedule,
        hierarchy,
        beat_test_epoch,
        beat_val_epoch,
    )

    print(f"Wrote analysis to {analysis_dir}")
    return analysis_dir


def main() -> None:
    args = parse_args()
    analyze_run(args.run_dir)


if __name__ == "__main__":
    main()
