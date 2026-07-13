#!/usr/bin/env python3
"""Regenerate selected thesis figures from existing local analysis exports.

This script is intentionally conservative: it does not query W&B/MLflow and it
uses only CSV exports already present in the repository.  It overwrites selected
PNG files in ``thesis/figures`` with more print-friendly versions.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.colors import SymLogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "thesis" / "figures"
REPORT = ROOT / "reports" / "thesis_deep_dive"
CTF = ROOT / "coarse-to-fine-curriculum"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 260,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = {
    "equal_epochs": "#4C78A8",
    "equal_time": "#F58518",
    "baseline": "#4C78A8",
    "random": "#F58518",
    "reverse": "#E45756",
    "self": "#72B7B2",
    "teacher": "#54A24B",
    "learned": "#F58518",
}
REGIME_LABELS = {"equal_epochs": "epoch-matched", "equal_time": "time-capped"}


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, bbox_inches="tight")
    plt.close()


def short_model(label: str) -> str:
    return (
        label.replace("cifar_resnet", "r")
        .replace("tiny-imagenet", "tiny")
        .replace("fashion-mnist", "fmnist")
    )


def pp(x: float) -> float:
    return 100.0 * float(x)


def trim_outer_whitespace(path: Path, tolerance: int = 10, pad: int = 20) -> None:
    """Crop excessive white margins from existing artifact composites."""
    image = Image.open(path).convert("RGB")
    background = Image.new("RGB", image.size, (255, 255, 255))
    diff = ImageChops.difference(image, background)
    mask = (np.asarray(diff) > tolerance).any(axis=2)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    box = (
        max(0, int(x0) - pad),
        max(0, int(y0) - pad),
        min(image.width, int(x1) + pad),
        min(image.height, int(y1) + pad),
    )
    if box != (0, 0, image.width, image.height):
        image.crop(box).save(path)


def regression_figures() -> None:
    delta = pd.read_csv(REPORT / "regression_curr_vs_single_deltas_enriched.csv")
    functions = list(delta[delta["regime"] == "equal_epochs"]["function"])
    y = np.arange(len(functions))
    offsets = {"equal_epochs": -0.18, "equal_time": 0.18}

    # Main headline quality bars.  A symmetric log x-scale prevents eggholder from
    # hiding the small-but-real deltas on smoother functions.
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for regime in ["equal_epochs", "equal_time"]:
        sub = delta[delta["regime"] == regime].set_index("function").loc[functions]
        vals = sub["delta_quality_single_minus_curriculum"].to_numpy()
        ax.barh(
            y + offsets[regime],
            vals,
            height=0.32,
            label=REGIME_LABELS[regime],
            color=COLORS[regime],
        )
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yticks(y)
    ax.set_yticklabels(functions)
    ax.invert_yaxis()
    ax.set_xlabel("Hard-validation loss delta: direct − curriculum (symlog)")
    ax.set_title("Quality effect by function")
    ax.legend(loc="lower right")
    save("curr_vs_single_quality_delta.png")

    # Runtime ratios are easier to compare than raw seconds across functions.
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for regime in ["equal_epochs", "equal_time"]:
        sub = delta[delta["regime"] == regime].set_index("function").loc[functions]
        vals = sub["runtime_ratio_curr_over_single"].to_numpy()
        ax.barh(
            y + offsets[regime],
            vals,
            height=0.32,
            label=REGIME_LABELS[regime],
            color=COLORS[regime],
        )
    ax.axvline(1, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(functions)
    ax.invert_yaxis()
    ax.set_xlabel("Median runtime ratio: curriculum / direct")
    ax.set_title("Runtime cost by function")
    ax.legend(loc="lower right")
    save("curr_vs_single_runtime_median.png")

    # Trade-off scatter with symmetric log y-scale.
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for regime, sub in delta.groupby("regime"):
        ax.scatter(
            sub["runtime_ratio_curr_over_single"],
            sub["delta_quality_single_minus_curriculum"],
            s=70,
            label=REGIME_LABELS.get(regime, regime.replace("_", " ")),
            color=COLORS.get(regime),
            alpha=0.88,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["function"],
                (
                    row["runtime_ratio_curr_over_single"],
                    row["delta_quality_single_minus_curriculum"],
                ),
                fontsize=7,
                xytext=(4, 3),
                textcoords="offset points",
            )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.axvline(1, color="black", linewidth=0.9)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Median runtime ratio: curriculum / direct")
    ax.set_ylabel("Hard-validation loss delta: direct − curriculum (symlog)")
    ax.set_title("Regression continuation: quality gain versus runtime cost")
    ax.legend(loc="lower left")
    save("deep_regression_quality_runtime_tradeoff.png")

    auc = pd.read_csv(REPORT / "regression_hardval_auc_delta.csv")
    # Support both old and explicit column names.
    delta_col = "auc_delta_single_minus_curriculum"
    if delta_col not in auc.columns:
        candidates = [c for c in auc.columns if "delta" in c.lower()]
        delta_col = candidates[0]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for regime in ["equal_epochs", "equal_time"]:
        sub = auc[auc["regime"] == regime].set_index("function").reindex(functions)
        ax.barh(
            y + offsets[regime],
            sub[delta_col].to_numpy(),
            height=0.32,
            label=REGIME_LABELS[regime],
            color=COLORS[regime],
        )
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(functions)
    ax.invert_yaxis()
    ax.set_xlabel("AUC delta: direct − curriculum (symlog; positive favors curriculum)")
    ax.set_title("Hard-target trajectory AUC deltas")
    ax.legend(loc="lower right")
    save("regression_hardval_auc_delta.png")

    heat = delta.pivot_table(
        index="function",
        columns="regime",
        values="delta_quality_single_minus_curriculum",
        aggfunc="first",
    ).loc[functions, ["equal_epochs", "equal_time"]]
    vals = heat.to_numpy()
    nonzero = np.abs(vals[np.isfinite(vals) & (vals != 0)])
    linthresh = max(1e-4, np.nanpercentile(nonzero, 25)) if len(nonzero) else 1e-3
    vmax = np.nanmax(np.abs(vals))
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    im = ax.imshow(
        vals,
        cmap="coolwarm",
        norm=SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax),
        aspect="auto",
    )
    fig.colorbar(im, ax=ax, label="Loss delta: direct − curriculum")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(
        [REGIME_LABELS.get(c, c.replace("_", " ")) for c in heat.columns], rotation=20, ha="right"
    )
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.2g}", ha="center", va="center", fontsize=7)
    ax.set_title("Function-level final-loss deltas")
    save("deep_regression_seed_delta_heatmap.png")


def meta_weighting_figures() -> None:
    source = (
        ROOT
        / "reports"
        / "analysis"
        / "meta-weighting-v1-eggholder__meta_weighting_eggholder_u1_20260504_221357"
        / "baseline_improvement_by_losses.csv"
    )
    if not source.exists():
        return

    data = pd.read_csv(source).sort_values("num_losses")
    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    ax.bar(
        data["num_losses"].astype(str),
        data["win_rate_vs_baseline"],
        color="#4C78A8",
        alpha=0.9,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.9)
    ax.set_ylim(0, 0.65)
    ax.set_xlabel("Number of continuation losses")
    ax.set_ylabel("Fraction improved over one-loss baseline")
    ax.set_title("eggholder: paired improvement frequency")
    save("meta_weighting_eggholder_u1_win_rate_vs_baseline.png")


def adaptive_and_dataset_figures() -> None:
    stage = pd.read_csv(CTF / "wandb_analysis_adaptive_followup" / "stage_summaries.csv")
    ad = stage[
        (stage["dataset"] == "cifar100") & stage["policy_label"].str.contains("adaptive", na=False)
    ].copy()
    ad_sum = ad.groupby(["model_label", "policy_label"], as_index=False).agg(
        coarse_epochs=("coarse_epochs", "mean"),
        fine_tune_epochs=("fine_tune_epochs", "mean"),
        n=("wandb_id", "count"),
    )
    ad_sum["row"] = (
        ad_sum["model_label"].map(short_model)
        + " / "
        + ad_sum["policy_label"].str.replace("adaptive_", "", regex=False)
    )
    ad_sum = ad_sum.sort_values(["model_label", "policy_label"])
    y = np.arange(len(ad_sum))
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.barh(y, ad_sum["coarse_epochs"], color="#E45756", label="coarse epochs")
    ax.barh(
        y,
        ad_sum["fine_tune_epochs"],
        left=ad_sum["coarse_epochs"],
        color="#4C78A8",
        label="fine-label epochs to peak",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(ad_sum["row"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean epochs up to peak test accuracy")
    ax.set_title("CIFAR-100 adaptive plateau stage allocation")
    ax.legend(loc="lower right")
    save("deep_adaptive_stage_allocation.png")

    agg = pd.read_csv(
        CTF / "wandb_analysis_roughness_followup_updated" / "aggregate_paired_deltas.csv"
    )
    plot = agg.copy()
    plot["gain_pp"] = 100 * plot["gain_best_mean"]
    plot["sd_pp"] = 100 * plot["gain_best_sd"].fillna(0)
    plot["row"] = (
        plot["dataset"].map(short_model)
        + " / "
        + plot["model_label"].map(short_model)
        + " / "
        + plot["label"]
    )
    plot = plot.sort_values(["dataset", "gain_pp"])
    colors = (
        plot["dataset"]
        .map(
            {
                "cifar100": "#4C78A8",
                "cifar10": "#72B7B2",
                "fashion-mnist": "#F58518",
                "tiny-imagenet": "#E45756",
            }
        )
        .fillna("#999999")
    )
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8.0, max(4.6, 0.30 * len(plot))))
    ax.barh(y, plot["gain_pp"], xerr=plot["sd_pp"].replace(0, np.nan), color=colors, alpha=0.9)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["row"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean best-test accuracy gain over baseline (pp)")
    ax.set_title("Dataset-control gains from existing W&B follow-up runs")
    save("roughness_followup_updated_all_datasets_best_acc_gains.png")


def hierarchy_figures() -> None:
    # Learned-vs-random hierarchy ablation.
    paired = pd.read_csv(
        CTF / "wandb_analysis_hierarchy_ablation" / "paired_hierarchy_deltas_by_seed.csv"
    )
    methods = [
        ("Learned hierarchy", "learned_gain", COLORS["learned"]),
        ("Random mean", "random_mean_gain", "#59A14F"),
        ("Best random", "random_best_of_seeds_gain", "#8CD17D"),
    ]
    means = [100 * paired[col].mean() for _, col, _ in methods]
    sds = [100 * paired[col].std(ddof=1) for _, col, _ in methods]
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=sds, capsize=4, color=[c for _, _, c in methods], alpha=0.9)
    for i, (_, col, _) in enumerate(methods):
        ax.scatter(
            np.full(len(paired), i) + np.linspace(-0.08, 0.08, len(paired)),
            100 * paired[col],
            color="black",
            s=18,
            zorder=3,
            alpha=0.75,
        )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in methods])
    ax.set_ylabel("Best-test accuracy gain over baseline (pp)")
    ax.set_title("CIFAR-100 weak-CNN hierarchy ablation")
    save("hierarchy_ablation_gain_comparison.png")

    # Learning curves for the same hierarchy ablation.
    runs = pd.read_csv(CTF / "wandb_analysis_hierarchy_ablation" / "runs_normalized.csv")
    hist = pd.read_csv(CTF / "wandb_export_hierarchy_ablation" / "wandb_history.csv")
    meta = runs[["wandb_id", "condition"]].drop_duplicates()
    hist = hist.merge(meta, on="wandb_id", how="left")
    curve_rows = []
    for condition, label in [
        ("baseline", "Baseline"),
        ("learned", "Learned hierarchy"),
        ("random", "Random hierarchy mean"),
    ]:
        sub = hist[hist["condition"] == condition]
        g = sub.groupby("epoch")["test_acc"].agg(["mean", "std", "count"]).reset_index()
        g["label"] = label
        curve_rows.append(g)
    curves = pd.concat(curve_rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for label, color in [
        ("Baseline", COLORS["baseline"]),
        ("Learned hierarchy", COLORS["learned"]),
        ("Random hierarchy mean", "#59A14F"),
    ]:
        sub = curves[curves["label"] == label]
        ax.plot(sub["epoch"], 100 * sub["mean"], label=label, color=color, linewidth=2)
        if sub["count"].max() > 1:
            ax.fill_between(
                sub["epoch"],
                100 * (sub["mean"] - sub["std"].fillna(0)),
                100 * (sub["mean"] + sub["std"].fillna(0)),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
    ax.axvline(20, color="black", linestyle="--", linewidth=0.9)
    ax.text(20.5, 5, "fine-label switch", fontsize=8, rotation=90, va="bottom")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Hierarchy ablation learning curves")
    ax.legend(loc="lower right")
    save("hierarchy_ablation_learning_curves.png")


def teacher_figures() -> None:
    out = CTF / "wandb_analysis_teacher_lr001_20260704b"
    aggregate = pd.read_csv(out / "aggregate_merged_comparison.csv")
    paired = pd.read_csv(out / "paired_merged_by_seed.csv")
    curves = pd.read_csv(out / "learning_curve_by_suite_condition.csv")
    runs_all = pd.read_csv(out / "runs_all.csv")

    method_map = {
        "Baseline": "Baseline",
        "Random hierarchy mean": "Random mean",
        "New teacher anti lr0.01": "Reverse-order teacher",
        "Self hierarchy": "Self hierarchy",
        "New teacher hierarchy lr0.01": "Teacher hierarchy",
    }
    order = list(method_map.keys())
    plot = aggregate[aggregate["method"].isin(order)].copy()
    plot["display"] = plot["method"].map(method_map)
    plot["sort"] = plot["method"].map({m: i for i, m in enumerate(order)})
    plot = plot.sort_values("sort")
    gains = 100 * plot["best_test_acc_gain_mean"].fillna(0)
    sd = 100 * plot["best_test_acc_gain_sd"].fillna(0)
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    colors = [
        COLORS["baseline"],
        COLORS["random"],
        COLORS["reverse"],
        COLORS["self"],
        COLORS["teacher"],
    ]
    x = np.arange(len(plot))
    ax.bar(x, gains, yerr=sd.replace(0, np.nan), capsize=4, color=colors, alpha=0.9)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(plot["display"], rotation=15, ha="right")
    ax.set_ylabel("Best-test accuracy gain over baseline (pp)")
    ax.set_title("CIFAR-100 teacher hierarchy controls")
    save("teacher_hierarchy_best_gain_bar_lr001.png")

    # Curves: merged suite selection matching the table above.
    curve_specs = [
        ("old_lr01_suite", "baseline", "Baseline", COLORS["baseline"]),
        ("old_lr01_suite", "random", "Random mean", COLORS["random"]),
        ("new_lr001_teacheronly", "teacher_anti", "Reverse-order teacher", COLORS["reverse"]),
        ("old_lr01_suite", "self", "Self hierarchy", COLORS["self"]),
        ("new_lr001_teacheronly", "teacher", "Teacher hierarchy", COLORS["teacher"]),
    ]

    def plot_curves(name: str, xlim: tuple[int, int] | None = None) -> None:
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        for suite, condition, label, color in curve_specs:
            sub = curves[(curves["suite"] == suite) & (curves["condition"] == condition)]
            ax.plot(sub["epoch"], 100 * sub["mean"], label=label, color=color, linewidth=2)
            ax.fill_between(
                sub["epoch"],
                100 * (sub["mean"] - sub["std"].fillna(0)),
                100 * (sub["mean"] + sub["std"].fillna(0)),
                color=color,
                alpha=0.10,
                linewidth=0,
            )
        ax.axvline(20, color="black", linestyle="--", linewidth=0.9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean test accuracy (%)")
        ax.set_title(
            "Recovery after coarse-to-fine switch" if xlim else "Teacher-control learning curves"
        )
        if xlim:
            ax.set_xlim(*xlim)
            ax.set_ylim(5, 39)
        ax.legend(loc="lower right")
        save(name)

    plot_curves("teacher_hierarchy_learning_curves_lr001.png")
    plot_curves("teacher_hierarchy_switch_zoom_lr001.png", xlim=(15, 50))

    # Epoch snapshots.
    epochs = [20, 21, 30, 50, 80]
    keys = [
        ("Baseline", "baseline", COLORS["baseline"]),
        ("Random mean", "random_mean", COLORS["random"]),
        ("Reverse-order", "new_anti", COLORS["reverse"]),
        ("Self", "self", COLORS["self"]),
        ("Teacher", "new_teacher", COLORS["teacher"]),
    ]
    fig, axes = plt.subplots(1, len(epochs), figsize=(9.4, 2.8), sharey=True)
    for ax, epoch in zip(axes, epochs, strict=True):
        vals = []
        for _, key, _ in keys:
            col = f"{key}_test_acc_epoch{epoch}"
            vals.append(100 * paired[col].mean())
        ax.bar(np.arange(len(keys)), vals, color=[c for _, _, c in keys], alpha=0.9)
        ax.set_title(f"Epoch {epoch}", fontsize=9)
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels([k[0] for k in keys], rotation=65, ha="right", fontsize=7)
        ax.axhline(
            100 * paired[f"baseline_test_acc_epoch{epoch}"].mean(),
            color="black",
            linewidth=0.7,
            alpha=0.45,
        )
    axes[0].set_ylabel("Mean test accuracy (%)")
    fig.suptitle("Teacher-control accuracy snapshots", y=1.04, fontsize=12)
    save("teacher_hierarchy_epoch_snapshots_lr001.png")

    # Per-seed gains with individual random controls.
    baseline_by_seed = paired.set_index("seed")["baseline_best_test_acc"].to_dict()
    random_runs = runs_all[runs_all["condition"] == "random"].copy()
    random_runs["gain_pp"] = random_runs.apply(
        lambda r: 100 * (r["best_test_acc"] - baseline_by_seed.get(r["seed"], np.nan)), axis=1
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(
        random_runs["seed"],
        random_runs["gain_pp"],
        color="0.65",
        s=28,
        label="Random individual",
        zorder=2,
    )
    seed_vals = paired["seed"].to_numpy()
    line_specs = [
        ("Random mean", "random_mean_best_test_acc_gain", COLORS["random"], "o"),
        ("Reverse-order", "new_anti_best_test_acc_gain", COLORS["reverse"], "s"),
        ("Self hierarchy", "self_best_test_acc_gain", COLORS["self"], "^"),
        ("Teacher hierarchy", "new_teacher_best_test_acc_gain", COLORS["teacher"], "D"),
    ]
    for label, col, color, marker in line_specs:
        ax.plot(
            seed_vals, 100 * paired[col], marker=marker, color=color, linewidth=1.8, label=label
        )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(seed_vals)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Best-test accuracy gain over baseline (pp)")
    ax.set_title("Per-seed teacher-control gains")
    ax.legend(ncol=2, loc="upper right")
    save("teacher_hierarchy_seed_gains_lr001.png")


def main() -> None:
    regression_figures()
    meta_weighting_figures()
    adaptive_and_dataset_figures()
    hierarchy_figures()
    teacher_figures()
    for name in [
        "mlflow_curr_vs_single_final_surfaces_seed42.png",
        "mlflow_curriculum_stage_progression_seed42.png",
        "mlflow_eggholder_snapshot_evolution_seed42.png",
    ]:
        trim_outer_whitespace(FIG / name)
    print("Updated selected thesis figures in", FIG)


if __name__ == "__main__":
    main()
