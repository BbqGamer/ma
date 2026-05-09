#!/usr/bin/env python3
"""Aggregate MLflow results for the meta-weighting experiment matrix.

Outputs:
- per-run table
- grouped summary tables
- baseline comparison table (vs num_losses=1)
- markdown report
- a small set of aggregate plots
"""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import typer

from ma_thesis.config import PROJ_ROOT, REPORTS_DIR

MLFLOW_DB = PROJ_ROOT / "mlflow.db"
ANALYSIS_DIR = REPORTS_DIR / "analysis"

app = typer.Typer(add_completion=False)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip()).strip("_").lower()


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _to_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _iqr(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return math.nan
    return float(series.quantile(0.75) - series.quantile(0.25))


def _stderr(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) <= 1:
        return math.nan
    return float(series.std(ddof=1) / math.sqrt(len(series)))


def _tracking_uri() -> str:
    return f"sqlite:///{MLFLOW_DB}"


def _parse_csv_str(values: str) -> list[str]:
    return [v.strip() for v in values.split(",") if v.strip()]


def _load_runs(experiment_names: list[str]) -> pd.DataFrame:
    if not MLFLOW_DB.exists():
        raise typer.BadParameter(f"Missing MLflow DB: {MLFLOW_DB}")

    mlflow.set_tracking_uri(_tracking_uri())
    client = MlflowClient()
    experiment_ids: list[str] = []
    missing: list[str] = []
    for experiment_name in experiment_names:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            missing.append(experiment_name)
        else:
            experiment_ids.append(experiment.experiment_id)

    if missing:
        raise typer.BadParameter(f"Unknown MLflow experiment(s): {', '.join(missing)}")

    runs = mlflow.search_runs(
        experiment_ids=experiment_ids,
        output_format="pandas",
        search_all_experiments=False,
        max_results=50000,
    )
    if runs.empty:
        return runs

    return runs


def _extract_seed(run_name: str) -> int | None:
    match = re.search(r"__seed(\d+)__", str(run_name))
    if match:
        return int(match.group(1))
    return None


def _prepare_runs(runs: pd.DataFrame, benchmark_id: str | None) -> pd.DataFrame:
    df = runs.copy()

    if benchmark_id:
        run_name = df.get("tags.mlflow.runName", pd.Series("", index=df.index)).fillna("")
        df = df[run_name.astype(str).str.contains(benchmark_id, regex=False)].copy()

    if df.empty:
        return df

    df = df[df.get("status", "").astype(str).eq("FINISHED")].copy()
    if df.empty:
        return df

    rename_map = {
        "run_id": "run_id",
        "experiment_id": "experiment_id",
        "status": "status",
        "artifact_uri": "artifact_uri",
        "start_time": "start_time",
        "end_time": "end_time",
        "tags.mlflow.runName": "run_name",
        "params.function": "function",
        "params.seed": "seed",
        "params.lr_model": "lr_model",
        "params.lr_meta": "lr_meta",
        "params.momentum": "momentum",
        "params.lr_decay_gamma": "lr_decay_gamma",
        "params.num_sigma_levels": "num_losses",
        "params.available_num_sigma_levels": "available_num_losses",
        "params.noise_ratio": "noise_ratio",
        "params.selected_sigma_cols": "selected_sigma_cols",
        "params.inner_steps": "inner_steps",
        "params.meta_unroll_steps": "meta_unroll_steps",
        "params.batch_size": "batch_size",
        "params.epochs": "epochs",
        "metrics.train_loss": "final_train_loss",
        "metrics.meta_loss": "final_meta_loss",
        "metrics.meta_hard_val_loss": "final_meta_hard_val_loss",
        "metrics.weighted_val_loss": "final_weighted_val_loss",
        "metrics.weight_entropy": "final_weight_entropy",
        "metrics.effective_num_losses": "final_effective_num_losses",
        "metrics.lr_model": "final_logged_lr_model",
        "metrics.lr_meta": "final_logged_lr_meta",
    }
    keep_cols = [c for c in rename_map if c in df.columns]
    df = df[keep_cols].rename(columns={c: rename_map[c] for c in keep_cols})

    if "metrics.hard_val_loss" in runs.columns:
        df["final_val_loss"] = runs.loc[df.index, "metrics.hard_val_loss"]
    elif "metrics.val_loss" in runs.columns:
        df["final_val_loss"] = runs.loc[df.index, "metrics.val_loss"]
    else:
        df["final_val_loss"] = math.nan

    for col in [
        "lr_model",
        "lr_meta",
        "momentum",
        "lr_decay_gamma",
        "noise_ratio",
        "inner_steps",
        "meta_unroll_steps",
        "batch_size",
        "epochs",
        "seed",
        "final_val_loss",
        "final_train_loss",
        "final_meta_loss",
        "final_meta_hard_val_loss",
        "final_weighted_val_loss",
        "final_weight_entropy",
        "final_effective_num_losses",
        "final_logged_lr_model",
        "final_logged_lr_meta",
    ]:
        if col in df.columns:
            df[col] = _to_float(df[col])

    for col in ["seed", "num_losses", "available_num_losses"]:
        if col in df.columns:
            df[col] = _to_int(df[col])

    if "seed" not in df.columns:
        df["seed"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    if "run_name" in df.columns:
        missing_seed = df["seed"].isna()
        if missing_seed.any():
            df.loc[missing_seed, "seed"] = (
                df.loc[missing_seed, "run_name"].map(_extract_seed).astype("Int64")
            )

    if "start_time" in df.columns and "end_time" in df.columns:
        df["runtime_sec"] = (
            pd.to_datetime(df["end_time"]) - pd.to_datetime(df["start_time"])
        ).dt.total_seconds()
    else:
        df["runtime_sec"] = math.nan

    df["regime"] = df["num_losses"].apply(
        lambda x: "baseline" if pd.notna(x) and int(x) == 1 else "meta_weighting"
    )

    rank_group_cols = [
        c
        for c in ["function", "seed", "noise_ratio", "lr_model", "lr_meta", "meta_unroll_steps"]
        if c in df.columns
    ]
    df["rank_within_setting"] = (
        df.groupby(rank_group_cols, dropna=False)["final_val_loss"]
        .rank(method="average", ascending=True)
        .astype(float)
    )

    sort_cols = [
        c for c in ["function", "seed", "num_losses", "noise_ratio", "lr_model", "run_name"] if c in df
    ]
    return df.sort_values(sort_cols).reset_index(drop=True)



def _aggregate_runs(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    agg_spec: dict[str, tuple[str, str | Any]] = {
        "n": ("run_id", "count"),
        "mean_final_val_loss": ("final_val_loss", "mean"),
        "std_final_val_loss": ("final_val_loss", "std"),
        "median_final_val_loss": ("final_val_loss", "median"),
        "iqr_final_val_loss": ("final_val_loss", _iqr),
        "se_final_val_loss": ("final_val_loss", _stderr),
        "mean_runtime_sec": ("runtime_sec", "mean"),
        "median_runtime_sec": ("runtime_sec", "median"),
        "mean_rank_within_setting": ("rank_within_setting", "mean"),
        "median_rank_within_setting": ("rank_within_setting", "median"),
    }
    optional_metrics = [
        "final_train_loss",
        "final_meta_loss",
        "final_meta_hard_val_loss",
        "final_weighted_val_loss",
        "final_weight_entropy",
        "final_effective_num_losses",
    ]
    for metric in optional_metrics:
        if metric in df.columns:
            agg_spec[f"mean_{metric}"] = (metric, "mean")
            agg_spec[f"median_{metric}"] = (metric, "median")

    grouped = df.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()
    return grouped.sort_values(group_cols).reset_index(drop=True)



def _baseline_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    key_cols = ["function", "seed", "noise_ratio", "lr_model", "lr_meta", "meta_unroll_steps"]
    missing_cols = [col for col in key_cols if col not in df.columns]
    if missing_cols:
        raise typer.BadParameter(
            "Missing columns required for paired baseline comparison: "
            + ", ".join(missing_cols)
        )

    baseline = df[df["num_losses"] == 1][key_cols + ["run_id", "run_name", "final_val_loss"]].rename(
        columns={
            "run_id": "baseline_run_id",
            "run_name": "baseline_run_name",
            "final_val_loss": "baseline_final_val_loss",
        }
    )
    if baseline.empty:
        return pd.DataFrame()

    merged = df.merge(baseline, on=key_cols, how="left", validate="many_to_one")
    merged["improvement_vs_baseline"] = (
        merged["baseline_final_val_loss"] - merged["final_val_loss"]
    )
    merged["relative_improvement_vs_baseline"] = (
        merged["improvement_vs_baseline"] / merged["baseline_final_val_loss"]
    )
    merged["win_vs_baseline"] = (merged["improvement_vs_baseline"] > 0).astype(float)
    return merged.sort_values(["function", "seed", "noise_ratio", "lr_model", "num_losses"])



def _paired_improvement_summary(comp: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if comp.empty:
        return pd.DataFrame()

    filtered = comp[comp["num_losses"] > 1].dropna(subset=["improvement_vs_baseline"]).copy()
    if filtered.empty:
        return pd.DataFrame()

    summary = (
        filtered.groupby(group_cols, dropna=False)
        .agg(
            n=("run_id", "count"),
            mean_improvement_vs_baseline=("improvement_vs_baseline", "mean"),
            median_improvement_vs_baseline=("improvement_vs_baseline", "median"),
            iqr_improvement_vs_baseline=("improvement_vs_baseline", _iqr),
            se_improvement_vs_baseline=("improvement_vs_baseline", _stderr),
            mean_relative_improvement=("relative_improvement_vs_baseline", "mean"),
            median_relative_improvement=("relative_improvement_vs_baseline", "median"),
            win_rate_vs_baseline=("win_vs_baseline", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(group_cols).reset_index(drop=True)


def _to_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)



def _save_noise_plot(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return

    functions = [f for f in summary["function"].dropna().unique().tolist()]
    if not functions:
        return

    fig, axes = plt.subplots(len(functions), 1, figsize=(8, 4 * len(functions)), squeeze=False)
    for ax, function in zip(axes[:, 0], functions, strict=False):
        sub = summary[summary["function"] == function].copy()
        for num_losses in sorted(sub["num_losses"].dropna().unique().tolist()):
            cur = sub[sub["num_losses"] == num_losses].sort_values("noise_ratio")
            ax.plot(
                cur["noise_ratio"],
                cur["mean_final_val_loss"],
                marker="o",
                label=f"losses={int(num_losses)}",
            )
        ax.set_title(f"{function}: final val loss vs noise")
        ax.set_xlabel("noise_ratio")
        ax.set_ylabel("mean final val loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def _save_lr_plot(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return

    functions = [f for f in summary["function"].dropna().unique().tolist()]
    if not functions:
        return

    fig, axes = plt.subplots(len(functions), 1, figsize=(8, 4 * len(functions)), squeeze=False)
    for ax, function in zip(axes[:, 0], functions, strict=False):
        sub = summary[summary["function"] == function].copy()
        for num_losses in sorted(sub["num_losses"].dropna().unique().tolist()):
            cur = sub[sub["num_losses"] == num_losses].sort_values("lr_model")
            ax.plot(
                cur["lr_model"],
                cur["mean_final_val_loss"],
                marker="o",
                label=f"losses={int(num_losses)}",
            )
        ax.set_xscale("log")
        ax.set_title(f"{function}: final val loss vs learning rate")
        ax.set_xlabel("lr_model")
        ax.set_ylabel("mean final val loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def _save_rank_plot(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return

    functions = [f for f in summary["function"].dropna().unique().tolist()]
    if not functions:
        return

    fig, axes = plt.subplots(len(functions), 1, figsize=(8, 4 * len(functions)), squeeze=False)
    for ax, function in zip(axes[:, 0], functions, strict=False):
        sub = summary[summary["function"] == function].copy()
        sub = sub.sort_values("num_losses")
        ax.plot(
            sub["num_losses"],
            sub["mean_rank_within_setting"],
            marker="o",
            label="mean rank",
        )
        ax.plot(
            sub["num_losses"],
            sub["median_rank_within_setting"],
            marker="s",
            label="median rank",
        )
        ax.set_title(f"{function}: rank across paired settings (lower is better)")
        ax.set_xlabel("num_losses")
        ax.set_ylabel("rank within (seed, noise, lr)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def _save_win_rate_plot(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return

    functions = [f for f in summary["function"].dropna().unique().tolist()]
    if not functions:
        return

    fig, axes = plt.subplots(len(functions), 1, figsize=(8, 4 * len(functions)), squeeze=False)
    for ax, function in zip(axes[:, 0], functions, strict=False):
        sub = summary[summary["function"] == function].copy().sort_values("num_losses")
        ax.bar(sub["num_losses"].astype(str), sub["win_rate_vs_baseline"])
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{function}: paired win rate vs baseline")
        ax.set_xlabel("num_losses")
        ax.set_ylabel("win rate")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def _save_improvement_heatmaps(comp: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if comp.empty:
        return paths

    filtered = comp[comp["num_losses"] > 1].copy()
    if filtered.empty:
        return paths

    heatmap_summary = (
        filtered.groupby(["function", "num_losses", "noise_ratio", "lr_model"], dropna=False)
        .agg(mean_improvement_vs_baseline=("improvement_vs_baseline", "mean"))
        .reset_index()
    )

    for function in sorted(heatmap_summary["function"].dropna().unique().tolist()):
        sub_f = heatmap_summary[heatmap_summary["function"] == function]
        loss_values = sorted(sub_f["num_losses"].dropna().unique().tolist())
        if not loss_values:
            continue

        fig, axes = plt.subplots(1, len(loss_values), figsize=(5 * len(loss_values), 4), squeeze=False)
        for ax, num_losses in zip(axes[0], loss_values, strict=False):
            cur = sub_f[sub_f["num_losses"] == num_losses]
            pivot = cur.pivot(index="noise_ratio", columns="lr_model", values="mean_improvement_vs_baseline")
            if pivot.empty:
                ax.set_visible(False)
                continue
            im = ax.imshow(
                pivot.values,
                aspect="auto",
                origin="lower",
                cmap="RdBu_r",
                vmin=-np.nanmax(np.abs(pivot.values)),
                vmax=np.nanmax(np.abs(pivot.values)),
            )
            ax.set_title(f"{function}: losses={int(num_losses)}")
            ax.set_xlabel("lr_model")
            ax.set_ylabel("noise_ratio")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{c:g}" for c in pivot.columns], rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{idx:g}" for idx in pivot.index])
            fig.colorbar(im, ax=ax, shrink=0.85, label="mean improvement vs baseline")

        fig.tight_layout()
        path = output_dir / f"improvement_heatmap_{_slugify(function)}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths


@app.command()
def main(
    experiment_name: str = typer.Option(
        ...,
        help="MLflow experiment name(s) to analyze. Use commas to combine multiple experiments.",
    ),
    benchmark_id: str | None = typer.Option(
        None,
        help="Optional run-name prefix/filter from the benchmark runner.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        help="Optional output directory. Defaults to reports/analysis/<experiment>[_<benchmark>].",
    ),
) -> None:
    experiment_names = _parse_csv_str(experiment_name)
    if not experiment_names:
        raise typer.BadParameter("At least one experiment name is required.")

    runs = _load_runs(experiment_names)
    prepared = _prepare_runs(runs, benchmark_id)
    if prepared.empty:
        raise typer.BadParameter("No finished runs found for the requested filter.")

    output_dir = output_dir or ANALYSIS_DIR / (
        _slugify(experiment_name)
        if not benchmark_id
        else f"{_slugify(experiment_name)}__{_slugify(benchmark_id)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    by_setting = _aggregate_runs(
        prepared,
        ["function", "num_losses", "noise_ratio", "lr_model", "lr_meta", "meta_unroll_steps"],
    )
    by_noise = _aggregate_runs(prepared, ["function", "num_losses", "noise_ratio"])
    by_lr = _aggregate_runs(prepared, ["function", "num_losses", "lr_model", "lr_meta"])
    by_losses = _aggregate_runs(prepared, ["function", "num_losses"])
    comp = _baseline_comparison(prepared)
    comp_by_setting = _paired_improvement_summary(
        comp,
        ["function", "num_losses", "noise_ratio", "lr_model", "lr_meta", "meta_unroll_steps"],
    )
    comp_by_losses = _paired_improvement_summary(comp, ["function", "num_losses"])

    runs_path = output_dir / "runs.csv"
    by_setting_path = output_dir / "summary_by_setting.csv"
    by_noise_path = output_dir / "summary_by_noise.csv"
    by_lr_path = output_dir / "summary_by_lr.csv"
    by_losses_path = output_dir / "summary_by_losses.csv"
    comp_path = output_dir / "baseline_comparison.csv"
    comp_summary_path = output_dir / "baseline_improvement_by_setting.csv"
    comp_by_losses_path = output_dir / "baseline_improvement_by_losses.csv"

    prepared.to_csv(runs_path, index=False)
    by_setting.to_csv(by_setting_path, index=False)
    by_noise.to_csv(by_noise_path, index=False)
    by_lr.to_csv(by_lr_path, index=False)
    by_losses.to_csv(by_losses_path, index=False)
    comp.to_csv(comp_path, index=False)
    comp_by_setting.to_csv(comp_summary_path, index=False)
    comp_by_losses.to_csv(comp_by_losses_path, index=False)

    noise_plot = output_dir / "final_val_loss_vs_noise.png"
    lr_plot = output_dir / "final_val_loss_vs_lr.png"
    _save_noise_plot(by_noise, noise_plot)
    _save_lr_plot(by_lr, lr_plot)
    rank_plot = output_dir / "rank_vs_num_losses.png"
    win_rate_plot = output_dir / "win_rate_vs_baseline.png"
    _save_rank_plot(by_losses, rank_plot)
    _save_win_rate_plot(comp_by_losses, win_rate_plot)
    heatmaps = _save_improvement_heatmaps(comp, output_dir)

    best_runs = prepared.nsmallest(15, "final_val_loss")[
        [
            "run_name",
            "function",
            "num_losses",
            "noise_ratio",
            "lr_model",
            "lr_meta",
            "meta_unroll_steps",
            "final_val_loss",
            "runtime_sec",
        ]
    ]
    best_by_function = (
        prepared.sort_values("final_val_loss")
        .groupby("function", as_index=False)
        .first()[
            [
                "function",
                "run_name",
                "num_losses",
                "noise_ratio",
                "lr_model",
                "lr_meta",
                "meta_unroll_steps",
                "final_val_loss",
                "runtime_sec",
            ]
        ]
    )

    best_ranked_settings = by_losses.sort_values(["function", "mean_rank_within_setting", "median_final_val_loss"])[
        [
            "function",
            "num_losses",
            "n",
            "mean_final_val_loss",
            "median_final_val_loss",
            "mean_rank_within_setting",
            "median_rank_within_setting",
        ]
    ]

    report_path = output_dir / "summary.md"
    report = [
        f"# Meta-weighting analysis: `{', '.join(experiment_names)}`",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Benchmark filter: `{benchmark_id}`" if benchmark_id else "Benchmark filter: (none)",
        "",
        "## Run counts",
        "",
        f"- finished runs analyzed: **{len(prepared)}**",
        f"- experiments: **{len(experiment_names)}**",
        f"- functions: **{prepared['function'].nunique()}**",
        f"- loss-count settings: **{prepared['num_losses'].nunique()}**",
        f"- noise levels: **{prepared['noise_ratio'].nunique()}**",
        f"- learning rates: **{prepared['lr_model'].nunique()}**",
        "",
        "## Best runs overall",
        "",
        _to_md_table(best_runs),
        "",
        "## Best run per function",
        "",
        _to_md_table(best_by_function),
        "",
        "## Aggregate by function and number of losses",
        "",
        _to_md_table(by_losses),
        "",
        "## Rank summary by function and number of losses",
        "",
        _to_md_table(best_ranked_settings),
        "",
    ]

    if not comp.empty:
        report.extend(
            [
                "## Paired improvement over baseline (`num_losses=1`, matched by seed/noise/lr)",
                "",
                _to_md_table(comp_by_losses),
                "",
            ]
        )

    report.extend(
        [
            "## Output files",
            "",
            f"- `{runs_path.name}`",
            f"- `{by_setting_path.name}`",
            f"- `{by_noise_path.name}`",
            f"- `{by_lr_path.name}`",
            f"- `{by_losses_path.name}`",
            f"- `{comp_path.name}`",
            f"- `{comp_summary_path.name}`",
            f"- `{comp_by_losses_path.name}`",
            f"- `{noise_plot.name}`",
            f"- `{lr_plot.name}`",
            f"- `{rank_plot.name}`",
            f"- `{win_rate_plot.name}`",
            *[f"- `{path.name}`" for path in heatmaps],
            "",
        ]
    )
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(f"Analysis written to: {output_dir}")
    print(f"Summary report: {report_path}")


if __name__ == "__main__":
    app()
