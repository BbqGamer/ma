#!/usr/bin/env python3
"""Analyze matched LLM vs Optuna schedule-search results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import typer

from ma_thesis.config import REPORTS_DIR

app = typer.Typer(add_completion=False)

ANALYSIS_ROOT = REPORTS_DIR / "analysis" / "schedule_search_comparison"
LLM_ROOT = REPORTS_DIR / "llm_schedule_search"
POLICY_ROOT = REPORTS_DIR / "benchmarks" / "policy"
OPTUNA_ROOT = REPORTS_DIR / "benchmarks" / "optuna_schedule"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _format_float(value: float | None, digits: int = 2) -> str:
    if value is None or math.isnan(value):
        return "—"
    return f"{value:.{digits}f}"


def _format_int(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{value:,}"


def _extract_failure_type(error: str | None) -> str:
    if not error:
        return "unknown"
    if "insufficient_quota" in error:
        return "quota"
    if "TypeError" in error:
        return "type_error"
    if "ValueError" in error:
        return "value_error"
    return "other"


def _load_llm_candidates(study_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    history_path = LLM_ROOT / study_name / "history.jsonl"
    token_path = LLM_ROOT / study_name / "token_summary.json"
    rows = _load_jsonl(history_path)
    for idx, row in enumerate(rows, start=1):
        row["eval_index"] = idx
        row["method"] = "llm"
        row["objective"] = _safe_float(row.get("mean_best_hard_val_loss"))
        row["final_objective"] = _safe_float(row.get("mean_final_hard_val_loss"))
        row["success"] = row.get("status") == "ok" and row["objective"] is not None
        row["failure_type"] = _extract_failure_type(row.get("error"))
        row["openai_total_tokens"] = int(float(row.get("openai_total_tokens", 0) or 0))
    return pd.DataFrame(rows), _load_json(token_path)


def _load_optuna_trials(benchmark_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    trials_path = OPTUNA_ROOT / benchmark_id / "all_trials.jsonl"
    aggregate_path = OPTUNA_ROOT / benchmark_id / "aggregate.json"
    rows = _load_jsonl(trials_path)
    for row in rows:
        row["eval_index"] = int(row["trial_number"]) + 1
        row["method"] = "optuna"
        row["objective"] = _safe_float(row.get("mean_best_hard_val_loss"))
        row["final_objective"] = _safe_float(row.get("mean_final_hard_val_loss"))
        row["success"] = row.get("status") == "ok" and row["objective"] is not None
    return pd.DataFrame(rows), _load_json(aggregate_path)


def _best_success_row(df: pd.DataFrame) -> pd.Series:
    ok = df[df["success"]].copy()
    if ok.empty:
        raise ValueError("No successful rows found.")
    return ok.sort_values("objective", ascending=True).iloc[0]


def _best_so_far(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    best = math.inf
    for _, row in df.sort_values("eval_index").iterrows():
        objective = row["objective"]
        if objective is not None and not math.isnan(objective):
            best = min(best, float(objective))
        rows.append(
            {
                "eval_index": int(row["eval_index"]),
                "best_so_far": None if math.isinf(best) else float(best),
            }
        )
    return pd.DataFrame(rows)


def _load_seed_results(path: Path) -> pd.DataFrame:
    rows = _load_jsonl(path)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in ["best_hard_val_loss", "final_hard_val_loss", "epochs_trained", "seed"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _load_trajectory_frame(label: str, seed_results: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, row in seed_results.iterrows():
        traj_path = row.get("trajectory_path")
        if not traj_path:
            continue
        df = pl.read_parquet(traj_path).to_pandas()
        df["seed"] = int(row["seed"])
        df["method"] = label
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _aggregate_trajectory(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["epoch", "median", "q25", "q75"])
    grouped = (
        df.groupby("epoch")[value_col]
        .agg(
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
        )
        .reset_index()
        .sort_values("epoch")
    )
    return grouped


def _plot_best_so_far(llm_df: pd.DataFrame, optuna_df: pd.DataFrame, out_path: Path) -> None:
    llm_prog = _best_so_far(llm_df)
    optuna_prog = _best_so_far(optuna_df)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(
        llm_prog["eval_index"],
        llm_prog["best_so_far"],
        marker="o",
        linewidth=2,
        label="LLM policy search",
    )
    ax.plot(
        optuna_prog["eval_index"],
        optuna_prog["best_so_far"],
        marker="s",
        linewidth=2,
        label="Optuna schedule search",
    )
    ax.set_xlabel("Evaluation / trial index")
    ax.set_ylabel("Best mean hard validation loss so far")
    ax.set_title("Search progress")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_objective_distribution(
    llm_df: pd.DataFrame,
    optuna_df: pd.DataFrame,
    out_path: Path,
) -> None:
    llm_ok = llm_df[llm_df["success"]]["objective"].dropna().tolist()
    optuna_ok = optuna_df[optuna_df["success"]]["objective"].dropna().tolist()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.boxplot([llm_ok, optuna_ok], tick_labels=["LLM", "Optuna"], widths=0.5)
    for xpos, values, marker in [(1, llm_ok, "o"), (2, optuna_ok, "s")]:
        if not values:
            continue
        spread = pd.Series(range(len(values)), dtype=float)
        spread = (spread - spread.mean()) / max(1.0, len(values) - 1)
        jitter = 0.08 * spread.to_numpy()
        ax.scatter([xpos + j for j in jitter], values, alpha=0.75, s=28, marker=marker)
    ax.set_ylabel("Mean best hard validation loss")
    ax.set_title("Distribution of completed evaluations")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_per_seed_comparison(
    llm_seed_df: pd.DataFrame,
    optuna_seed_df: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    llm_plot = llm_seed_df[["seed", "best_hard_val_loss"]].rename(
        columns={"best_hard_val_loss": "llm_best_hard_val_loss"}
    )
    optuna_plot = optuna_seed_df[["seed", "best_hard_val_loss"]].rename(
        columns={"best_hard_val_loss": "optuna_best_hard_val_loss"}
    )
    merged = llm_plot.merge(optuna_plot, on="seed", how="outer").sort_values("seed")

    x = range(len(merged))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.bar(
        [v - width / 2 for v in x],
        merged["llm_best_hard_val_loss"],
        width=width,
        label="LLM best candidate",
    )
    ax.bar(
        [v + width / 2 for v in x],
        merged["optuna_best_hard_val_loss"],
        width=width,
        label="Optuna best trial",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"seed {int(v)}" for v in merged["seed"]])
    ax.set_ylabel("Best hard validation loss")
    ax.set_title("Per-seed comparison of the best schedules")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return merged


def _mean_trajectory_over_seeds(seed_results: pd.DataFrame) -> pd.DataFrame:
    traj = _load_trajectory_frame("tmp", seed_results)
    if traj.empty:
        return pd.DataFrame()
    value_cols = [col for col in traj.columns if col != "seed" and col != "method"]
    return traj.groupby("epoch", as_index=False)[value_cols].mean().sort_values("epoch")


def _plot_schedule_gallery(
    rows: pd.DataFrame,
    *,
    kind: str,
    out_path: Path,
    benchmark_id: str | None = None,
    n_weights: int = 4,
    per_page: int = 5,
) -> list[Path]:
    ok = rows[rows["success"]].copy().sort_values("objective", ascending=True)
    if ok.empty:
        return []

    weight_cols = [f"weight_{i}" for i in range(n_weights)]
    weight_labels = [
        f"sigma {i} ({'easiest' if i == 0 else 'hardest' if i == n_weights - 1 else 'middle'})"
        for i in range(n_weights)
    ]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    title = "LLM candidate schedules" if kind == "llm" else "Optuna trial schedules"
    all_items = _collect_mean_trajectories(ok, kind=kind, optuna_benchmark_id=benchmark_id)
    all_loss_trajs = [traj for _, traj in all_items]
    loss_lo, loss_hi = _loss_axis_limits(all_loss_trajs)
    out_paths: list[Path] = []

    for page_idx, start in enumerate(range(0, len(ok), per_page), start=1):
        chunk = ok.iloc[start : start + per_page]
        n = len(chunk)
        fig, axes = plt.subplots(n, 1, figsize=(8.8, max(2.5 * n, 6.0)), sharex=True)
        if n == 1:
            axes = [axes]

        for rank, ((_, row), ax) in enumerate(zip(chunk.iterrows(), axes), start=start + 1):
            seed_path = _seed_results_path(kind, row, optuna_benchmark_id=benchmark_id)
            run_id = str(row["candidate_id"])

            if not seed_path.exists():
                ax.set_visible(False)
                continue
            seed_df = _load_seed_results(seed_path)
            mean_traj = _mean_trajectory_over_seeds(seed_df)
            if mean_traj.empty:
                ax.set_visible(False)
                continue

            for idx, (col, label) in enumerate(zip(weight_cols, weight_labels)):
                ax.plot(
                    mean_traj["epoch"],
                    mean_traj[col],
                    color=colors[idx % len(colors)],
                    linewidth=1.8,
                    label=label,
                )

            loss_ax = ax.twinx()
            loss_ax.plot(
                mean_traj["epoch"],
                _smooth_series(mean_traj["val_hard_loss"]),
                color="black",
                linestyle="--",
                linewidth=2.0,
                alpha=0.55,
            )
            loss_ax.set_ylim(loss_lo, loss_hi)
            loss_ax.set_ylabel("Hard val loss", color="black")
            loss_ax.tick_params(axis="y", labelcolor="black")

            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.25)
            ax.set_ylabel("Weight")
            ax.set_title(
                f"#{rank}  {run_id}  |  mean best hard val = {float(row['objective']):.2f}",
                loc="left",
                fontsize=10,
            )
            if rank == 1:
                ax.legend(loc="upper right", ncol=2, fontsize=8, frameon=False)

        axes[-1].set_xlabel("Epoch")
        fig.suptitle(title + f" (one subplot per run, sorted best to worst; page {page_idx})")
        fig.tight_layout()
        fig.subplots_adjust(top=0.96)
        page_path = out_path.with_name(f"{out_path.stem}_p{page_idx:02d}{out_path.suffix}")
        fig.savefig(page_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(page_path)

    return out_paths


def _smooth_series(values: pd.Series, window: int = 7) -> pd.Series:
    if len(values) <= 2:
        return values
    return values.rolling(window=window, min_periods=1, center=True).mean()


def _seed_results_path(kind: str, row: pd.Series, *, optuna_benchmark_id: str | None = None) -> Path:
    if kind == "llm":
        return POLICY_ROOT / str(row["benchmark_id"]) / "per_seed_results.jsonl"
    if optuna_benchmark_id is None:
        raise ValueError("optuna_benchmark_id is required for optuna rows.")
    return OPTUNA_ROOT / optuna_benchmark_id / "trials" / str(row["candidate_id"]) / "per_seed_results.jsonl"


def _collect_mean_trajectories(
    rows: pd.DataFrame,
    *,
    kind: str,
    optuna_benchmark_id: str | None = None,
) -> list[tuple[pd.Series, pd.DataFrame]]:
    out: list[tuple[pd.Series, pd.DataFrame]] = []
    for _, row in rows.iterrows():
        seed_path = _seed_results_path(kind, row, optuna_benchmark_id=optuna_benchmark_id)
        if not seed_path.exists():
            continue
        traj = _mean_trajectory_over_seeds(_load_seed_results(seed_path))
        if not traj.empty:
            out.append((row, traj))
    return out


def _aggregate_seed_trajectories(seed_results: pd.DataFrame) -> pd.DataFrame:
    if seed_results.empty or "trajectory_path" not in seed_results.columns:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    max_epoch = -1
    for _, row in seed_results.iterrows():
        traj_path = row.get("trajectory_path")
        if not traj_path:
            continue
        df = pl.read_parquet(traj_path).to_pandas().sort_values("epoch")
        if df.empty:
            continue
        df["seed"] = int(row["seed"]) if "seed" in row and pd.notna(row["seed"]) else -1
        df["running_best_hard_val_loss"] = df["val_hard_loss"].cummin()
        max_epoch = max(max_epoch, int(df["epoch"].max()))
        parts.append(df)
    if not parts:
        return pd.DataFrame()

    padded_parts: list[pd.DataFrame] = []
    for df in parts:
        seed = int(df["seed"].iloc[0])
        df_idx = df.set_index("epoch").sort_index()
        full_index = pd.RangeIndex(0, max_epoch + 1)
        df_idx = df_idx.reindex(full_index).ffill()
        df_idx.index.name = "epoch"
        df_idx["seed"] = seed
        padded_parts.append(df_idx.reset_index())

    all_df = pd.concat(padded_parts, ignore_index=True)
    value_cols = [col for col in all_df.columns if col not in {"seed"}]
    return all_df.groupby("epoch", as_index=False)[value_cols].mean().sort_values("epoch")


def _loss_axis_limits(trajectories: list[pd.DataFrame]) -> tuple[float, float]:
    if not trajectories:
        return (0.0, 1.0)
    vals = pd.concat([traj[["val_hard_loss"]] for traj in trajectories], ignore_index=True)["val_hard_loss"]
    lo = float(vals.min())
    hi = float(vals.max())
    pad = 0.05 * (hi - lo) if hi > lo else max(1.0, 0.05 * max(abs(lo), 1.0))
    return lo - pad, hi + pad


def _plot_top_runs_pages(
    llm_df: pd.DataFrame,
    optuna_df: pd.DataFrame,
    *,
    out_path: Path,
    optuna_benchmark_id: str,
    n_weights: int = 4,
    top_k: int = 10,
) -> list[Path]:
    llm_ok = llm_df[llm_df["success"]].copy().sort_values("objective", ascending=True).head(top_k)
    optuna_ok = (
        optuna_df[optuna_df["success"]].copy().sort_values("objective", ascending=True).head(top_k)
    )
    llm_items: list[tuple[pd.Series, pd.DataFrame]] = []
    for _, row in llm_ok.iterrows():
        seed_path = _seed_results_path("llm", row)
        if not seed_path.exists():
            continue
        traj = _aggregate_seed_trajectories(_load_seed_results(seed_path))
        if not traj.empty:
            llm_items.append((row, traj))

    optuna_items: list[tuple[pd.Series, pd.DataFrame]] = []
    for _, row in optuna_ok.iterrows():
        seed_path = _seed_results_path("optuna", row, optuna_benchmark_id=optuna_benchmark_id)
        if not seed_path.exists():
            continue
        traj = _aggregate_seed_trajectories(_load_seed_results(seed_path))
        if not traj.empty:
            optuna_items.append((row, traj))

    pages: list[tuple[str, list[tuple[pd.Series, pd.DataFrame]]]] = []
    if llm_items:
        pages.append(("Top 10 LLM schedules", llm_items[:top_k]))
    if optuna_items:
        pages.append(("Top 10 Optuna schedules", optuna_items[:top_k]))
    if not pages:
        return []

    all_trajs = [traj for _, items in pages for _, traj in items]
    loss_lo, loss_hi = _loss_axis_limits(all_trajs)
    weight_cols = [f"weight_{i}" for i in range(n_weights)]
    weight_labels = [
        f"sigma {i} ({'easiest' if i == 0 else 'hardest' if i == n_weights - 1 else 'middle'})"
        for i in range(n_weights)
    ]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    out_paths: list[Path] = []

    def _draw_panel(ax: Any, row: pd.Series, traj: pd.DataFrame, title_text: str, show_legend: bool) -> None:
        for idx, (col, label) in enumerate(zip(weight_cols, weight_labels)):
            ax.plot(traj["epoch"], traj[col], color=colors[idx % len(colors)], linewidth=1.8, label=label)
        loss_ax = ax.twinx()
        loss_ax.plot(
            traj["epoch"],
            _smooth_series(traj["running_best_hard_val_loss"]),
            color="black",
            linestyle="--",
            linewidth=2.0,
            alpha=0.55,
        )
        loss_ax.set_ylim(loss_lo, loss_hi)
        loss_ax.set_ylabel("Hard val loss", color="black")
        loss_ax.tick_params(axis="y", labelcolor="black")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("Weight")
        ax.set_title(title_text, loc="left", fontsize=10, pad=12)
        if show_legend:
            ax.legend(loc="upper right", ncol=2, fontsize=8, frameon=False)

    for page_idx, (page_title, items) in enumerate(pages, start=1):
        fig, axes = plt.subplots(5, 2, figsize=(12.0, 17.0), sharex=True)
        axes_flat = axes.flatten()
        for ax_idx, ax in enumerate(axes_flat):
            if ax_idx >= len(items):
                ax.axis("off")
                continue
            row, traj = items[ax_idx]
            prefix = "LLM" if "LLM" in page_title else "Optuna"
            _draw_panel(
                ax,
                row,
                traj,
                f"{prefix} #{ax_idx + 1}  {str(row['candidate_id'])}  |  mean best hard val = {float(row['objective']):.2f}",
                show_legend=(ax_idx == 0),
            )
        for ax in axes[-1, :]:
            ax.set_xlabel("Epoch")
        fig.suptitle(page_title, y=0.992, fontsize=15)
        fig.subplots_adjust(top=0.965, bottom=0.05, left=0.07, right=0.95, hspace=0.6, wspace=0.4)
        page_path = out_path.with_name(f"{out_path.stem}_p{page_idx:02d}{out_path.suffix}")
        fig.savefig(page_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(page_path)

    return out_paths


def _plot_best_trajectories(
    llm_traj: pd.DataFrame,
    optuna_traj: pd.DataFrame,
    out_path: Path,
    hard_weight_col: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)
    colors = {"LLM": "tab:blue", "Optuna": "tab:orange"}

    for label, df in [("LLM", llm_traj), ("Optuna", optuna_traj)]:
        if df.empty:
            continue
        color = colors[label]
        for _, seed_df in df.groupby("seed"):
            axes[0].plot(seed_df["epoch"], seed_df["val_hard_loss"], color=color, alpha=0.20)
            axes[1].plot(seed_df["epoch"], seed_df[hard_weight_col], color=color, alpha=0.20)

        hard_curve = _aggregate_trajectory(df, "val_hard_loss")
        weight_curve = _aggregate_trajectory(df, hard_weight_col)
        axes[0].plot(hard_curve["epoch"], hard_curve["median"], color=color, linewidth=2, label=label)
        axes[0].fill_between(
            hard_curve["epoch"],
            hard_curve["q25"],
            hard_curve["q75"],
            color=color,
            alpha=0.12,
        )
        axes[1].plot(weight_curve["epoch"], weight_curve["median"], color=color, linewidth=2)
        axes[1].fill_between(
            weight_curve["epoch"],
            weight_curve["q25"],
            weight_curve["q75"],
            color=color,
            alpha=0.12,
        )

    axes[0].set_ylabel("Hard validation loss")
    axes[0].set_title("Best-schedule training trajectories")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Hardest-loss weight")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _to_md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        cols = [str(col) for col in df.columns]
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            values = [str(row[col]) for col in df.columns]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)


def _write_markdown_summary(
    out_path: Path,
    *,
    report_id: str,
    llm_best: pd.Series,
    optuna_best: pd.Series,
    llm_df: pd.DataFrame,
    optuna_df: pd.DataFrame,
    token_summary: dict[str, Any],
    per_seed_df: pd.DataFrame,
    headline_df: pd.DataFrame,
) -> None:
    delta = float(optuna_best["objective"]) - float(llm_best["objective"])
    lines: list[str] = []
    lines.append(f"# Schedule search comparison: {report_id}")
    lines.append("")
    lines.append(
        "Matched comparison of LLM-generated schedule policies against Optuna schedule search "
        "on the Eggholder 4-loss benchmark. Lower is better."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- Best completed **LLM** candidate: `{llm_best['candidate_id']}` "
        f"with mean best hard validation loss **{_format_float(float(llm_best['objective']))}**."
    )
    lines.append(
        f"- Best **Optuna** trial: `{optuna_best['candidate_id']}` "
        f"with mean best hard validation loss **{_format_float(float(optuna_best['objective']))}**."
    )
    lines.append(
        f"- Absolute gap (Optuna - LLM): **{_format_float(delta)}** "
        f"({('LLM better' if delta > 0 else 'Optuna better')})."
    )
    lines.append(
        f"- LLM run status: **{int(llm_df['success'].sum())}/{len(llm_df)}** completed successfully; "
        f"failures were partly generation errors and partly quota exhaustion."
    )
    lines.append(
        f"- Token usage for the LLM run: **{_format_int(int(token_summary['total_tokens']))}** total "
        f"({_format_int(int(token_summary['input_tokens']))} input, "
        f"{_format_int(int(token_summary['output_tokens']))} output)."
    )
    lines.append("")
    lines.append("## Headline table")
    lines.append("")
    lines.append(_to_md_table(headline_df))
    lines.append("")
    lines.append("## Per-seed comparison of the best schedules")
    lines.append("")
    lines.append(_to_md_table(per_seed_df))
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append("![Search progress](best_so_far.png)")
    lines.append("")
    lines.append("![Distribution of completed evaluations](objective_distribution.png)")
    lines.append("")
    lines.append("![Per-seed comparison](per_seed_best.png)")
    lines.append("")
    lines.append("![Best-schedule trajectories](best_trajectories.png)")
    lines.append("")
    lines.append("![LLM schedule gallery](llm_schedule_gallery.png)")
    lines.append("")
    lines.append("![Optuna schedule gallery](optuna_schedule_gallery.png)")
    lines.append("")
    lines.append("## Note")
    lines.append("")
    lines.append(
        "The current LLM result should be described as a partial 20-candidate search, because the "
        "final four candidate slots were blocked by API quota."
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tex_table(out_path: Path, headline_df: pd.DataFrame) -> None:
    rows = []
    for _, row in headline_df.iterrows():
        rows.append(
            "{} & {} & {} & {} & {} \\\\".format(
                row["method"],
                row["completed"],
                row["best_candidate"],
                row["best_mean_best_hard_val_loss"],
                row["best_std_best_hard_val_loss"],
            )
        )
    content = "\n".join(
        [
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Method & Completed & Best id & Mean best hard val & Std across seeds \\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    out_path.write_text(content + "\n", encoding="utf-8")


@app.command()
def main(
    llm_study_name: str = typer.Option(..., help="LLM study name under reports/llm_schedule_search."),
    optuna_benchmark_id: str = typer.Option(
        ..., help="Optuna benchmark id under reports/benchmarks/optuna_schedule."
    ),
    report_id: str | None = typer.Option(None, help="Optional output directory name."),
) -> None:
    resolved_report_id = report_id or f"{llm_study_name}__vs__{optuna_benchmark_id}"
    out_dir = ANALYSIS_ROOT / resolved_report_id
    out_dir.mkdir(parents=True, exist_ok=True)

    llm_df, token_summary = _load_llm_candidates(llm_study_name)
    optuna_df, optuna_aggregate = _load_optuna_trials(optuna_benchmark_id)

    llm_best = _best_success_row(llm_df)
    optuna_best = _best_success_row(optuna_df)

    llm_best_benchmark = str(llm_best["benchmark_id"])
    optuna_best_trial_id = str(optuna_best["candidate_id"])

    llm_seed_df = _load_seed_results(
        POLICY_ROOT / llm_best_benchmark / "per_seed_results.jsonl"
    )
    optuna_seed_df = _load_seed_results(
        OPTUNA_ROOT / optuna_benchmark_id / "trials" / optuna_best_trial_id / "per_seed_results.jsonl"
    )

    llm_traj = _load_trajectory_frame("LLM", llm_seed_df)
    optuna_traj = _load_trajectory_frame("Optuna", optuna_seed_df)

    llm_df.to_csv(out_dir / "llm_candidates.csv", index=False)
    optuna_df.to_csv(out_dir / "optuna_trials.csv", index=False)
    llm_seed_df.to_csv(out_dir / "best_llm_per_seed.csv", index=False)
    optuna_seed_df.to_csv(out_dir / "best_optuna_per_seed.csv", index=False)

    failure_df = (
        llm_df.loc[~llm_df["success"]]
        .groupby("failure_type")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    failure_df.to_csv(out_dir / "llm_failures.csv", index=False)

    headline_df = pd.DataFrame(
        [
            {
                "method": "LLM",
                "attempted": int(len(llm_df)),
                "completed": int(llm_df["success"].sum()),
                "best_candidate": str(llm_best["candidate_id"]),
                "best_mean_best_hard_val_loss": _format_float(float(llm_best["objective"])),
                "best_std_best_hard_val_loss": _format_float(
                    _safe_float(llm_best.get("std_best_hard_val_loss"))
                ),
            },
            {
                "method": "Optuna",
                "attempted": int(len(optuna_df)),
                "completed": int(optuna_df["success"].sum()),
                "best_candidate": str(optuna_best["candidate_id"]),
                "best_mean_best_hard_val_loss": _format_float(float(optuna_best["objective"])),
                "best_std_best_hard_val_loss": _format_float(
                    _safe_float(optuna_best.get("std_best_hard_val_loss"))
                ),
            },
        ]
    )
    headline_df.to_csv(out_dir / "headline_table.csv", index=False)

    _plot_best_so_far(llm_df, optuna_df, out_dir / "best_so_far.png")
    _plot_objective_distribution(llm_df, optuna_df, out_dir / "objective_distribution.png")
    per_seed_df = _plot_per_seed_comparison(
        llm_seed_df,
        optuna_seed_df,
        out_dir / "per_seed_best.png",
    )
    per_seed_df.to_csv(out_dir / "per_seed_comparison.csv", index=False)
    _plot_best_trajectories(llm_traj, optuna_traj, out_dir / "best_trajectories.png", "weight_3")
    llm_gallery_paths = _plot_schedule_gallery(
        llm_df,
        kind="llm",
        out_path=out_dir / "llm_schedule_gallery.png",
    )
    optuna_gallery_paths = _plot_schedule_gallery(
        optuna_df,
        kind="optuna",
        out_path=out_dir / "optuna_schedule_gallery.png",
        benchmark_id=optuna_benchmark_id,
    )
    top_runs_paths = _plot_top_runs_pages(
        llm_df,
        optuna_df,
        out_path=out_dir / "top10_runs.png",
        optuna_benchmark_id=optuna_benchmark_id,
    )

    summary = {
        "report_id": resolved_report_id,
        "llm_study_name": llm_study_name,
        "optuna_benchmark_id": optuna_benchmark_id,
        "llm_attempted": int(len(llm_df)),
        "llm_completed": int(llm_df["success"].sum()),
        "optuna_attempted": int(len(optuna_df)),
        "optuna_completed": int(optuna_df["success"].sum()),
        "best_llm_candidate_id": str(llm_best["candidate_id"]),
        "best_llm_benchmark_id": llm_best_benchmark,
        "best_llm_mean_best_hard_val_loss": float(llm_best["objective"]),
        "best_optuna_trial_id": optuna_best_trial_id,
        "best_optuna_mean_best_hard_val_loss": float(optuna_best["objective"]),
        "optuna_minus_llm": float(optuna_best["objective"]) - float(llm_best["objective"]),
        "llm_total_tokens": int(token_summary["total_tokens"]),
        "llm_input_tokens": int(token_summary["input_tokens"]),
        "llm_output_tokens": int(token_summary["output_tokens"]),
        "optuna_best_trial_number": int(optuna_aggregate["best_trial_number"]),
        "llm_gallery_pages": [str(path.name) for path in llm_gallery_paths],
        "optuna_gallery_pages": [str(path.name) for path in optuna_gallery_paths],
        "top_runs_pages": [str(path.name) for path in top_runs_paths],
    }
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    _write_markdown_summary(
        out_dir / "summary.md",
        report_id=resolved_report_id,
        llm_best=llm_best,
        optuna_best=optuna_best,
        llm_df=llm_df,
        optuna_df=optuna_df,
        token_summary=token_summary,
        per_seed_df=per_seed_df,
        headline_df=headline_df,
    )
    _write_tex_table(out_dir / "headline_table.tex", headline_df)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
