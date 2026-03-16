#!/usr/bin/env python3
"""Analyze curriculum vs single benchmark runs.

Reads reports/benchmarks/<benchmark_id>_runs.csv and joins with mlflow.db runs/metrics.
Writes aggregate CSV tables and a markdown summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "reports" / "benchmarks"
MLFLOW_DB = ROOT / "mlflow.db"

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class AggregateRow:
    function: str
    regime: str
    method: str
    n: int
    median_quality: float
    iqr_quality: float
    median_runtime_sec: float
    iqr_runtime_sec: float


def _iqr(series: pd.Series) -> float:
    if series.empty:
        return math.nan
    return float(series.quantile(0.75) - series.quantile(0.25))


def _fetch_mlflow_runs(run_names: list[str]) -> pd.DataFrame:
    if not run_names:
        return pd.DataFrame()

    conn = sqlite3.connect(MLFLOW_DB)
    chunks: list[pd.DataFrame] = []

    chunk_size = 500
    for i in range(0, len(run_names), chunk_size):
        batch = run_names[i : i + chunk_size]
        placeholders = ",".join(["?"] * len(batch))
        query = f"""
            select
                r.run_uuid,
                r.name as run_name,
                r.status as mlflow_status,
                r.start_time,
                r.end_time,
                max(case when lm.key='final_hard_val_loss' then lm.value end) as final_hard_val_loss,
                max(case when lm.key='best_val_loss' then lm.value end) as best_val_loss,
                max(case when lm.key='total_epochs' then lm.value end) as total_epochs
            from runs r
            left join latest_metrics lm on lm.run_uuid = r.run_uuid
            where r.name in ({placeholders})
            group by r.run_uuid, r.name, r.status, r.start_time, r.end_time
        """
        chunk_df = pd.read_sql_query(query, conn, params=batch)
        chunks.append(chunk_df)

    conn.close()
    if not chunks:
        return pd.DataFrame()

    all_rows = pd.concat(chunks, ignore_index=True)
    if all_rows.empty:
        return all_rows

    # Keep the latest run per run_name in case of retries
    all_rows = all_rows.sort_values(["run_name", "end_time"]).drop_duplicates(
        ["run_name"], keep="last"
    )
    return all_rows


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[AggregateRow] = []
    for (function, regime, method), g in df.groupby(["function", "regime", "method"]):
        q = g["final_hard_val_loss"].dropna()
        t = g["runtime_sec"].dropna()
        rows.append(
            AggregateRow(
                function=function,
                regime=regime,
                method=method,
                n=int(len(g)),
                median_quality=float(q.median()) if not q.empty else math.nan,
                iqr_quality=_iqr(q),
                median_runtime_sec=float(t.median()) if not t.empty else math.nan,
                iqr_runtime_sec=_iqr(t),
            )
        )
    out = pd.DataFrame([r.__dict__ for r in rows])
    if out.empty:
        return pd.DataFrame(
            columns=[
                "function",
                "regime",
                "method",
                "n",
                "median_quality",
                "iqr_quality",
                "median_runtime_sec",
                "iqr_runtime_sec",
            ]
        )
    return out.sort_values(["regime", "function", "method"])


def _pairwise_delta(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return pd.DataFrame(
            columns=[
                "function",
                "regime",
                "median_quality_single",
                "median_quality_curriculum",
                "median_runtime_sec_single",
                "median_runtime_sec_curriculum",
                "delta_quality_single_minus_curriculum",
                "delta_runtime_sec_single_minus_curriculum",
            ]
        )

    wide = agg.pivot_table(
        index=["function", "regime"],
        columns="method",
        values=["median_quality", "median_runtime_sec", "n"],
        aggfunc="first",
    )

    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    out = wide.reset_index()

    if "median_quality_single" in out.columns and "median_quality_curriculum" in out.columns:
        out["delta_quality_single_minus_curriculum"] = (
            out["median_quality_single"] - out["median_quality_curriculum"]
        )
    else:
        out["delta_quality_single_minus_curriculum"] = math.nan

    if "median_runtime_sec_single" in out.columns and "median_runtime_sec_curriculum" in out.columns:
        out["delta_runtime_sec_single_minus_curriculum"] = (
            out["median_runtime_sec_single"] - out["median_runtime_sec_curriculum"]
        )
    else:
        out["delta_runtime_sec_single_minus_curriculum"] = math.nan

    return out.sort_values(["regime", "function"])


def _time_to_target(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "function",
                "regime",
                "method",
                "target_single_median_quality",
                "hit_rate",
                "median_runtime_sec_for_hits",
                "n",
            ]
        )

    rows: list[dict[str, Any]] = []
    for (function, regime), g in df.groupby(["function", "regime"]):
        single_q = g.loc[g["method"] == "single", "final_hard_val_loss"].dropna()
        if single_q.empty:
            target = math.nan
        else:
            target = float(single_q.median())

        for method in sorted(g["method"].unique()):
            gm = g[g["method"] == method].copy()
            if math.isnan(target):
                hit_rate = math.nan
                median_time_hit = math.nan
            else:
                hits = gm["final_hard_val_loss"] <= target
                hit_rate = float(hits.mean()) if len(gm) else math.nan
                times_hit = gm.loc[hits, "runtime_sec"].dropna()
                median_time_hit = float(times_hit.median()) if not times_hit.empty else math.nan

            rows.append(
                {
                    "function": function,
                    "regime": regime,
                    "method": method,
                    "target_single_median_quality": target,
                    "hit_rate": hit_rate,
                    "median_runtime_sec_for_hits": median_time_hit,
                    "n": int(len(gm)),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "function",
                "regime",
                "method",
                "target_single_median_quality",
                "hit_rate",
                "median_runtime_sec_for_hits",
                "n",
            ]
        )
    return out.sort_values(["regime", "function", "method"])


def _to_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    try:
        return df.to_markdown(index=False)
    except ImportError:
        # Optional dependency (tabulate) may be missing in minimal environments.
        return df.to_string(index=False)


@app.command()
def main(
    benchmark_id: str = typer.Option(..., help="Benchmark id (prefix in reports/benchmarks)."),
) -> None:
    runs_path = BENCHMARK_DIR / f"{benchmark_id}_runs.csv"
    if not runs_path.exists():
        raise typer.BadParameter(f"Missing run table: {runs_path}")
    if not MLFLOW_DB.exists():
        raise typer.BadParameter(f"Missing MLflow DB: {MLFLOW_DB}")

    out_dir = BENCHMARK_DIR / benchmark_id
    out_dir.mkdir(parents=True, exist_ok=True)

    bench = pd.read_csv(runs_path)
    ml_runs = _fetch_mlflow_runs(bench["run_name"].astype(str).tolist())

    merged = bench.merge(ml_runs, on="run_name", how="left", suffixes=("", "_ml"))

    # Runtime from runner wall-clock; fallback to MLflow start/end when missing
    merged["runtime_sec"] = pd.to_numeric(merged["elapsed_sec"], errors="coerce")
    ml_runtime = (pd.to_numeric(merged["end_time"], errors="coerce") - pd.to_numeric(merged["start_time"], errors="coerce")) / 1000.0
    merged.loc[merged["runtime_sec"].isna(), "runtime_sec"] = ml_runtime

    for col in ["final_hard_val_loss", "best_val_loss", "total_epochs"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    success = merged[
        merged["status"].isin(["OK", "FAILED", "TIMEOUT"])
    ].copy()

    agg = _aggregate(success)
    delta = _pairwise_delta(agg)
    ttt = _time_to_target(success)

    merged_path = out_dir / "runs_joined.csv"
    agg_path = out_dir / "aggregate.csv"
    delta_path = out_dir / "delta.csv"
    ttt_path = out_dir / "time_to_target.csv"
    md_path = out_dir / "summary.md"

    merged.to_csv(merged_path, index=False)
    agg.to_csv(agg_path, index=False)
    delta.to_csv(delta_path, index=False)
    ttt.to_csv(ttt_path, index=False)

    lines = []
    lines.append(f"# Benchmark Summary: {benchmark_id}")
    lines.append("")
    lines.append(f"Generated at `{datetime.now().astimezone().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(_to_md_table(agg))
    lines.append("")
    lines.append("## Single Minus Curriculum Deltas")
    lines.append("")
    lines.append(_to_md_table(delta))
    lines.append("")
    lines.append("## Time-to-Target (Target = single median quality)")
    lines.append("")
    lines.append(_to_md_table(ttt))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Joined runs: {merged_path}")
    print(f"Aggregate: {agg_path}")
    print(f"Delta: {delta_path}")
    print(f"Time-to-target: {ttt_path}")
    print(f"Summary: {md_path}")


if __name__ == "__main__":
    app()
