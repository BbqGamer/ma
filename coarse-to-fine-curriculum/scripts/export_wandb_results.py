#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export W&B runs to CSV files for offline analysis")
    parser.add_argument("--entity", default=None, help="W&B entity/team. Optional for default entity.")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--group", default=None, help="Optional W&B group filter")
    parser.add_argument("--name-regex", default=None, help="Optional regex filter on run name")
    parser.add_argument("--tag", action="append", default=[], help="Optional tag filter; can be repeated")
    parser.add_argument("--output-dir", type=Path, default=Path("wandb_export"))
    parser.add_argument("--history-samples", type=int, default=100000)
    return parser.parse_args()


def flatten(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in payload.items():
        if key.startswith("_"):
            continue
        out_key = f"{prefix}{key}"
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[out_key] = value
        else:
            try:
                row[out_key] = json.dumps(value)
            except TypeError:
                row[out_key] = str(value)
    return row


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:180]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    histories_dir = args.output_dir / "histories"
    histories_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    path = f"{args.entity}/{args.project}" if args.entity else args.project
    filters: dict[str, Any] = {}
    if args.group:
        filters["group"] = args.group
    if args.tag:
        filters["tags"] = {"$all": args.tag}

    name_re = re.compile(args.name_regex) if args.name_regex else None
    runs = list(api.runs(path, filters=filters))
    if name_re is not None:
        runs = [run for run in runs if name_re.search(run.name or "")]

    summary_rows = []
    history_frames = []
    for run in runs:
        row = {
            "wandb_id": run.id,
            "name": run.name,
            "group": run.group,
            "state": run.state,
            "url": run.url,
            "created_at": run.created_at,
            "tags": ",".join(run.tags or []),
        }
        row.update(flatten("config_", dict(run.config)))
        row.update(flatten("summary_", dict(run.summary)))
        summary_rows.append(row)

        history = run.history(samples=args.history_samples, pandas=True)
        history.insert(0, "wandb_id", run.id)
        history.insert(1, "name", run.name)
        history.insert(2, "group", run.group)
        history_path = histories_dir / f"{safe_name(run.name or run.id)}_{run.id}.csv"
        history.to_csv(history_path, index=False)
        history_frames.append(history)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "wandb_runs_summary.csv", index=False)
    if history_frames:
        pd.concat(history_frames, ignore_index=True, sort=False).to_csv(
            args.output_dir / "wandb_history.csv",
            index=False,
        )
    print(f"Exported {len(runs)} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
