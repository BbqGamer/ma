from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LOG_RE = re.compile(
    r"Epoch \[(?P<epoch>\d+)/(?P<epochs>\d+)\] \| "
    r"Train Loss: (?P<train_loss>[0-9.]+) \| "
    r"Val Fine-CE: (?P<val_fine_ce>[0-9.]+) \| "
    r"Fine Hit@1: (?P<fine_hit>[0-9.]+)% \| "
    r"Coarse Hit@1: (?P<coarse_hit>[0-9.]+)% \| "
    r"HierDist: (?P<hier_dist>[0-9.]+) \| "
    r"LR: (?P<lr>[0-9.]+)"
)
RUN_RE = re.compile(r"Run ID: (?P<run_id>.+)$")
SELECT_RE = re.compile(r"select_frac=(?P<select_frac>[0-9.]+)")
SELECTED_RE = re.compile(r"Selected (?P<selected>\d+)/(?P<total>\d+) fine classes")


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
OUT_DIR = ROOT / "reports" / "run_analysis"
PLOTS_DIR = OUT_DIR / "plots"


def parse_log(path: Path) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    meta: dict[str, object] = {"file": str(path.relative_to(ROOT))}
    selected_counts: list[int] = []

    for line in path.read_text().splitlines():
        if "Initialized on" in line:
            match = RUN_RE.search(line)
            if match:
                meta["run_id"] = match.group("run_id")
            if "Mode:" in line:
                meta["mode"] = line.split("Mode:", 1)[1].split("|", 1)[0].strip().lower()
        if "select_frac=" in line:
            match = SELECT_RE.search(line)
            if match:
                meta["select_frac"] = float(match.group("select_frac"))
        if "Selected " in line:
            match = SELECTED_RE.search(line)
            if match:
                selected_counts.append(int(match.group("selected")))
                meta["selected_total"] = int(match.group("total"))
        match = LOG_RE.search(line)
        if match:
            row = {k: float(v) for k, v in match.groupdict().items() if k not in {"epoch", "epochs"}}
            row["epoch"] = int(match.group("epoch"))
            row["epochs"] = int(match.group("epochs"))
            rows.append(row)

    if not rows:
        raise ValueError(f"No epoch rows found in {path}")

    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    df["fine_hit"] /= 100.0
    df["coarse_hit"] /= 100.0
    df["coarse_from_fine_acc"] = 2.0 - df["fine_hit"] - df["hier_dist"]
    df["sibling_confusion_rate"] = df["coarse_from_fine_acc"] - df["fine_hit"]
    df["coarse_wrong_rate"] = 1.0 - df["coarse_from_fine_acc"]

    meta["num_epochs"] = int(df["epochs"].iloc[-1])
    if selected_counts:
        meta["selected_min"] = min(selected_counts)
        meta["selected_max"] = max(selected_counts)
        meta["selected_mean"] = sum(selected_counts) / len(selected_counts)

    return df, meta


def make_summary(df: pd.DataFrame, meta: dict) -> dict:
    idx_best_fine = df["fine_hit"].idxmax()
    idx_best_hier = df["hier_dist"].idxmin()
    idx_best_derived_coarse = df["coarse_from_fine_acc"].idxmax()
    idx_best_val_ce = df["val_fine_ce"].idxmin()
    idx_best_coarse_head = df["coarse_hit"].idxmax()
    final = df.iloc[-1]

    return {
        "mode": meta.get("mode"),
        "run_id": meta.get("run_id"),
        "select_frac": meta.get("select_frac"),
        "epochs": meta.get("num_epochs"),
        "best_fine_hit_epoch": int(df.loc[idx_best_fine, "epoch"]),
        "best_fine_hit": float(df.loc[idx_best_fine, "fine_hit"]),
        "best_hierdist_epoch": int(df.loc[idx_best_hier, "epoch"]),
        "best_hierdist": float(df.loc[idx_best_hier, "hier_dist"]),
        "best_derived_coarse_epoch": int(df.loc[idx_best_derived_coarse, "epoch"]),
        "best_derived_coarse": float(df.loc[idx_best_derived_coarse, "coarse_from_fine_acc"]),
        "best_val_fine_ce_epoch": int(df.loc[idx_best_val_ce, "epoch"]),
        "best_val_fine_ce": float(df.loc[idx_best_val_ce, "val_fine_ce"]),
        "best_coarse_head_epoch": int(df.loc[idx_best_coarse_head, "epoch"]),
        "best_coarse_head": float(df.loc[idx_best_coarse_head, "coarse_hit"]),
        "final_fine_hit": float(final["fine_hit"]),
        "final_hierdist": float(final["hier_dist"]),
        "final_derived_coarse": float(final["coarse_from_fine_acc"]),
        "final_coarse_head": float(final["coarse_hit"]),
        "final_val_fine_ce": float(final["val_fine_ce"]),
        "final_sibling_confusion": float(final["sibling_confusion_rate"]),
        "final_coarse_wrong_rate": float(final["coarse_wrong_rate"]),
    }


def plot_main(curves: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = [
        ("fine_hit", "Fine Hit@1", True),
        ("hier_dist", "HierDist", False),
        ("val_fine_ce", "Validation fine CE", False),
        ("coarse_from_fine_acc", "Derived coarse accuracy from fine prediction", True),
    ]

    colors = {"baseline": "tab:blue", "hcl": "tab:orange", "hier": "tab:green"}

    for ax, (metric, title, is_pct) in zip(axes.flat, metrics):
        for mode, df in curves.items():
            y = df[metric] * 100 if is_pct else df[metric]
            ax.plot(df["epoch"], y, label=mode.upper(), linewidth=2, color=colors.get(mode))
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        if is_pct:
            ax.set_ylabel("Percent")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("CIFAR-100 baseline vs HCL from runs/", fontsize=14)
    fig.savefig(PLOTS_DIR / "main_metrics.png", dpi=180)
    plt.close(fig)


def plot_coarse(curves: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    colors = {"baseline": "tab:blue", "hcl": "tab:orange", "hier": "tab:green"}

    for mode, df in curves.items():
        axes[0].plot(df["epoch"], df["coarse_hit"] * 100, label=mode.upper(), linewidth=2, color=colors.get(mode))
        axes[1].plot(df["epoch"], df["sibling_confusion_rate"] * 100, label=mode.upper(), linewidth=2, color=colors.get(mode))

    axes[0].set_title("Logged coarse-head Hit@1")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Percent")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Sibling-confusion rate")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Percent")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Coarse behavior diagnostics", fontsize=14)
    fig.savefig(PLOTS_DIR / "coarse_diagnostics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    log_paths = sorted(RUNS_DIR.glob("training_log_*.txt"))
    curves: dict[str, pd.DataFrame] = {}
    summaries: list[dict] = []

    for path in log_paths:
        df, meta = parse_log(path)
        mode = str(meta.get("mode") or path.stem.replace("training_log_", "")).lower()
        curves[mode] = df
        df.to_csv(OUT_DIR / f"{mode}_epochs.csv", index=False)
        summaries.append(make_summary(df, meta))

    pd.DataFrame(summaries).sort_values("mode").to_csv(OUT_DIR / "summary.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summaries, indent=2))

    if curves:
        plot_main(curves)
        plot_coarse(curves)


if __name__ == "__main__":
    main()
