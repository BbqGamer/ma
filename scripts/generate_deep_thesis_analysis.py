from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "thesis_deep_dive"
FIG = ROOT / "thesis" / "figures"
APP = ROOT / "thesis" / "appendices" / "deep_dive_experiments.tex"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

plt.style.use("ggplot")


def esc(s: object) -> str:
    return str(s).replace("_", "\\_")


def pct(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{100*x:.2f}"


def pp(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{100*x:+.2f}"


def savefig(name: str) -> str:
    path = FIG / name
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return f"figures/{name}"


# ----------------------------- Regression benchmark -----------------------------
bench_dir = ROOT / "reports" / "benchmarks" / "curr_vs_single_20260315_175826"
runs = pd.read_csv(bench_dir / "runs_joined.csv")
delta = pd.read_csv(bench_dir / "delta.csv")

delta["runtime_ratio_curr_over_single"] = (
    delta["median_runtime_sec_curriculum"] / delta["median_runtime_sec_single"]
)
delta["quality_winner"] = np.where(
    delta["delta_quality_single_minus_curriculum"] > 0, "curriculum", "single"
)
delta.to_csv(OUT / "regression_curr_vs_single_deltas_enriched.csv", index=False)

plt.figure(figsize=(8.4, 5.2))
for regime, sub in delta.groupby("regime"):
    plt.scatter(
        sub["runtime_ratio_curr_over_single"],
        sub["delta_quality_single_minus_curriculum"],
        s=80,
        label=regime,
        alpha=0.85,
    )
    for _, r in sub.iterrows():
        plt.annotate(r["function"], (r["runtime_ratio_curr_over_single"], r["delta_quality_single_minus_curriculum"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
plt.axhline(0, color="black", linewidth=1)
plt.axvline(1, color="black", linewidth=1)
plt.xlabel("Median runtime ratio: curriculum / single")
plt.ylabel("Median quality delta: single loss - curriculum loss")
plt.title("Regression continuation: quality gain versus runtime cost")
plt.legend()
reg_tradeoff_fig = savefig("deep_regression_quality_runtime_tradeoff.png")

# Paired seed deltas: single - curriculum per function/regime/seed
paired = runs.pivot_table(
    index=["function", "regime", "seed"],
    columns="method",
    values="final_hard_val_loss",
    aggfunc="first",
).reset_index()
paired["delta_single_minus_curriculum"] = paired["single"] - paired["curriculum"]
paired.to_csv(OUT / "regression_seed_paired_deltas.csv", index=False)
heat = paired.pivot_table(index="function", columns="regime", values="delta_single_minus_curriculum", aggfunc="median")
plt.figure(figsize=(6.8, 5.0))
im = plt.imshow(heat.values, cmap="coolwarm", aspect="auto", vmin=-np.nanmax(abs(heat.values)), vmax=np.nanmax(abs(heat.values)))
plt.colorbar(im, label="Median seed-paired loss delta (single - curriculum)")
plt.xticks(range(len(heat.columns)), [esc(c) for c in heat.columns], rotation=20)
plt.yticks(range(len(heat.index)), [esc(i) for i in heat.index])
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        val = heat.iloc[i, j]
        plt.text(j, i, f"{val:.2g}", ha="center", va="center", fontsize=8)
plt.title("Regression paired deltas by function and budget regime")
reg_heat_fig = savefig("deep_regression_seed_delta_heatmap.png")

# ----------------------------- MLflow v3 sweeps -----------------------------
conn = sqlite3.connect(ROOT / "mlflow.db")
query = """
select e.name as experiment_name, r.run_uuid, r.name as run_name, r.status,
       max(case when p.key='function' then p.value end) as function,
       max(case when p.key='model_arch' then p.value end) as model_arch,
       cast(max(case when p.key='n_params' then p.value end) as real) as n_params,
       max(case when m.key='best_val_loss' then m.value end) as best_val_loss,
       max(case when m.key='final_test_loss' then m.value end) as final_test_loss,
       max(case when m.key='hessian_trace' then m.value end) as hessian_trace,
       max(case when m.key='critical_sharpness' then m.value end) as critical_sharpness,
       max(case when m.key='grad_cosine_sim' then m.value end) as grad_cosine_sim,
       max(case when m.key='grad_noise_scale' then m.value end) as grad_noise_scale,
       max(case when m.key='train_per_param' then m.value end) as train_per_param
from runs r
join experiments e on r.experiment_id=e.experiment_id
left join params p on p.run_uuid=r.run_uuid
left join latest_metrics m on m.run_uuid=r.run_uuid
where e.name like 'sweep-%-v3-ratio' and r.status='FINISHED'
group by e.name, r.run_uuid
"""
v3 = pd.read_sql_query(query, conn)
v3 = v3.dropna(subset=["function", "model_arch", "best_val_loss"])
v3.to_csv(OUT / "mlflow_v3_sweeps_export.csv", index=False)

best_v3 = v3.sort_values("best_val_loss").groupby("function", as_index=False).first()
best_v3.to_csv(OUT / "mlflow_v3_best_by_function.csv", index=False)
plt.figure(figsize=(8.4, 4.8))
order = best_v3.sort_values("final_test_loss" if best_v3["final_test_loss"].notna().any() else "best_val_loss")
vals = order["final_test_loss"].fillna(order["best_val_loss"])
colors = {"mlp": "#4c78a8", "siren": "#f58518", "fourier": "#54a24b"}
plt.bar(order["function"], vals, color=[colors.get(a, "gray") for a in order["model_arch"]])
plt.yscale("log")
plt.xticks(rotation=35, ha="right")
plt.ylabel("Best available loss (log scale)")
plt.title("MLflow v3 ratio sweeps: best run by benchmark function")
for i, (_, r) in enumerate(order.iterrows()):
    plt.text(i, vals.iloc[i], r["model_arch"], ha="center", va="bottom", fontsize=8, rotation=90)
mlflow_best_fig = savefig("deep_mlflow_v3_best_losses.png")

# Top-k architecture distribution by function
v3_ranked = v3.sort_values("best_val_loss").groupby("function").head(20)
arch_counts = pd.crosstab(v3_ranked["function"], v3_ranked["model_arch"])
arch_counts.to_csv(OUT / "mlflow_v3_top20_arch_counts.csv")
arch_counts.plot(kind="bar", stacked=True, figsize=(8.4, 5.0), color=[colors.get(c, "gray") for c in arch_counts.columns])
plt.ylabel("Count among top 20 trials")
plt.title("MLflow v3 sweeps: architecture mix among top trials")
plt.xticks(rotation=35, ha="right")
mlflow_arch_fig = savefig("deep_mlflow_v3_top20_arch_distribution.png")

# ----------------------------- Policy schedule exploratory reports -----------------------------
policy_rows = []
for p in (ROOT / "reports" / "benchmarks" / "policy").glob("*/aggregate.json"):
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    policy_rows.append({
        "benchmark_id": d.get("benchmark_id"),
        "candidate_id": d.get("candidate_id"),
        "function": d.get("function"),
        "num_losses": d.get("num_losses"),
        "mean_best_hard_val_loss": d.get("mean_best_hard_val_loss"),
        "std_best_hard_val_loss": d.get("std_best_hard_val_loss"),
        "mean_final_hard_val_loss": d.get("mean_final_hard_val_loss"),
        "mean_epochs_trained": d.get("mean_epochs_trained"),
        "status": d.get("status"),
    })
policy = pd.DataFrame(policy_rows)
if not policy.empty:
    policy = policy.dropna(subset=["mean_best_hard_val_loss"]).sort_values("mean_best_hard_val_loss")
    policy.to_csv(OUT / "policy_schedule_aggregates_ranked.csv", index=False)

optuna_sched_path = ROOT / "reports" / "benchmarks" / "optuna_schedule" / "eggholder_optuna_k5_l4_v1" / "aggregate.json"
optuna_sched = json.loads(optuna_sched_path.read_text()) if optuna_sched_path.exists() else {}

# ----------------------------- Vision model-size -----------------------------
model_size = pd.read_csv(ROOT / "coarse-to-fine-curriculum" / "wandb_analysis_cifar100_model_size_updated" / "raw_runs_with_auc.csv")
best_curr = pd.read_csv(ROOT / "coarse-to-fine-curriculum" / "wandb_analysis_cifar100_model_size_updated" / "best_curriculum_by_model.csv")
length_pivot = pd.read_csv(ROOT / "coarse-to-fine-curriculum" / "wandb_analysis_cifar100_model_size_updated" / "best_test_acc_pivot.csv")
alt = pd.read_csv(ROOT / "coarse-to-fine-curriculum" / "wandb_analysis_cifar100_model_size_updated" / "best_curriculum_alt_metric_gains.csv")

# heatmap of curriculum length gain over baseline
lp = length_pivot.copy()
model_col = "model_label" if "model_label" in lp.columns else lp.columns[0]
# infer baseline and CL columns
value_cols = [c for c in lp.columns if c not in {model_col, "family", "summary_num_trainable_parameters"}]
# normalize possible names
lp_long = lp.melt(id_vars=[model_col], value_vars=value_cols, var_name="length", value_name="best_acc")
lp_long["length_num"] = lp_long["length"].astype(str).str.extract(r"(\d+)").fillna(0).astype(int)
base = lp_long[lp_long["length_num"] == 0][[model_col, "best_acc"]].rename(columns={"best_acc": "base_acc"})
lp_long = lp_long.merge(base, on=model_col, how="left")
lp_long["gain_pp"] = 100 * (lp_long["best_acc"] - lp_long["base_acc"])
lp_long.to_csv(OUT / "cifar100_length_gain_long.csv", index=False)
heat_len = lp_long[lp_long["length_num"] > 0].pivot(index=model_col, columns="length_num", values="gain_pp")
plt.figure(figsize=(7.8, 5.2))
mx = np.nanmax(np.abs(heat_len.values))
im = plt.imshow(heat_len.values, cmap="coolwarm", aspect="auto", vmin=-mx, vmax=mx)
plt.colorbar(im, label="Best accuracy gain over baseline (pp)")
plt.xticks(range(len(heat_len.columns)), heat_len.columns)
plt.yticks(range(len(heat_len.index)), [esc(i) for i in heat_len.index])
for i in range(heat_len.shape[0]):
    for j in range(heat_len.shape[1]):
        val = heat_len.iloc[i, j]
        if not pd.isna(val):
            plt.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8)
plt.xlabel("Curriculum epochs")
plt.title("CIFAR-100 curriculum length sensitivity")
len_heat_fig = savefig("deep_cifar100_curriculum_length_heatmap.png")

# milestone gains for best curriculum per model from raw columns
milestone_cols = ["test_acc_epoch_10", "test_acc_epoch_20", "test_acc_epoch_50", "test_acc_epoch_100"]
ms_rows = []
for model, sub in model_size.groupby("model_label"):
    base_rows = sub[sub["label"] == "baseline"]
    if base_rows.empty:
        continue
    base_row = base_rows.iloc[0]
    curr = sub[sub["label"] != "baseline"].sort_values("summary_best_test_acc", ascending=False)
    if curr.empty:
        continue
    curr_row = curr.iloc[0]
    for c in milestone_cols:
        if c in sub.columns:
            ms_rows.append({"model_label": model, "epoch": int(c.split("_")[-1]), "gain_pp": 100*(curr_row[c]-base_row[c]), "best_label": curr_row["label"]})
ms = pd.DataFrame(ms_rows)
ms.to_csv(OUT / "cifar100_milestone_gains.csv", index=False)
plt.figure(figsize=(8.4, 5.0))
for model, sub in ms.groupby("model_label"):
    plt.plot(sub["epoch"], sub["gain_pp"], marker="o", label=model)
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Epoch")
plt.ylabel("Best-curriculum gain at epoch (pp)")
plt.title("CIFAR-100: when does the best curriculum overtake baseline?")
plt.legend(fontsize=7, ncol=2)
ms_fig = savefig("deep_cifar100_milestone_gains.png")

# Alternative metric gains heatmap
metric_cols = [c for c in alt.columns if "gain" in c.lower() or "delta" in c.lower()]
if "model_label" not in alt.columns and "Model" in alt.columns:
    alt = alt.rename(columns={"Model": "model_label"})
alt_h = alt.set_index("model_label")[metric_cols]
plt.figure(figsize=(8.0, 4.8))
mx = np.nanmax(np.abs(alt_h.values))
im = plt.imshow(alt_h.values, cmap="coolwarm", aspect="auto", vmin=-mx, vmax=mx)
plt.colorbar(im, label="Change vs baseline")
plt.xticks(range(len(alt_h.columns)), [esc(c) for c in alt_h.columns], rotation=35, ha="right", fontsize=8)
plt.yticks(range(len(alt_h.index)), [esc(i) for i in alt_h.index], fontsize=8)
plt.title("CIFAR-100: curriculum effects across metrics")
alt_fig = savefig("deep_cifar100_alt_metric_heatmap.png")

# ----------------------------- Roughness/adaptive follow-up -----------------------------
rough_dir = ROOT / "coarse-to-fine-curriculum" / "wandb_analysis_roughness_followup_updated"
rough_agg = pd.read_csv(rough_dir / "aggregate_paired_deltas.csv")
rough_agg.to_csv(OUT / "vision_roughness_aggregate_paired_deltas.csv", index=False)

plt.figure(figsize=(9.2, 4.2))
for idx, metric in enumerate(["log10_final_sharp_mean", "log10_final_hess_mean"], start=1):
    plt.subplot(1, 2, idx)
    for ds, sub in rough_agg.groupby("dataset"):
        plt.scatter(sub[metric], 100*sub["gain_best_mean"], label=ds, s=60, alpha=0.85)
        for _, r in sub.iterrows():
            plt.annotate(r["model_label"], (r[metric], 100*r["gain_best_mean"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel(metric.replace("log10_", "log10 ").replace("_mean", ""))
    plt.ylabel("Best accuracy gain (pp)")
    plt.title("Gain vs " + ("sharpness" if "sharp" in metric else "Hessian"))
plt.legend(fontsize=7, loc="best")
rough_scatter_fig = savefig("deep_roughness_gain_scatter.png")

# Dataset/model gain heatmap: choose labels rows
rough_agg["row"] = rough_agg["dataset"] + " / " + rough_agg["model_label"] + " / " + rough_agg["label"]
ra = rough_agg.sort_values(["dataset", "model_label", "label"])
plt.figure(figsize=(7.0, max(4.5, 0.28*len(ra))))
vals = (100*ra["gain_best_mean"]).to_numpy().reshape(-1, 1)
mx = max(1, np.nanmax(np.abs(vals)))
im = plt.imshow(vals, cmap="coolwarm", aspect="auto", vmin=-mx, vmax=mx)
plt.colorbar(im, label="Mean best-accuracy gain (pp)")
plt.xticks([0], ["gain"])
plt.yticks(range(len(ra)), [esc(x) for x in ra["row"]], fontsize=7)
for i, val in enumerate(vals[:, 0]):
    plt.text(0, i, f"{val:+.2f}", ha="center", va="center", fontsize=7)
plt.title("Vision follow-up: paired curriculum gains")
gain_heat_fig = savefig("deep_vision_followup_gain_heatmap.png")

# Adaptive stage allocation
stage = pd.read_csv(ROOT / "coarse-to-fine-curriculum" / "wandb_analysis_adaptive_followup" / "stage_summaries.csv")
ad = stage[stage["policy_label"].str.contains("adaptive", na=False)].copy()
ad_sum = ad.groupby(["dataset", "model_label", "policy_label"], as_index=False).agg(
    coarse_epochs=("coarse_epochs", "mean"), fine_tune_epochs=("fine_tune_epochs", "mean"), n=("wandb_id", "count")
)
ad_sum.to_csv(OUT / "adaptive_stage_allocation_summary.csv", index=False)
ad_sum["row"] = ad_sum["dataset"] + " / " + ad_sum["model_label"] + " / " + ad_sum["policy_label"]
plt.figure(figsize=(8.2, max(4.5, 0.28*len(ad_sum))))
y = np.arange(len(ad_sum))
plt.barh(y, ad_sum["coarse_epochs"], label="coarse epochs")
plt.barh(y, ad_sum["fine_tune_epochs"], left=ad_sum["coarse_epochs"], label="fine-label epochs to peak")
plt.yticks(y, [esc(x) for x in ad_sum["row"]], fontsize=7)
plt.xlabel("Epochs up to peak test accuracy")
plt.title("Adaptive plateau schedules: stage allocation")
plt.legend()
stage_fig = savefig("deep_adaptive_stage_allocation.png")

# Hierarchy ablation
hier = pd.read_csv(ROOT / "coarse-to-fine-curriculum" / "wandb_analysis_hierarchy_ablation" / "paired_hierarchy_deltas_by_seed.csv")
hier.to_csv(OUT / "hierarchy_ablation_seed_deltas.csv", index=False)
plt.figure(figsize=(7.2, 4.6))
x = np.arange(len(hier))
width = 0.25
plt.bar(x - width, 100*hier["learned_gain"], width, label="learned")
plt.bar(x, 100*hier["random_mean_gain"], width, label="random mean")
plt.bar(x + width, 100*hier["random_best_of_seeds_gain"], width, label="best random")
plt.axhline(0, color="black", linewidth=1)
plt.xticks(x, [str(s) for s in hier["seed"]])
plt.xlabel("Training seed")
plt.ylabel("Gain over baseline (pp)")
plt.title("Hierarchy ablation: learned hierarchy versus random controls")
plt.legend()
hier_fig = savefig("deep_hierarchy_seed_gains.png")

# ----------------------------- LaTeX appendix -----------------------------
# compact table strings
def regression_delta_rows() -> str:
    rows = []
    tmp = delta.sort_values(["regime", "function"])
    for _, r in tmp.iterrows():
        rows.append(
            f"{esc(r['function'])} & {esc(r['regime'])} & {r['delta_quality_single_minus_curriculum']:.3g} & {r['runtime_ratio_curr_over_single']:.2f} & {esc(r['quality_winner'])} \\\\"
        )
    return "\n".join(rows)


def best_v3_rows() -> str:
    rows = []
    for _, r in best_v3.sort_values("function").iterrows():
        loss = r["final_test_loss"] if not pd.isna(r["final_test_loss"]) else r["best_val_loss"]
        rows.append(f"{esc(r['function'])} & {esc(r['model_arch'])} & {int(r['n_params']) if not pd.isna(r['n_params']) else '--'} & {r['best_val_loss']:.3g} & {loss:.3g} \\\\")
    return "\n".join(rows)


def rough_rows() -> str:
    rows = []
    cols = ["dataset", "model_label", "label", "gain_best_mean", "gain_auc_mean", "delta_ece_mean", "log10_final_sharp_mean", "log10_final_hess_mean"]
    for _, r in rough_agg.sort_values(["dataset", "model_label", "label"]).iterrows():
        rows.append(f"{esc(r['dataset'])} & {esc(r['model_label'])} & {esc(r['label'])} & {pp(r['gain_best_mean'])} & {pp(r['gain_auc_mean'])} & {pp(r['delta_ece_mean'])} & {r['log10_final_sharp_mean']:.2f} & {r['log10_final_hess_mean']:.2f} \\\\")
    return "\n".join(rows)

policy_note = ""
if not policy.empty:
    top = policy.head(5)
    optuna_loss = optuna_sched.get("mean_best_hard_val_loss")
    policy_note = "\n".join(
        f"{esc(r['candidate_id'])} & {r['mean_best_hard_val_loss']:.1f} & {r['std_best_hard_val_loss']:.1f} & {r['mean_epochs_trained']:.1f} \\\\" for _, r in top.iterrows()
    )
    if optuna_loss is not None:
        policy_note += f"\nOptuna parametric schedule & {optuna_loss:.1f} & {optuna_sched.get('std_best_hard_val_loss', float('nan')):.1f} & {optuna_sched.get('mean_epochs_trained', float('nan')):.1f} \\\\"

tex = rf"""
\chapter{{Additional Deep-Dive Analysis of Experiment Runs}}
\label{{app:deep-dive-experiments}}

This appendix intentionally collects more detail than the main thesis needs. It
is a staging area for exploratory analysis of the Runpod/W\&B vision runs and
the MLflow regression runs. The purpose is to expose patterns that may later be
pruned into the main text or converted into a smaller set of final figures.

\section{{Regression continuation: quality--runtime trade-off}}

Figure~\ref{{fig:deep-regression-tradeoff}} compares the median quality gain of
curriculum training against its runtime cost for the controlled
\texttt{{curr\_vs\_single\_20260315\_175826}} benchmark. The horizontal axis is
larger than one for every function and budget regime, which means that the
curriculum was always slower. The vertical axis is positive only when curriculum
reduced hard-validation loss.

\begin{{figure}}[h]
    \centering
    \includegraphics[width=0.86\textwidth]{{{reg_tradeoff_fig}}}
    \caption{{Quality gain versus runtime cost for Gaussian-continuation
    regression. Positive y-values favor curriculum; x-values above one mean the
    curriculum run was slower than direct single-target training.}}
    \label{{fig:deep-regression-tradeoff}}
\end{{figure}}

\begin{{table}}[h]
\centering
\small
\begin{{tabular}}{{llrrl}}
\toprule
Function & Regime & Quality delta & Runtime ratio & Winner \\
\midrule
{regression_delta_rows()}
\bottomrule
\end{{tabular}}
\caption{{Function-level regression deltas. Quality delta is median
single-target hard-validation loss minus median curriculum hard-validation loss,
so positive values favor curriculum. Runtime ratio is curriculum runtime divided
by single-target runtime.}}
\label{{tab:deep-regression-deltas}}
\end{{table}}

Figure~\ref{{fig:deep-regression-seed-heatmap}} shows the same comparison after
pairing runs by seed. The sign pattern is not random: \texttt{{bukin}},
\texttt{{franke}}, and \texttt{{levy}} are the clearest positive cases, while
\texttt{{ackley}}, \texttt{{eggholder}}, and \texttt{{peaks}} are negative in
both budget regimes.

\begin{{figure}}[h]
    \centering
    \includegraphics[width=0.72\textwidth]{{{reg_heat_fig}}}
    \caption{{Median seed-paired hard-validation loss delta by function and
    budget regime. Positive values favor curriculum.}}
    \label{{fig:deep-regression-seed-heatmap}}
\end{{figure}}

\section{{MLflow v3 ratio sweeps}}

The v3 ratio-constrained sweeps add a stricter setting than the original sweep
pack: noisy labels, train/test files, and an explicit parameter-to-data ratio.
Table~\ref{{tab:deep-v3-best}} reports the best finished MLflow run per function
according to validation loss. Figure~\ref{{fig:deep-v3-best-losses}} visualizes
how uneven the functions are: \texttt{{eggholder}} remains orders of magnitude
harder than the smoother benchmark functions.

\begin{{table}}[h]
\centering
\small
\begin{{tabular}}{{llrrr}}
\toprule
Function & Best arch. & Params & Best val loss & Test/final loss \\
\midrule
{best_v3_rows()}
\bottomrule
\end{{tabular}}
\caption{{Best MLflow v3 ratio-sweep run per function. The final column uses
final test loss when available and otherwise falls back to best validation loss.}}
\label{{tab:deep-v3-best}}
\end{{table}}

\begin{{figure}}[h]
    \centering
    \includegraphics[width=0.86\textwidth]{{{mlflow_best_fig}}}
    \caption{{Best available loss from the v3 ratio-constrained MLflow sweeps.
    Bar labels show the architecture of the best trial.}}
    \label{{fig:deep-v3-best-losses}}
\end{{figure}}

\begin{{figure}}[h]
    \centering
    \includegraphics[width=0.86\textwidth]{{{mlflow_arch_fig}}}
    \caption{{Architecture composition among the top 20 v3 sweep trials per
    function. This is more informative than only reporting the single best trial,
    because it shows whether an architecture family is consistently competitive.}}
    \label{{fig:deep-v3-arch-distribution}}
\end{{figure}}

\section{{Exploratory schedule-policy search}}

The repository also contains exploratory policy-schedule benchmarks for
\texttt{{eggholder}}. These are not as central as the controlled curriculum and
vision studies, but they are useful because they show that automatic schedule
search can find competitive regions while remaining noisy across seeds. Lower
loss is better.

\begin{{table}}[h]
\centering
\small
\begin{{tabular}}{{lrrr}}
\toprule
Schedule & Mean best hard-val loss & Std. & Mean epochs \\
\midrule
{policy_note if policy_note else 'No policy aggregate files found. & -- & -- & -- \\\\'}
\bottomrule
\end{{tabular}}
\caption{{Top exploratory policy schedules and the best Optuna parametric
schedule when available. These runs are best read as schedule-search evidence,
not as final method comparisons.}}
\label{{tab:deep-policy-search}}
\end{{table}}

\section{{CIFAR-100 curriculum length and capacity}}

Figure~\ref{{fig:deep-cifar100-length}} expands the model-size appendix by
showing every tested curriculum length as a gain over the matching baseline.
The within-family CNN pattern is especially clear: as width increases, the best
curriculum becomes shorter and eventually disappears.

\begin{{figure}}[h]
    \centering
    \includegraphics[width=0.86\textwidth]{{{len_heat_fig}}}
    \caption{{CIFAR-100 curriculum-length sensitivity. Values are best-test
    accuracy gains in percentage points relative to the matching baseline.}}
    \label{{fig:deep-cifar100-length}}
\end{{figure}}

Figure~\ref{{fig:deep-cifar100-milestones}} asks when the best curriculum run is
above or below the baseline. Many curricula are below the baseline early and
recover later, which explains why peak accuracy and AUC tell different stories.

\begin{{figure}}[h]
    \centering
    \includegraphics[width=0.86\textwidth]{{{ms_fig}}}
    \caption{{Milestone test-accuracy gains of each model's best curriculum run
    relative to its baseline. Negative early values indicate the fine-label
    delay introduced by coarse training.}}
    \label{{fig:deep-cifar100-milestones}}
\end{{figure}}

\begin{{figure}}[h]
    \centering
    \includegraphics[width=0.86\textwidth]{{{alt_fig}}}
    \caption{{CIFAR-100 metric heatmap for the best curriculum run of each
    model. Accuracy, macro-F1, top-5, hierarchy-aware scores, and calibration do
    not always move together.}}
    \label{{fig:deep-cifar100-alt-metrics}}
\end{{figure}}

\section{{Vision follow-up: gains, roughness, and schedule allocation}}

Table~\ref{{tab:deep-roughness-deltas}} integrates the paired follow-up deltas
from the W\&B exports. The gain columns are reported in percentage points.
Negative AUC gains are common, reinforcing the interpretation that curricula can
improve the best solution reached while still delaying the fine-label trajectory.

\begin{{table}}[h]
\centering
\scriptsize
\begin{{tabular}}{{lllrrrrr}}
\toprule
Dataset & Model & Label & Best gain & AUC gain & ECE delta & Final sharp. & Final Hess. \\
\midrule
{rough_rows()}
\bottomrule
\end{{tabular}}
\caption{{Paired W\&B follow-up deltas by dataset, model, and curriculum label.
Sharpness and Hessian columns are log10 curriculum/baseline ratios; negative
values mean lower diagnostic value for curriculum.}}
\label{{tab:deep-roughness-deltas}}
\end{{table}}

\begin{{figure}}[h]
    \centering
    \includegraphics[width=0.92\textwidth]{{{rough_scatter_fig}}}
    \caption{{Relationship between best-accuracy gain and roughness diagnostics.
    The mixed pattern supports using roughness as a diagnostic, not as proof of a
    universal mechanism.}}
    \label{{fig:deep-roughness-scatter}}
\end{{figure}}

\begin{{figure}}[h]
    \centering
    \includegraphics[width=0.74\textwidth]{{{gain_heat_fig}}}
    \caption{{Paired best-accuracy gains across all vision follow-up settings.}}
    \label{{fig:deep-vision-gain-heatmap}}
\end{{figure}}

\begin{{figure}}[h]
    \centering
    \includegraphics[width=0.86\textwidth]{{{stage_fig}}}
    \caption{{Adaptive plateau stage allocation. Larger CIFAR-100 CNNs often
    spend too long in coarse stages relative to the short fixed curricula that
    worked best in the screening study.}}
    \label{{fig:deep-adaptive-stage-allocation}}
\end{{figure}}

\section{{Hierarchy ablation details}}

Figure~\ref{{fig:deep-hierarchy-seed-gains}} shows the seed-level random-hierarchy
control. The learned hierarchy beats the random mean for every seed and also
beats the best random hierarchy tried for each seed. This is stronger than a
single aggregate number because it rules out the explanation that one lucky seed
created the learned-hierarchy advantage.

\begin{{figure}}[h]
    \centering
    \includegraphics[width=0.78\textwidth]{{{hier_fig}}}
    \caption{{Seed-level hierarchy ablation gains for CIFAR-100
    \texttt{{cnn\_w0.5}}.}}
    \label{{fig:deep-hierarchy-seed-gains}}
\end{{figure}}

\section{{What these additional plots suggest}}

The deeper analysis strengthens four thesis-level points. First, the regression
track is genuinely mixed: continuation sometimes improves quality, but the
runtime cost is systematic. Second, the vision track has a clearer positive
region, but that region is bounded by capacity, schedule length, and dataset
structure. Third, adaptive schedules fail in an interpretable way: they often
allocate too many epochs to coarse objectives. Fourth, roughness diagnostics are
interesting but not decisive; they help characterize runs, but they do not yet
explain all positive curriculum effects.
"""
APP.write_text(tex, encoding="utf-8")
print(f"Wrote {APP}")
print(f"Figures written to {FIG}")
print(f"CSVs written to {OUT}")
