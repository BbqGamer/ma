#!/usr/bin/env python3
"""Build a run-by-run CIFAR-100 report from a result bundle."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import tarfile
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import typer

app = typer.Typer(add_completion=False)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _extract_bundle(bundle_path: Path) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="cifar100_bundle_"))
    with tarfile.open(bundle_path, "r:gz") as tar:
        tar.extractall(tmpdir)
    return tmpdir


def _resolve_input_root(path: Path) -> Path:
    if path.is_dir():
        return path
    return _extract_bundle(path)


def _to_float_list(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def _extract_reasoning_lines(code: str) -> list[str]:
    lines: list[str] = []
    for raw in code.splitlines():
        stripped = raw.strip()
        if not stripped and not lines:
            continue
        if stripped.startswith("#"):
            lines.append(stripped.removeprefix("#").strip())
            continue
        break
    return lines


def _code_snippet(code: str) -> str:
    return code.rstrip()


def _plot_run(trajectory_path: Path, *, title: str, output_path: Path) -> dict[str, Any]:
    rows = _read_csv(trajectory_path)
    epochs = _to_float_list(rows, "epoch")
    weight_easy = _to_float_list(rows, "weight_easy")
    weight_hard = _to_float_list(rows, "weight_hard")
    train_easy_loss = [float(row["train_easy_loss"]) if row["train_easy_loss"] else None for row in rows]
    train_hard_loss = [float(row["train_hard_loss"]) if row["train_hard_loss"] else None for row in rows]
    val_easy_loss = _to_float_list(rows, "val_easy_loss")
    val_hard_loss = _to_float_list(rows, "val_hard_loss")
    best_val_hard_loss = _to_float_list(rows, "best_val_hard_loss")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(epochs, weight_easy, label="easy weight", color="tab:green", linewidth=2)
    axes[0].plot(epochs, weight_hard, label="hard weight", color="tab:blue", linewidth=2)
    axes[0].set_ylabel("Weight")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(loc="best")
    axes[0].set_title(title)

    def _plot_optional(series: list[float | None], label: str, color: str, linestyle: str = "-") -> None:
        xs = [x for x, y in zip(epochs, series, strict=True) if y is not None]
        ys = [y for y in series if y is not None]
        if xs:
            axes[1].plot(xs, ys, label=label, color=color, linestyle=linestyle, linewidth=1.8)

    _plot_optional(train_easy_loss, "train easy loss", "tab:green", "-")
    _plot_optional(train_hard_loss, "train hard loss", "tab:blue", "-")
    axes[1].plot(epochs, val_easy_loss, label="val easy loss", color="tab:olive", linestyle="--")
    axes[1].plot(epochs, val_hard_loss, label="val hard loss", color="tab:red", linestyle="--")
    axes[1].plot(epochs, best_val_hard_loss, label="running best val hard loss", color="black", linestyle=":")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="best", ncol=2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    best_idx = min(range(len(val_hard_loss)), key=lambda i: val_hard_loss[i])
    return {
        "epochs": len(rows),
        "hard_weight_start": weight_hard[0],
        "hard_weight_end": weight_hard[-1],
        "hard_weight_min": min(weight_hard),
        "hard_weight_max": max(weight_hard),
        "best_epoch": int(epochs[best_idx]),
        "best_val_hard_loss": val_hard_loss[best_idx],
        "final_val_hard_loss": val_hard_loss[-1],
    }


def _plot_objective_distribution(runs: list[dict[str, Any]], output_path: Path) -> None:
    llm_vals = [run["summary"]["best_hard_val_loss"] for run in runs if run["family"] == "llm"]
    opt_vals = [run["summary"]["best_hard_val_loss"] for run in runs if run["family"] == "optuna"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([llm_vals, opt_vals], tick_labels=["LLM", "Optuna"], widths=0.5)
    ax.scatter([1] * len(llm_vals), llm_vals, color="tab:blue", alpha=0.8)
    ax.scatter([2] * len(opt_vals), opt_vals, color="tab:orange", alpha=0.8)
    ax.set_ylabel("Best hard validation loss")
    ax.set_title("Overall search outcome distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_best_so_far(
    llm_history_rows: list[dict[str, Any]],
    optuna_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    llm_best: list[float] = []
    current_best: float | None = None
    for row in llm_history_rows:
        if row.get("status") == "ok":
            value = float(row["mean_best_hard_val_loss"])
            current_best = value if current_best is None else min(current_best, value)
        llm_best.append(current_best if current_best is not None else float("nan"))

    opt_best: list[float] = []
    current_best = None
    for row in optuna_rows:
        value = float(row["mean_best_hard_val_loss"])
        current_best = value if current_best is None else min(current_best, value)
        opt_best.append(current_best)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(llm_best) + 1), llm_best, marker="o", color="tab:blue", label="LLM")
    ax.plot(range(1, len(opt_best) + 1), opt_best, marker="o", color="tab:orange", label="Optuna")
    ax.set_xlabel("Candidate / trial index")
    ax.set_ylabel("Best-so-far hard validation loss")
    ax.set_title("Search progress")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_loss_vs_accuracy(runs: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for family, color in [("llm", "tab:blue"), ("optuna", "tab:orange")]:
        family_runs = [run for run in runs if run["family"] == family]
        xs = [run["summary"]["best_hard_val_loss"] for run in family_runs]
        ys = [run["summary"]["best_hard_val_acc"] for run in family_runs]
        ax.scatter(xs, ys, color=color, label=family.upper(), alpha=0.85)
    ax.set_xlabel("Best hard validation loss")
    ax.set_ylabel("Best hard validation accuracy")
    ax.set_title("Loss-accuracy tradeoff across runs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_weight_geometry(runs: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for family, color in [("llm", "tab:blue"), ("optuna", "tab:orange")]:
        family_runs = [run for run in runs if run["family"] == family]
        starts = [run["metrics"]["hard_weight_start"] for run in family_runs]
        ends = [run["metrics"]["hard_weight_end"] for run in family_runs]
        losses = [run["summary"]["best_hard_val_loss"] for run in family_runs]
        axes[0].scatter(starts, ends, color=color, alpha=0.85, label=family.upper())
        axes[1].scatter(ends, losses, color=color, alpha=0.85, label=family.upper())

    axes[0].set_xlabel("Start hard weight")
    axes[0].set_ylabel("End hard weight")
    axes[0].set_title("Schedule geometry")
    axes[0].legend()

    axes[1].set_xlabel("End hard weight")
    axes[1].set_ylabel("Best hard validation loss")
    axes[1].set_title("Outcome vs late hard emphasis")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _collect_successful_runs(extract_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    llm_history_rows: list[dict[str, Any]] = []
    optuna_rows: list[dict[str, Any]] = []

    llm_root = extract_root / "reports" / "cifar100_easy_hard" / "llm_search"
    if llm_root.exists():
        for study_dir in sorted(path for path in llm_root.iterdir() if path.is_dir()):
            history_path = study_dir / "history.jsonl"
            if not history_path.exists():
                continue
            study_rows = _read_jsonl(history_path)
            llm_history_rows.extend(study_rows)
            for row in study_rows:
                if row.get("status") == "ok":
                    benchmark_id = str(row["benchmark_id"])
                    run_dir = (
                        extract_root
                        / "reports"
                        / "cifar100_easy_hard"
                        / "policy"
                        / benchmark_id
                        / "seed_42"
                    )
                    summary = _read_json(run_dir / "summary.json")
                    candidate_file = (
                        extract_root
                        / "llm_schedules"
                        / "generated"
                        / study_dir.name
                        / f"candidate_{row['candidate_id']}.py"
                    )
                    code = candidate_file.read_text(encoding="utf-8") if candidate_file.exists() else "# missing"
                    prompt_path = study_dir / "prompts" / f"candidate_{row['candidate_id']}.md"
                    prompt_text = (
                        prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else "# missing"
                    )
                    runs.append(
                        {
                            "family": "llm",
                            "study_name": study_dir.name,
                            "run_id": str(row["candidate_id"]),
                            "label": f"LLM {row['candidate_id']}",
                            "sort_metric": float(row["mean_best_hard_val_loss"]),
                            "trajectory_path": run_dir / "trajectory.csv",
                            "summary": summary,
                            "row": row,
                            "code": code,
                            "prompt_text": prompt_text,
                            "reasoning_lines": _extract_reasoning_lines(code),
                        }
                    )
                else:
                    failures.append(
                        {
                            "family": "llm",
                            "study_name": study_dir.name,
                            "run_id": str(row.get("candidate_id", "?")),
                            "error": str(row.get("error", "")),
                            "status": str(row.get("status", "failed")),
                        }
                    )

    optuna_root = extract_root / "reports" / "cifar100_easy_hard" / "optuna"
    if optuna_root.exists():
        for study_dir in sorted(path for path in optuna_root.iterdir() if path.is_dir()):
            all_trials_path = study_dir / "all_trials.jsonl"
            if not all_trials_path.exists():
                continue
            study_rows = _read_jsonl(all_trials_path)
            optuna_rows.extend(study_rows)
            for row in study_rows:
                run_id = str(row["candidate_id"])
                run_dir = study_dir / "trials" / run_id / "seed_42"
                summary = _read_json(run_dir / "summary.json")
                runs.append(
                    {
                        "family": "optuna",
                        "study_name": study_dir.name,
                        "run_id": run_id,
                        "label": f"Optuna {run_id}",
                        "sort_metric": float(row["mean_best_hard_val_loss"]),
                        "trajectory_path": run_dir / "trajectory.csv",
                        "summary": summary,
                        "row": row,
                    }
                )

    runs.sort(key=lambda run: (run["family"], run["sort_metric"]))
    failures.sort(key=lambda row: (row["family"], row["run_id"]))
    return runs, failures, llm_history_rows, optuna_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


@app.command()
def main(
    bundle_path: Path = typer.Argument(..., exists=True, readable=True),
    output_dir: Path = typer.Option(
        Path("reports/analysis/cifar100_easy_hard/all_runs_report"),
        help="Directory for the generated report.",
    ),
) -> None:
    extract_root = _resolve_input_root(bundle_path)
    runs, failures, llm_history_rows, optuna_rows = _collect_successful_runs(extract_root)

    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    enriched_rows: list[dict[str, Any]] = []
    sections: list[str] = []

    grouped = {
        "llm": [run for run in runs if run["family"] == "llm"],
        "optuna": [run for run in runs if run["family"] == "optuna"],
    }

    for family_runs in grouped.values():
        for run in family_runs:
            title = (
                f"{run['label']} | best val hard loss={run['summary']['best_hard_val_loss']:.4f} | "
                f"best val hard acc={run['summary']['best_hard_val_acc']:.4f}"
            )
            idx = family_runs.index(run) + 1
            figure_name = f"{run['family']}_{idx:02d}_{run['run_id']}.png"
            run["figure_path"] = figures_dir / figure_name
            run["metrics"] = _plot_run(
                run["trajectory_path"],
                title=title,
                output_path=run["figure_path"],
            )

    objective_distribution_path = figures_dir / "objective_distribution.png"
    best_so_far_path = figures_dir / "best_so_far.png"
    loss_vs_accuracy_path = figures_dir / "loss_vs_accuracy.png"
    weight_geometry_path = figures_dir / "weight_geometry.png"

    _plot_objective_distribution(runs, objective_distribution_path)
    _plot_best_so_far(llm_history_rows, optuna_rows, best_so_far_path)
    _plot_loss_vs_accuracy(runs, loss_vs_accuracy_path)
    _plot_weight_geometry(runs, weight_geometry_path)

    llm_losses = [run["summary"]["best_hard_val_loss"] for run in grouped["llm"]]
    optuna_losses = [run["summary"]["best_hard_val_loss"] for run in grouped["optuna"]]
    best_llm = min(grouped["llm"], key=lambda run: run["summary"]["best_hard_val_loss"])
    best_optuna = min(grouped["optuna"], key=lambda run: run["summary"]["best_hard_val_loss"])

    sections.append("# CIFAR-100 run-by-run diagnostics report\n")
    sections.append(
        f"Bundle: `{bundle_path.name}`  \nSuccessful plotted runs: **{len(runs)}**  \nLLM failures without trajectories: **{len(failures)}**\n"
    )
    sections.append(
        "This report combines per-run training diagnostics with code-level inspection of the LLM-generated schedules. Each successful run includes: (1) easy/hard weights across epochs, (2) easy/hard training and validation losses, and for LLM runs also (3) the top reasoning comments and a code snippet from the generated module.\n"
    )

    sections.append("## Overall LLM vs Optuna comparison\n")
    sections.append(
        f"- Best LLM run: **{best_llm['run_id']}** with best hard val loss **{best_llm['summary']['best_hard_val_loss']:.4f}**  \n"
        f"- Best Optuna run: **{best_optuna['run_id']}** with best hard val loss **{best_optuna['summary']['best_hard_val_loss']:.4f}**  \n"
        f"- Mean LLM best hard val loss: **{statistics.mean(llm_losses):.4f}**  \n"
        f"- Mean Optuna best hard val loss: **{statistics.mean(optuna_losses):.4f}**\n"
    )
    sections.append(
        "The plots below summarize overall performance rather than individual runs. They make it easier to judge whether the search was coherent, whether one method dominated, and whether the discovered schedules formed distinct families.\n"
    )
    sections.append(f"![Objective distribution]({objective_distribution_path.relative_to(output_dir).as_posix()})\n")
    sections.append(f"![Best so far]({best_so_far_path.relative_to(output_dir).as_posix()})\n")
    sections.append(f"![Loss vs accuracy]({loss_vs_accuracy_path.relative_to(output_dir).as_posix()})\n")
    sections.append(f"![Weight geometry]({weight_geometry_path.relative_to(output_dir).as_posix()})\n")
    sections.append(
        "**Interpretation.** The LLM runs cluster tightly in a narrow loss band and mostly share the same easy-first to hard-heavy geometry. Optuna explores a wider range of weight profiles; its best trial is competitive, but many lower-ranked trials are substantially worse. This makes the LLM search look more internally coherent on this single-seed run, while Optuna appears more variable.\n"
    )

    for family in ["llm", "optuna"]:
        family_runs = grouped[family]
        if not family_runs:
            continue
        title = "LLM runs" if family == "llm" else "Optuna runs"
        sections.append(f"## {title}\n")
        for run in family_runs:
            row = run["row"]
            summary = run["summary"]
            metrics = run["metrics"]
            enriched = {
                "family": family,
                "study_name": run["study_name"],
                "run_id": run["run_id"],
                "best_hard_val_loss": summary["best_hard_val_loss"],
                "best_hard_val_acc": summary["best_hard_val_acc"],
                "final_hard_val_loss": summary["final_hard_val_loss"],
                "test_hard_loss": summary["test_hard_loss"],
                "test_hard_acc": summary["test_hard_acc"],
                "epochs_trained": summary["epochs_trained"],
                "hard_weight_start": metrics["hard_weight_start"],
                "hard_weight_end": metrics["hard_weight_end"],
                "hard_weight_min": metrics["hard_weight_min"],
                "hard_weight_max": metrics["hard_weight_max"],
                "best_epoch": metrics["best_epoch"],
                "figure_path": str(run["figure_path"].relative_to(output_dir)),
            }
            if family == "llm":
                enriched["candidate_module"] = row.get("candidate_module")
                enriched["openai_total_tokens"] = row.get("openai_total_tokens")
                enriched["reasoning"] = " | ".join(run.get("reasoning_lines", []))
            else:
                enriched["schedule_params"] = json.dumps(row.get("schedule_params", {}), sort_keys=True)
            enriched_rows.append(enriched)

            sections.append(f"### {run['label']}\n")
            sections.append(
                f"- Study: `{run['study_name']}`  \n"
                f"- Best hard val loss: **{summary['best_hard_val_loss']:.4f}**  \n"
                f"- Best hard val acc: **{summary['best_hard_val_acc']:.4f}**  \n"
                f"- Final hard val loss: **{summary['final_hard_val_loss']:.4f}**  \n"
                f"- Test hard loss / acc: **{summary['test_hard_loss']:.4f} / {summary['test_hard_acc']:.4f}**  \n"
                f"- Hard weight start → end: **{metrics['hard_weight_start']:.3f} → {metrics['hard_weight_end']:.3f}**  \n"
                f"- Hard weight min/max: **{metrics['hard_weight_min']:.3f} / {metrics['hard_weight_max']:.3f}**  \n"
                f"- Best epoch: **{metrics['best_epoch']}**\n"
            )
            if family == "llm":
                sections.append(f"- Candidate module: `{row.get('candidate_module', '')}`  ")
                sections.append(f"- OpenAI total tokens: **{int(float(row.get('openai_total_tokens', 0.0) or 0.0)):,}**\n")
                reasoning_lines = run.get("reasoning_lines", [])
                if reasoning_lines:
                    sections.append("**LLM reasoning header**\n")
                    for reason in reasoning_lines:
                        sections.append(f"- {reason}")
                    sections.append("")
                sections.append("**Exact prompt shown to the LLM**\n")
                sections.append(f"```text\n{run['prompt_text'].rstrip()}\n```\n")
                sections.append("**Generated code**\n")
                sections.append(f"```python\n{_code_snippet(run['code'])}\n```\n")
            else:
                sections.append(
                    f"- Schedule params: `{json.dumps(row.get('schedule_params', {}), sort_keys=True)}`\n"
                )
            sections.append(f"![{run['label']}]({run['figure_path'].relative_to(output_dir).as_posix()})\n")

    if failures:
        sections.append("## Failed LLM attempts\n")
        sections.append("These runs did not produce trajectory files, so they are listed without plots.\n")
        for row in failures:
            sections.append(
                f"- `{row['study_name']}` / `{row['run_id']}`: `{row['error']}`\n"
            )

    (output_dir / "summary.md").write_text("\n".join(sections), encoding="utf-8")

    _write_csv(
        output_dir / "runs.csv",
        enriched_rows,
        [
            "family",
            "study_name",
            "run_id",
            "best_hard_val_loss",
            "best_hard_val_acc",
            "final_hard_val_loss",
            "test_hard_loss",
            "test_hard_acc",
            "epochs_trained",
            "hard_weight_start",
            "hard_weight_end",
            "hard_weight_min",
            "hard_weight_max",
            "best_epoch",
            "candidate_module",
            "openai_total_tokens",
            "reasoning",
            "schedule_params",
            "figure_path",
        ],
    )
    _write_csv(output_dir / "failures.csv", failures, ["family", "study_name", "run_id", "status", "error"])
    (output_dir / "report_summary.json").write_text(
        json.dumps(
            {
                "bundle_path": str(bundle_path.resolve()),
                "num_successful_runs": len(runs),
                "num_failed_runs": len(failures),
                "num_llm_runs": len(grouped["llm"]),
                "num_optuna_runs": len(grouped["optuna"]),
                "best_llm_loss": best_llm["summary"]["best_hard_val_loss"],
                "best_optuna_loss": best_optuna["summary"]["best_hard_val_loss"],
                "mean_llm_loss": statistics.mean(llm_losses),
                "mean_optuna_loss": statistics.mean(optuna_losses),
                "output_dir": str(output_dir.resolve()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"output_dir": str(output_dir.resolve()), "num_runs": len(runs)}, indent=2))


if __name__ == "__main__":
    app()
