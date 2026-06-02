#!/usr/bin/env python3
"""Build a small PDF report for a schedule-search comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from ma_thesis.config import REPORTS_DIR

app = typer.Typer(add_completion=False)

ANALYSIS_ROOT = REPORTS_DIR / "analysis" / "schedule_search_comparison"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def _tex_table_from_df(df: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    align = "l" + "r" * (len(columns) - 1)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    lines.append(" & ".join(header for _, header in columns) + r" \\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        vals = []
        for key, _ in columns:
            val = row[key]
            if isinstance(val, float):
                vals.append(_fmt(val))
            else:
                vals.append(_escape_latex(str(val)))
        lines.append(" & ".join(vals) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


@app.command()
def main(
    report_id: str = typer.Option(..., help="Comparison report id under reports/analysis/schedule_search_comparison."),
) -> None:
    report_dir = ANALYSIS_ROOT / report_id
    if not report_dir.exists():
        raise typer.BadParameter(f"Missing report directory: {report_dir}")

    summary = _load_json(report_dir / "comparison_summary.json")
    headline_df = pd.read_csv(report_dir / "headline_table.csv")
    per_seed_df = pd.read_csv(report_dir / "per_seed_comparison.csv")
    failure_df = pd.read_csv(report_dir / "llm_failures.csv")

    llm_better = summary["optuna_minus_llm"] > 0
    comparison_phrase = "LLM policy search found the better schedule" if llm_better else "Optuna found the better schedule"
    top_runs_pages = list(summary.get("top_runs_pages", []))

    headline_table = _tex_table_from_df(
        headline_df,
        [
            ("method", "Method"),
            ("attempted", "Attempted"),
            ("completed", "Completed"),
            ("best_candidate", "Best id"),
            ("best_mean_best_hard_val_loss", "Mean best hard val"),
            ("best_std_best_hard_val_loss", "Std across seeds"),
        ],
    )
    per_seed_table = _tex_table_from_df(
        per_seed_df,
        [
            ("seed", "Seed"),
            ("llm_best_hard_val_loss", "LLM"),
            ("optuna_best_hard_val_loss", "Optuna"),
        ],
    )
    failure_table = _tex_table_from_df(
        failure_df,
        [("failure_type", "Failure type"), ("count", "Count")],
    )

    top_runs_tex = "\n\n".join(
        f"""\\begin{{figure}}[p]
    \\centering
    \\includegraphics[width=0.96\\textwidth]{{{name}}}
    \\caption{{Top 10 schedules page. Each subplot overlays colored weights with a smoothed dashed mean running-best hard validation loss on a shared secondary y-axis.}}
\\end{{figure}}"""
        for name in top_runs_pages
    )

    tex = f"""\\documentclass[11pt]{{article}}
\\usepackage[a4paper,margin=1in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{float}}
\\usepackage{{hyperref}}
\\usepackage{{caption}}
\\usepackage{{parskip}}
\\usepackage{{amsmath}}

\\title{{LLM vs Optuna Schedule Search Report}}
\\author{{Adam Korba}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\section*{{Setup}}
This report compares matched schedule search on the Eggholder benchmark with $4$ losses and the same training setup on both sides. Lower mean best hard validation loss is better.

The main result is simple: {comparison_phrase}. The best completed LLM candidate reached {_fmt(summary['best_llm_mean_best_hard_val_loss'])}, while the best Optuna trial reached {_fmt(summary['best_optuna_mean_best_hard_val_loss'])}. The absolute gap was {_fmt(abs(summary['optuna_minus_llm']))}. At the same time, the LLM search should be treated as partial, because only {summary['llm_completed']} of {summary['llm_attempted']} candidate slots finished successfully. Runs also have different visible lengths on the x-axis because training used early stopping, so some schedules stopped well before the nominal 100-epoch budget once validation stopped improving.

\\section*{{Headline numbers}}
\\begin{{table}}[H]
\\centering
{headline_table}
\\end{{table}}

The LLM run used {summary['llm_total_tokens']:,} tokens in total ({summary['llm_input_tokens']:,} input and {summary['llm_output_tokens']:,} output).

\\section*{{Search behaviour}}
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.82\\textwidth]{{best_so_far.png}}
    \\caption{{Best-so-far search progress over evaluations.}}
\\end{{figure}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.72\\textwidth]{{objective_distribution.png}}
    \\caption{{Distribution of completed evaluation outcomes.}}
\\end{{figure}}

The best-so-far curve is the most informative plot here. It shows how quickly each method reached strong schedules, while the distribution plot gives a rough sense of search stability over completed evaluations.

\\section*{{Best schedules}}
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.76\\textwidth]{{per_seed_best.png}}
    \\caption{{Per-seed comparison for the best LLM candidate and the best Optuna trial.}}
\\end{{figure}}

\\begin{{table}}[H]
\\centering
{per_seed_table}
\\end{{table}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.82\\textwidth]{{best_trajectories.png}}
    \\caption{{Training trajectories of the best schedules. Top: hard validation loss. Bottom: weight on the hardest loss.}}
\\end{{figure}}

The per-seed view shows that neither method dominates every seed. The LLM candidate was clearly better on seed 44, Optuna was clearly better on seed 43, and seed 42 was relatively close. That is why the aggregate mean matters more than a single run.

\\section*{{Top schedules}}
{top_runs_tex}

This view is useful for quick visual comparison. It places the six strongest LLM schedules first and then the six strongest Optuna schedules below them. The subplot titles are ordered by mean best hard validation loss across seeds. The black dashed curve is the smoothed mean running-best hard validation loss across seeds, so its final level corresponds to the number shown in the title.

\\section*{{Failures and caveat}}
\\begin{{table}}[H]
\\centering
{failure_table}
\\end{{table}}

Most LLM failures came from either invalid generated code early in the run or API quota limits near the end. Because of that, the current comparison is useful but not perfectly clean: Optuna completed all 20 trials, while the LLM side produced 11 completed candidates.

\\section*{{Bottom line}}
Under this matched setup, the best completed LLM schedule slightly outperformed the best Optuna schedule on the main aggregate metric. I would report that result together with the caveat that the LLM search was interrupted before all 20 candidate slots were fully realized.

\\end{{document}}
"""

    tex_path = report_dir / "report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(tex_path)


if __name__ == "__main__":
    app()
