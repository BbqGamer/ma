#!/usr/bin/env python3
"""Build a single PDF from the CIFAR-100 all-runs markdown report."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

import typer

app = typer.Typer(add_completion=False)


def _inject_page_breaks(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    seen_first_run = False
    for line in lines:
        if line.startswith("### "):
            if seen_first_run:
                out.append("\\newpage")
                out.append("")
            seen_first_run = True
        elif line.startswith("## Optuna runs") or line.startswith("## Failed LLM attempts"):
            out.append("\\newpage")
            out.append("")
        out.append(line)
    return "\n".join(out) + "\n"


@app.command()
def main(
    report_dir: Path = typer.Option(
        Path("reports/analysis/cifar100_easy_hard/cifar100_v1_all_runs_report"),
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory containing summary.md and figure files.",
    ),
    output_name: str = typer.Option("report.pdf", help="Output PDF filename inside report_dir."),
) -> None:
    summary_path = report_dir / "summary.md"
    if not summary_path.exists():
        raise typer.BadParameter(f"Missing markdown report: {summary_path}")

    pdf_path = report_dir / output_name
    source = _inject_page_breaks(summary_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="cifar100_pdf_") as tmpdir:
        tmp_md = Path(tmpdir) / "report.md"
        tmp_md.write_text(source, encoding="utf-8")
        cmd = [
            "pandoc",
            str(tmp_md),
            "--from",
            "markdown+raw_tex",
            "--standalone",
            "--toc",
            "--pdf-engine=pdflatex",
            "-V",
            "geometry:margin=1in",
            "-V",
            "colorlinks=true",
            "--resource-path",
            str(report_dir),
            "-o",
            str(pdf_path),
        ]
        subprocess.run(cmd, check=True)

    print(pdf_path)


if __name__ == "__main__":
    app()
