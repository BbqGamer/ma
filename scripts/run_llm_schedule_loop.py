#!/usr/bin/env python3
"""Run an outer LLM loop that proposes and evaluates schedule policies."""

from __future__ import annotations

from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any
from urllib import error, request

import polars as pl
import typer

from ma_thesis.config import REPORTS_DIR
from ma_thesis.schedule_api import ScheduleContext, ScheduleInitContext

ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = ROOT / ".venv" / "bin" / "python"
BENCHMARK_SCRIPT = ROOT / "scripts" / "run_policy_benchmark.py"
LLM_SCHEDULES_DIR = ROOT / "llm_schedules"
GENERATED_DIR = LLM_SCHEDULES_DIR / "generated"

app = typer.Typer(add_completion=False)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return value or "study"


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _code_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _next_candidate_id(history: list[dict[str, Any]]) -> str:
    if not history:
        return "0001"
    nums = [int(str(row["candidate_id"])) for row in history if "candidate_id" in row]
    return f"{(max(nums) + 1) if nums else 1:04d}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _read_code(path: Path, *, max_lines: int) -> str:
    if not path.exists():
        return "# missing candidate file"
    lines = path.read_text(encoding="utf-8").splitlines()
    trimmed = lines[:max_lines]
    if len(lines) > max_lines:
        trimmed.append("# ... truncated ...")
    return "\n".join(trimmed)


def _trajectory_summary(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    traj_path = Path(path)
    if not traj_path.exists():
        return None
    try:
        df = pl.read_parquet(traj_path)
    except Exception:
        return None
    if df.height == 0:
        return None
    summary: dict[str, Any] = {}
    hard_col = "val_hard_loss"
    if hard_col in df.columns:
        hard_values = df[hard_col].to_list()
        summary["start_hard_val"] = float(hard_values[0])
        summary["final_hard_val"] = float(hard_values[-1])
        summary["best_hard_val"] = float(min(hard_values))
        best_idx = min(range(len(hard_values)), key=lambda i: hard_values[i])
        summary["best_epoch"] = int(df["epoch"].to_list()[best_idx])
    weight_cols = sorted([c for c in df.columns if c.startswith("weight_")])
    if weight_cols:
        final_row = df.tail(1).to_dicts()[0]
        summary["final_weights"] = {k: round(float(final_row[k]), 4) for k in weight_cols}
    return summary


def _elite_context(
    history: list[dict[str, Any]],
    *,
    limit: int,
    max_code_lines: int,
) -> list[dict[str, Any]]:
    scored = [
        row for row in history if row.get("status") == "ok" and row.get("mean_best_hard_val_loss") is not None
    ]
    scored.sort(key=lambda row: float(row["mean_best_hard_val_loss"]))
    elites: list[dict[str, Any]] = []
    for row in scored[:limit]:
        candidate_path = Path(str(row["candidate_path"]))
        benchmark_id = str(row.get("benchmark_id", ""))
        per_seed_path = REPORTS_DIR / "benchmarks" / "policy" / benchmark_id / "per_seed_results.jsonl"
        per_seed_rows = _read_jsonl(per_seed_path)
        per_seed_summary = []
        for seed_row in per_seed_rows:
            if seed_row.get("status") != "ok":
                continue
            per_seed_summary.append(
                {
                    "seed": seed_row.get("seed"),
                    "best_hard_val_loss": seed_row.get("best_hard_val_loss"),
                    "final_hard_val_loss": seed_row.get("final_hard_val_loss"),
                    "epochs_trained": seed_row.get("epochs_trained"),
                    "trajectory": _trajectory_summary(seed_row.get("trajectory_path")),
                }
            )
        elites.append(
            {
                "candidate_id": row.get("candidate_id"),
                "llm_note": row.get("llm_note", ""),
                "mean_best_hard_val_loss": row.get("mean_best_hard_val_loss"),
                "std_best_hard_val_loss": row.get("std_best_hard_val_loss"),
                "mean_final_hard_val_loss": row.get("mean_final_hard_val_loss"),
                "num_successful_runs": row.get("num_successful_runs"),
                "best_seed": row.get("best_seed"),
                "code": _read_code(candidate_path, max_lines=max_code_lines),
                "per_seed": per_seed_summary,
            }
        )
    return elites


def _failure_context(
    history: list[dict[str, Any]],
    *,
    limit: int,
    max_code_lines: int,
) -> list[dict[str, Any]]:
    failed = [row for row in history if row.get("status") != "ok"]
    failed = failed[-limit:]
    rows: list[dict[str, Any]] = []
    for row in failed:
        candidate_path = Path(str(row.get("candidate_path", "")))
        rows.append(
            {
                "candidate_id": row.get("candidate_id"),
                "llm_note": row.get("llm_note", ""),
                "error": row.get("error", ""),
                "code": _read_code(candidate_path, max_lines=max_code_lines)
                if candidate_path.exists()
                else "# missing candidate file",
            }
        )
    return rows


def _exploration_directive(candidate_id: str) -> str:
    families = [
        "Stage-based switch: explicitly define early/mid/late phases with different hard-loss emphasis.",
        "Plateau trigger: detect stalled hardest loss and sharply reallocate weight toward hard sigma.",
        "Rank-based schedule: use sigma rank/difficulty order instead of only raw magnitudes.",
        "Pairwise-gap schedule: use hard-vs-easy and hard-vs-medium gaps to drive weight moves.",
        "Stateful momentum schedule: keep short internal memory of whether prior reallocations helped.",
        "Sparse-focus schedule: at some stages concentrate strongly on 1-2 sigmas instead of smoothing all equally.",
        "Recovery schedule: if hard loss worsens, temporarily return mass to easier/medium losses before refocusing.",
        "Monotone hardening is NOT required; allow reversals if they are justified by stalled progress.",
    ]
    idx = max(0, int(candidate_id) - 1) % len(families)
    return families[idx]


def _build_prompt(
    *,
    study_name: str,
    function: str,
    num_losses: int,
    history_window: int,
    eval_seeds: str,
    elites: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    candidate_module: str,
    candidate_path: Path,
    candidate_id: str,
) -> str:
    elite_json = json.dumps(elites, indent=2, sort_keys=True)
    failure_json = json.dumps(failures, indent=2, sort_keys=True)
    exploration_directive = _exploration_directive(candidate_id)
    return f"""You are designing a Python schedule policy for multi-loss curriculum learning.

Task:
- Improve mean best hard validation loss across seeds: {eval_seeds}
- Function: {function}
- Number of sigma losses: {num_losses}
- History window available in context: {history_window}
- Write only valid Python code to: {candidate_path}
- Module import path must remain: {candidate_module}
- Important: learn from MULTIPLE good candidates, not only the single best one.
- Prefer a thoughtful mutation or recombination of the elite candidates below.
- This iteration's exploration directive: {exploration_directive}

Required API:
```python
from ma_thesis.schedule_api import ScheduleContext, ScheduleInitContext

class Policy:
    def reset(self, ctx: ScheduleInitContext) -> None:
        ...

    def get_weights(self, ctx: ScheduleContext) -> list[float]:
        ...

policy = Policy()
```

Available context fields:
- ctx.epoch, ctx.total_epochs
- ctx.sigma_cols
- ctx.sigma_indices  # increasing difficulty order
- ctx.hard_index     # index of hardest loss
- ctx.current_train_losses, ctx.current_val_losses
- ctx.prev_train_losses, ctx.prev_val_losses
- ctx.ema_train_losses, ctx.ema_val_losses
- ctx.best_val_losses
- ctx.prev_weights
- ctx.best_hard_val_loss
- ctx.recent_train_losses, ctx.recent_val_losses

Rules:
- Return one non-negative weight per sigma.
- Keep the code simple and deterministic.
- Avoid imports beyond the standard library unless really necessary.
- Be defensive: context fields may be None early in training.
- Use sigma order explicitly; later sigma indices are harder.
- The schedule should not stay nearly uniform when losses/trends differ substantially.
- Do not modify training code; only define the policy module.
- Do not just copy the current best candidate unchanged. Make a clear, testable improvement.
- Avoid boring near-uniform schedules unless you can justify them with a very specific mechanism.
- You may use sharper, more asymmetric schedules than before if the hard loss is much worse.

Elite candidates with metrics, per-seed behavior, trajectory summaries, and full code:
```json
{elite_json}
```

Recent failed candidates with code and errors to avoid:
```json
{failure_json}
```

Instructions for this iteration:
1. Identify 1-2 mechanisms that seem common among strong candidates.
2. Identify 1 weakness or instability pattern from the elites or failures.
3. Propose a new schedule that combines strong mechanisms while addressing that weakness.
4. Also make it meaningfully different from the current elite family; do not just tweak constants.
5. Keep the implementation robust to tuples, nested history, and None values.
6. Make weights measurably non-uniform when the harder losses are clearly worse or more stalled.
7. Put a concrete multi-line strategy comment at the top of the file using this template:
   # Hypothesis: ...
   # Mechanism: ...
   # Expected effect: ...

Study name: {study_name}
Candidate id: {candidate_id}
Use a different idea family if the recent elites are too similar.
Output only the Python module.
"""


def _run_proposer(command_template: str, *, placeholders: dict[str, str], candidate_path: Path) -> str:
    command = command_template.format(**placeholders)
    result = subprocess.run(
        command,
        shell=True,
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "LLM proposer command failed with return code "
            f"{result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
        )
    stdout = result.stdout.strip()
    if (not candidate_path.exists() or not candidate_path.read_text(encoding="utf-8").strip()) and stdout:
        candidate_path.write_text(stdout + "\n", encoding="utf-8")
    return stdout


def _extract_python_code(text: str) -> str:
    matches = re.findall(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if not matches:
        matches = re.findall(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if matches:
        return matches[0].strip() + "\n"
    return text.strip() + "\n"


def _response_text(response_json: dict[str, Any]) -> str:
    text = str(response_json.get("output_text", "") or "")
    if text.strip():
        return text

    output = response_json.get("output", [])
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for chunk in content:
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("type") == "output_text" and chunk.get("text"):
                    parts.append(str(chunk["text"]))
        if parts:
            return "\n".join(parts)
    return ""


def _usage_cost_summary(
    response_json: dict[str, Any],
    *,
    input_cost_per_1m: float,
    output_cost_per_1m: float,
) -> dict[str, float]:
    usage = response_json.get("usage", {}) if isinstance(response_json, dict) else {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    estimated_cost_usd = (
        input_tokens * input_cost_per_1m + output_tokens * output_cost_per_1m
    ) / 1_000_000.0
    return {
        "input_tokens": float(input_tokens),
        "output_tokens": float(output_tokens),
        "total_tokens": float(total_tokens),
        "estimated_cost_usd": float(estimated_cost_usd),
    }


def _openai_generate_candidate(
    *,
    prompt: str,
    candidate_path: Path,
    model: str,
    api_key: str,
    base_url: str,
    input_cost_per_1m: float,
    output_cost_per_1m: float,
    max_retries: int = 3,
) -> tuple[str, dict[str, float]]:
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            }
        ],
        "temperature": 1.15,
    }
    body = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(max_retries):
        req = request.Request(
            url=base_url.rstrip("/") + "/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=180) as resp:
                raw = resp.read().decode("utf-8")
            response_json = json.loads(raw)
            text = _response_text(response_json)
            if not text:
                raise RuntimeError(f"OpenAI response missing output text: {raw}")
            candidate_code = _extract_python_code(text)
            candidate_path.write_text(candidate_code, encoding="utf-8")
            return raw, _usage_cost_summary(
                response_json,
                input_cost_per_1m=input_cost_per_1m,
                output_cost_per_1m=output_cost_per_1m,
            )
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"OpenAI API error {exc.code}: {detail}")
            if exc.code not in {429, 500, 502, 503, 504} or attempt == max_retries - 1:
                raise last_error from exc
        except error.URLError as exc:
            last_error = RuntimeError(f"OpenAI API request failed: {exc}")
            if attempt == max_retries - 1:
                raise last_error from exc
        except Exception as exc:
            last_error = exc
            if attempt == max_retries - 1:
                raise
        time.sleep(2.0 * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError("OpenAI generation failed without a captured error.")


def _write_token_summary(study_dir: Path, history: list[dict[str, Any]]) -> None:
    token_rows = [row for row in history if row.get("generator") == "openai"]
    summary = {
        "num_openai_candidates": len(token_rows),
        "input_tokens": int(sum(float(row.get("openai_input_tokens", 0) or 0) for row in token_rows)),
        "output_tokens": int(sum(float(row.get("openai_output_tokens", 0) or 0) for row in token_rows)),
        "total_tokens": int(sum(float(row.get("openai_total_tokens", 0) or 0) for row in token_rows)),
        "estimated_cost_usd": float(sum(float(row.get("openai_estimated_cost_usd", 0.0) or 0.0) for row in token_rows)),
    }
    (study_dir / "token_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _candidate_note(candidate_path: Path) -> str:
    lines = candidate_path.read_text(encoding="utf-8").splitlines()
    comments: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            comments.append(stripped)
            if len(comments) >= 3:
                break
            continue
        if stripped:
            break
    return " ".join(comments)[:400]


def _preflight_candidate(candidate_module: str) -> None:
    module = importlib.import_module(candidate_module)
    policy = getattr(module, "policy", None)
    if policy is None or not hasattr(policy, "get_weights"):
        raise RuntimeError("Generated module must define `policy` with get_weights().")
    sigma_cols = ("y_sigma_0", "y_sigma_1", "y_sigma_2", "y_sigma_3")
    sigma_indices = (0, 1, 2, 3)
    init_ctx = ScheduleInitContext(
        sigma_cols=sigma_cols,
        sigma_indices=sigma_indices,
        hard_index=len(sigma_cols) - 1,
        total_epochs=10,
        history_window=5,
        seed=42,
        run_name="preflight",
    )
    if hasattr(policy, "reset"):
        policy.reset(init_ctx)

    scenarios = [
        {
            "name": "difficulty_spread",
            "current": (10.0, 20.0, 40.0, 80.0),
            "prev": (12.0, 21.0, 38.0, 78.0),
            "ema": (11.0, 20.0, 39.0, 79.0),
            "best": (9.0, 18.0, 35.0, 70.0),
        },
        {
            "name": "hard_stalled",
            "current": (10.0, 18.0, 35.0, 90.0),
            "prev": (9.0, 16.0, 31.0, 89.0),
            "ema": (9.5, 17.0, 33.0, 89.5),
            "best": (8.0, 14.0, 28.0, 75.0),
        },
        {
            "name": "easy_stalled",
            "current": (25.0, 18.0, 20.0, 30.0),
            "prev": (24.0, 16.0, 18.0, 29.0),
            "ema": (24.5, 17.0, 19.0, 29.5),
            "best": (20.0, 14.0, 16.0, 24.0),
        },
    ]
    max_weight_range = 0.0
    for scenario in scenarios:
        current = scenario["current"]
        prev = scenario["prev"]
        ema = scenario["ema"]
        best = scenario["best"]
        ctx = ScheduleContext(
            epoch=6,
            total_epochs=10,
            sigma_cols=sigma_cols,
            sigma_indices=sigma_indices,
            hard_index=len(sigma_cols) - 1,
            current_train_losses=current,
            current_val_losses=current,
            prev_train_losses=prev,
            prev_val_losses=prev,
            ema_train_losses=ema,
            ema_val_losses=ema,
            best_val_losses=best,
            prev_weights=tuple(1.0 / len(sigma_cols) for _ in sigma_cols),
            best_hard_val_loss=current[-1],
            recent_train_losses=(prev, current),
            recent_val_losses=(prev, current),
        )
        weights = list(policy.get_weights(ctx))
        if len(weights) != len(sigma_cols):
            raise RuntimeError("Generated policy returned the wrong number of weights in preflight.")
        max_weight_range = max(max_weight_range, max(weights) - min(weights))
    if max_weight_range < 0.02:
        raise RuntimeError(
            "Generated policy is effectively uniform across informative preflight scenarios. "
            "Make weight allocation respond more strongly to difficulty/trend differences."
        )


def _build_benchmark_command(
    *,
    function: str,
    benchmark_id: str,
    experiment_name: str,
    input_path: str | None,
    regenerate_data: bool,
    num_samples: int,
    num_sigmas: int,
    sigma_scale: float,
    train_samples: int | None,
    noise_ratio: float,
    data_seed: int,
    eval_seeds: str,
    schedule_module: str,
    num_losses: int,
    history_window: int,
    ema_alpha: float,
    model_arch: str,
    hidden_dim: int,
    num_blocks: int,
    activation: str,
    num_layers: int,
    omega_0: float,
    num_fourier: int,
    fourier_sigma: float,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    min_delta: float,
    min_train_per_param: float,
    candidate_id: str,
    llm_note: str,
) -> list[str]:
    cmd = [
        str(PYTHON_BIN),
        str(BENCHMARK_SCRIPT),
        "--function",
        function,
        "--benchmark-id",
        benchmark_id,
        "--experiment-name",
        experiment_name,
        "--eval-seeds",
        eval_seeds,
        "--schedule-module",
        schedule_module,
        "--num-losses",
        str(num_losses),
        "--history-window",
        str(history_window),
        "--ema-alpha",
        str(ema_alpha),
        "--num-samples",
        str(num_samples),
        "--num-sigmas",
        str(num_sigmas),
        "--sigma-scale",
        str(sigma_scale),
        "--noise-ratio",
        str(noise_ratio),
        "--data-seed",
        str(data_seed),
        "--model-arch",
        model_arch,
        "--hidden-dim",
        str(hidden_dim),
        "--num-blocks",
        str(num_blocks),
        "--activation",
        activation,
        "--num-layers",
        str(num_layers),
        "--omega-0",
        str(omega_0),
        "--num-fourier",
        str(num_fourier),
        "--fourier-sigma",
        str(fourier_sigma),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--patience",
        str(patience),
        "--min-delta",
        str(min_delta),
        "--min-train-per-param",
        str(min_train_per_param),
        "--candidate-id",
        candidate_id,
        "--llm-note",
        llm_note,
    ]
    if input_path is not None:
        cmd.extend(["--input-path", input_path])
    if regenerate_data:
        cmd.append("--regenerate-data")
    if train_samples is not None:
        cmd.extend(["--train-samples", str(train_samples)])
    return cmd


@app.command()
def main(
    study_name: str = typer.Option("llm_schedule_search"),
    iterations: int = typer.Option(5, help="Number of propose/evaluate iterations."),
    proposer_command_template: str | None = typer.Option(
        None,
        help=(
            "Optional shell command template used to generate candidate code. "
            "Available placeholders: {prompt_path}, {candidate_path}, {candidate_module}, "
            "{history_path}, {study_dir}, {candidate_id}. If omitted, OpenAI API is used."
        ),
    ),
    openai_model: str = typer.Option("gpt-5.4-mini", help="OpenAI model used for generation."),
    openai_api_key_env: str = typer.Option(
        "OPENAI_API_KEY",
        help="Environment variable holding the OpenAI API key.",
    ),
    openai_base_url: str = typer.Option(
        "https://api.openai.com/v1",
        help="OpenAI API base URL.",
    ),
    openai_input_cost_per_1m: float = typer.Option(
        0.0,
        help="Estimated USD cost per 1M input tokens for the selected OpenAI model.",
    ),
    openai_output_cost_per_1m: float = typer.Option(
        0.0,
        help="Estimated USD cost per 1M output tokens for the selected OpenAI model.",
    ),
    function: str = "ackley",
    experiment_name: str = "llm-policy-search",
    input_path: str | None = None,
    regenerate_data: bool = False,
    num_samples: int = 20000,
    num_sigmas: int = 3,
    sigma_scale: float = 5.0,
    train_samples: int | None = None,
    noise_ratio: float = 0.02,
    data_seed: int = 42,
    eval_seeds: str = "42,43,44,45,46",
    num_losses: int = 4,
    history_window: int = 5,
    ema_alpha: float = 0.3,
    model_arch: str = "fourier",
    hidden_dim: int = 16,
    num_blocks: int = 4,
    activation: str = "silu",
    num_layers: int = 4,
    omega_0: float = 30.0,
    num_fourier: int = 128,
    fourier_sigma: float = 10.0,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 30,
    min_delta: float = 1e-5,
    min_train_per_param: float = 10.0,
    top_k_history: int = 3,
    top_k_failures: int = 5,
    max_candidate_code_lines: int = 200,
    manual_stop_after_prompt: bool = typer.Option(
        False,
        help="Write the prompt and candidate path, then stop without calling the proposer.",
    ),
) -> None:
    if not PYTHON_BIN.exists():
        raise typer.BadParameter(f"Missing Python interpreter: {PYTHON_BIN}")
    if not BENCHMARK_SCRIPT.exists():
        raise typer.BadParameter(f"Missing benchmark script: {BENCHMARK_SCRIPT}")
    if not manual_stop_after_prompt and proposer_command_template is None:
        if not os.getenv(openai_api_key_env):
            raise typer.BadParameter(
                f"Environment variable {openai_api_key_env} is required for OpenAI generation."
            )

    study_slug = _slug(study_name)
    study_dir = REPORTS_DIR / "llm_schedule_search" / study_slug
    prompts_dir = study_dir / "prompts"
    results_dir = study_dir / "results"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    package_dir = GENERATED_DIR / study_slug
    package_dir.mkdir(parents=True, exist_ok=True)
    (LLM_SCHEDULES_DIR / "generated" / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    history_path = study_dir / "history.jsonl"

    for _ in range(iterations):
        history = _read_history(history_path)
        candidate_id = _next_candidate_id(history)
        candidate_name = f"candidate_{candidate_id}"
        candidate_path = package_dir / f"{candidate_name}.py"
        candidate_module = f"llm_schedules.generated.{study_slug}.{candidate_name}"
        prompt_path = prompts_dir / f"{candidate_name}.md"
        proposer_stdout_path = prompts_dir / f"{candidate_name}_stdout.txt"
        llm_note = ""
        benchmark_id = ""
        openai_usage: dict[str, float] = {}

        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "candidate_module": candidate_module,
            "candidate_path": str(candidate_path),
            "prompt_path": str(prompt_path),
            "generator": "external_command" if proposer_command_template is not None else "openai",
            "openai_model": None if proposer_command_template is not None else openai_model,
        }

        try:
            elites = _elite_context(
                history,
                limit=top_k_history,
                max_code_lines=max_candidate_code_lines,
            )
            failures = _failure_context(
                history,
                limit=top_k_failures,
                max_code_lines=max_candidate_code_lines,
            )
            prompt = _build_prompt(
                study_name=study_name,
                function=function,
                num_losses=num_losses,
                history_window=history_window,
                eval_seeds=eval_seeds,
                elites=elites,
                failures=failures,
                candidate_module=candidate_module,
                candidate_path=candidate_path,
                candidate_id=candidate_id,
            )
            prompt_path.write_text(prompt, encoding="utf-8")
            prompt_context_path = prompts_dir / f"{candidate_name}_context.json"
            prompt_context_path.write_text(
                json.dumps({"elites": elites, "failures": failures}, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            if manual_stop_after_prompt:
                print(f"Prompt written to {prompt_path}")
                print(
                    f"Write candidate code to {candidate_path} and rerun without --manual-stop-after-prompt"
                )
                return

            proposer_stdout = ""
            if proposer_command_template is not None:
                placeholders = {
                    "prompt_path": shlex.quote(str(prompt_path)),
                    "candidate_path": shlex.quote(str(candidate_path)),
                    "candidate_module": candidate_module,
                    "history_path": shlex.quote(str(history_path)),
                    "study_dir": shlex.quote(str(study_dir)),
                    "candidate_id": candidate_id,
                }
                proposer_stdout = _run_proposer(
                    proposer_command_template,
                    placeholders=placeholders,
                    candidate_path=candidate_path,
                )
                proposer_stdout_path.write_text(proposer_stdout, encoding="utf-8")
            else:
                proposer_stdout, openai_usage = _openai_generate_candidate(
                    prompt=prompt,
                    candidate_path=candidate_path,
                    model=openai_model,
                    api_key=os.environ[openai_api_key_env],
                    base_url=openai_base_url,
                    input_cost_per_1m=openai_input_cost_per_1m,
                    output_cost_per_1m=openai_output_cost_per_1m,
                )
                proposer_stdout_path.write_text(proposer_stdout, encoding="utf-8")

            if not candidate_path.exists():
                raise RuntimeError(f"Proposer did not create candidate file: {candidate_path}")

            _preflight_candidate(candidate_module)
            llm_note = _candidate_note(candidate_path)
            benchmark_id = f"{study_slug}_{candidate_name}_{_timestamp()}"
            benchmark_cmd = _build_benchmark_command(
                function=function,
                benchmark_id=benchmark_id,
                experiment_name=experiment_name,
                input_path=input_path,
                regenerate_data=regenerate_data,
                num_samples=num_samples,
                num_sigmas=num_sigmas,
                sigma_scale=sigma_scale,
                train_samples=train_samples,
                noise_ratio=noise_ratio,
                data_seed=data_seed,
                eval_seeds=eval_seeds,
                schedule_module=candidate_module,
                num_losses=num_losses,
                history_window=history_window,
                ema_alpha=ema_alpha,
                model_arch=model_arch,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                activation=activation,
                num_layers=num_layers,
                omega_0=omega_0,
                num_fourier=num_fourier,
                fourier_sigma=fourier_sigma,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
                min_delta=min_delta,
                min_train_per_param=min_train_per_param,
                candidate_id=candidate_id,
                llm_note=llm_note,
            )
            result = subprocess.run(
                benchmark_cmd,
                check=False,
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
            benchmark_stdout_path = results_dir / f"{candidate_name}_benchmark_stdout.txt"
            benchmark_stderr_path = results_dir / f"{candidate_name}_benchmark_stderr.txt"
            benchmark_stdout_path.write_text(result.stdout, encoding="utf-8")
            benchmark_stderr_path.write_text(result.stderr, encoding="utf-8")

            row.update(
                {
                    "candidate_sha256": _code_hash(candidate_path),
                    "benchmark_id": benchmark_id,
                    "benchmark_command": benchmark_cmd,
                    "llm_note": llm_note,
                    "openai_input_tokens": openai_usage.get("input_tokens", 0.0),
                    "openai_output_tokens": openai_usage.get("output_tokens", 0.0),
                    "openai_total_tokens": openai_usage.get("total_tokens", 0.0),
                    "openai_estimated_cost_usd": openai_usage.get("estimated_cost_usd", 0.0),
                }
            )
            aggregate_path = REPORTS_DIR / "benchmarks" / "policy" / benchmark_id / "aggregate.json"
            if result.returncode == 0 and aggregate_path.exists():
                aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
                row.update(aggregate)
                row["status"] = str(aggregate.get("status", "ok"))
            else:
                row["status"] = "failed"
                row["error"] = result.stderr.strip() or result.stdout.strip() or "benchmark failed"
        except Exception as exc:
            row.update(
                {
                    "candidate_sha256": _code_hash(candidate_path) if candidate_path.exists() else None,
                    "benchmark_id": benchmark_id,
                    "llm_note": llm_note,
                    "openai_input_tokens": openai_usage.get("input_tokens", 0.0),
                    "openai_output_tokens": openai_usage.get("output_tokens", 0.0),
                    "openai_total_tokens": openai_usage.get("total_tokens", 0.0),
                    "openai_estimated_cost_usd": openai_usage.get("estimated_cost_usd", 0.0),
                    "status": "failed",
                    "error": repr(exc),
                }
            )

        _append_history(history_path, row)
        _write_token_summary(study_dir, _read_history(history_path))
        print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
