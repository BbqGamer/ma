#!/usr/bin/env python3
"""Run a simple LLM-guided search for CIFAR-100 easy-vs-hard schedules."""

from __future__ import annotations

from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any
from urllib import error, request

import typer

from ma_thesis.cifar100_schedule import TaskScheduleContext, TaskScheduleInitContext
from ma_thesis.config import REPORTS_DIR

ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = ROOT / ".venv" / "bin" / "python"
BENCHMARK_SCRIPT = ROOT / "scripts" / "run_cifar100_policy_benchmark.py"
GENERATED_ROOT = ROOT / "llm_schedules" / "generated"
SEARCH_ROOT = REPORTS_DIR / "cifar100_easy_hard" / "llm_search"

app = typer.Typer(add_completion=False)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _next_candidate_id(history: list[dict[str, Any]]) -> str:
    nums = [int(str(row["candidate_id"])) for row in history if row.get("candidate_id")]
    return f"{(max(nums) + 1) if nums else 1:04d}"


def _response_text(response_json: dict[str, Any]) -> str:
    if isinstance(response_json.get("output_text"), str) and response_json["output_text"].strip():
        return str(response_json["output_text"])
    output = response_json.get("output", [])
    chunks: list[str] = []
    for item in output:
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text)
    return "\n".join(chunks).strip()


def _extract_python_code(text: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return fenced[-1].strip() + "\n"
    return text.strip() + "\n"


def _usage_cost_summary(response_json: dict[str, Any]) -> dict[str, float]:
    usage = response_json.get("usage", {}) if isinstance(response_json, dict) else {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    return {
        "input_tokens": float(input_tokens),
        "output_tokens": float(output_tokens),
        "total_tokens": float(total_tokens),
    }


def _openai_generate_candidate(
    *,
    prompt: str,
    candidate_path: Path,
    model: str,
    api_key: str,
    base_url: str,
    max_retries: int = 3,
) -> tuple[str, dict[str, float]]:
    url = base_url.rstrip("/") + "/responses"
    payload = {
        "model": model,
        "input": prompt,
        "reasoning": {"effort": "medium"},
        "text": {"verbosity": "medium"},
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        req = request.Request(url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=180) as response:
                raw = response.read().decode("utf-8")
            response_json = json.loads(raw)
            text = _response_text(response_json)
            if not text:
                raise RuntimeError(f"OpenAI response missing output text: {raw}")
            candidate_code = _extract_python_code(text)
            candidate_path.write_text(candidate_code, encoding="utf-8")
            return raw, _usage_cost_summary(response_json)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            should_retry = exc.code in {408, 409, 429, 500, 502, 503, 504}
            last_error = RuntimeError(f"OpenAI API error {exc.code}: {body}")
            if not should_retry or attempt == max_retries:
                break
            time.sleep(2**attempt)
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(2**attempt)

    raise RuntimeError(str(last_error) if last_error else "OpenAI generation failed.")


def _trajectory_summary(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    traj_path = Path(path)
    if not traj_path.exists():
        return None
    rows = [line.strip() for line in traj_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 2:
        return None
    import csv
    from io import StringIO

    reader = csv.DictReader(StringIO("\n".join(rows)))
    data = list(reader)
    if not data:
        return None
    hard_vals = [float(row["val_hard_loss"]) for row in data]
    weights_hard = [float(row["weight_hard"]) for row in data]
    epochs = [int(float(row["epoch"])) for row in data]
    best_idx = min(range(len(hard_vals)), key=lambda i: hard_vals[i])
    return {
        "start_val_hard_loss": hard_vals[0],
        "best_val_hard_loss": hard_vals[best_idx],
        "best_epoch": epochs[best_idx],
        "final_val_hard_loss": hard_vals[-1],
        "final_hard_weight": weights_hard[-1],
    }


def _elite_context(history: list[dict[str, Any]], *, limit: int = 3) -> list[dict[str, Any]]:
    ok = [row for row in history if row.get("status") == "ok" and row.get("mean_best_hard_val_loss") is not None]
    ok.sort(key=lambda row: float(row["mean_best_hard_val_loss"]))
    elites: list[dict[str, Any]] = []
    for row in ok[:limit]:
        candidate_path = Path(str(row["candidate_path"]))
        code = candidate_path.read_text(encoding="utf-8") if candidate_path.exists() else "# missing"
        benchmark_id = row.get("benchmark_id", "")
        per_seed_path = REPORTS_DIR / "cifar100_easy_hard" / "policy" / str(benchmark_id) / "per_seed_results.jsonl"
        per_seed = []
        if per_seed_path.exists():
            for line in per_seed_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                seed_row = json.loads(line)
                per_seed.append(
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
                "candidate_id": row["candidate_id"],
                "mean_best_hard_val_loss": row.get("mean_best_hard_val_loss"),
                "std_best_hard_val_loss": row.get("std_best_hard_val_loss"),
                "llm_note": row.get("llm_note", ""),
                "code": code,
                "per_seed": per_seed,
            }
        )
    return elites


def _failure_context(history: list[dict[str, Any]], *, limit: int = 3) -> list[dict[str, Any]]:
    failed = [row for row in history if row.get("status") != "ok"]
    rows: list[dict[str, Any]] = []
    for row in failed[-limit:]:
        candidate_path = Path(str(row.get("candidate_path", "")))
        rows.append(
            {
                "candidate_id": row.get("candidate_id"),
                "error": row.get("error", ""),
                "code": candidate_path.read_text(encoding="utf-8") if candidate_path.exists() else "# missing",
            }
        )
    return rows


def _build_prompt(
    *,
    study_name: str,
    candidate_id: str,
    elites: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    candidate_path: Path,
) -> str:
    return f"""You are designing a Python policy module for a CIFAR-100 easy-vs-hard curriculum.

Task setup:
- Easy task: CIFAR-100 coarse superclass prediction (20 classes)
- Hard task: CIFAR-100 fine label prediction (100 classes)
- Goal: minimize mean best hard validation loss across seeds
- The policy outputs two weights: [easy_weight, hard_weight]
- Write only valid Python code to: {candidate_path}

Required API:
```python
from ma_thesis.cifar100_schedule import TaskScheduleContext, TaskScheduleInitContext

class Policy:
    def reset(self, ctx: TaskScheduleInitContext) -> None:
        ...

    def get_weights(self, ctx: TaskScheduleContext) -> list[float]:
        ...

policy = Policy()
```

Available context fields:
- ctx.epoch, ctx.total_epochs
- ctx.task_names
- ctx.easy_index, ctx.hard_index
- ctx.current_train_losses, ctx.current_val_losses
- ctx.prev_train_losses, ctx.prev_val_losses
- ctx.ema_train_losses, ctx.ema_val_losses
- ctx.best_val_losses
- ctx.prev_weights
- ctx.best_hard_val_loss
- ctx.recent_train_losses, ctx.recent_val_losses

Design goals:
- Keep the code simple and deterministic.
- Use the hard task explicitly; do not stay close to 50/50 if the hard loss is clearly lagging.
- It is okay to start easier and then shift toward the hard task.
- Be robust to None values in early epochs.
- Learn from multiple good candidates, not only the best one.

Elite candidates:
```json
{json.dumps(elites, indent=2, sort_keys=True)}
```

Recent failures:
```json
{json.dumps(failures, indent=2, sort_keys=True)}
```

Put a short top comment at the beginning of the file using:
# Hypothesis: ...
# Mechanism: ...
# Expected effect: ...

Study name: {study_name}
Candidate id: {candidate_id}
Output only the Python module.
"""


def _preflight_candidate(module_path: str) -> None:
    module = importlib.import_module(module_path)
    importlib.reload(module)
    policy = getattr(module, "policy")
    init_ctx = TaskScheduleInitContext(
        task_names=("easy_coarse", "hard_fine"),
        easy_index=0,
        hard_index=1,
        total_epochs=30,
        history_window=5,
        seed=42,
        run_name="preflight",
    )
    policy.reset(init_ctx)
    contexts = [
        TaskScheduleContext(
            epoch=0,
            total_epochs=30,
            task_names=("easy_coarse", "hard_fine"),
            easy_index=0,
            hard_index=1,
            current_train_losses=None,
            current_val_losses=None,
            prev_train_losses=None,
            prev_val_losses=None,
            ema_train_losses=None,
            ema_val_losses=None,
            best_val_losses=None,
            prev_weights=None,
            best_hard_val_loss=float("inf"),
            recent_train_losses=(),
            recent_val_losses=(),
        ),
        TaskScheduleContext(
            epoch=12,
            total_epochs=30,
            task_names=("easy_coarse", "hard_fine"),
            easy_index=0,
            hard_index=1,
            current_train_losses=(1.5, 3.8),
            current_val_losses=(1.7, 4.0),
            prev_train_losses=(1.6, 3.9),
            prev_val_losses=(1.8, 4.1),
            ema_train_losses=(1.55, 3.85),
            ema_val_losses=(1.75, 4.05),
            best_val_losses=(1.65, 3.7),
            prev_weights=(0.45, 0.55),
            best_hard_val_loss=3.7,
            recent_train_losses=((1.7, 4.0), (1.6, 3.9), (1.5, 3.8)),
            recent_val_losses=((1.9, 4.3), (1.8, 4.1), (1.7, 4.0)),
        ),
    ]
    for ctx in contexts:
        weights = policy.get_weights(ctx)
        if len(weights) != 2:
            raise ValueError(f"Expected exactly 2 weights, got {weights}")
        vals = [float(v) for v in weights]
        if not all(v == v and abs(v) != float("inf") for v in vals):
            raise ValueError(f"Non-finite weights returned: {weights}")


@app.command()
def main(
    study_name: str = typer.Option(f"cifar100_llm_{_timestamp()}"),
    iterations: int = 10,
    eval_seeds: str = typer.Option("42"),
    batch_size: int = 256,
    num_workers: int = 8,
    max_epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    ema_alpha: float = 0.3,
    history_window: int = 5,
    val_fraction: float = 0.1,
    patience: int = 8,
    min_delta: float = 1e-4,
    use_early_stopping: bool = typer.Option(
        False,
        help="Enable early stopping. Leave off for fixed-budget fair search.",
    ),
    accelerator: str = "auto",
    devices: str = "1",
    precision: str = "16-mixed",
    openai_model: str = "gpt-5.4-mini",
    openai_api_key_env: str = "OPENAI_API_KEY",
    openai_base_url: str = "https://api.openai.com/v1",
) -> None:
    if openai_api_key_env not in os.environ:
        raise typer.BadParameter(f"Missing API key environment variable: {openai_api_key_env}")

    study_dir = SEARCH_ROOT / study_name
    prompts_dir = study_dir / "prompts"
    history_path = study_dir / "history.jsonl"
    module_dir = GENERATED_ROOT / study_name
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / "__init__.py").write_text("", encoding="utf-8")
    (GENERATED_ROOT / "__init__.py").write_text("", encoding="utf-8")

    for _ in range(iterations):
        history = _read_history(history_path)
        candidate_id = _next_candidate_id(history)
        module_name = f"candidate_{candidate_id}"
        candidate_path = module_dir / f"{module_name}.py"
        module_path = f"llm_schedules.generated.{study_name}.{module_name}"
        prompt_path = prompts_dir / f"candidate_{candidate_id}.md"
        response_path = prompts_dir / f"candidate_{candidate_id}_response.json"

        prompt = _build_prompt(
            study_name=study_name,
            candidate_id=candidate_id,
            elites=_elite_context(history),
            failures=_failure_context(history),
            candidate_path=candidate_path,
        )
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")

        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "candidate_path": str(candidate_path),
            "candidate_module": module_path,
            "prompt_path": str(prompt_path),
            "openai_model": openai_model,
            "status": "failed",
        }
        usage: dict[str, float] = {}
        try:
            raw, usage = _openai_generate_candidate(
                prompt=prompt,
                candidate_path=candidate_path,
                model=openai_model,
                api_key=os.environ[openai_api_key_env],
                base_url=openai_base_url,
            )
            response_path.write_text(raw, encoding="utf-8")
            importlib.invalidate_caches()
            _preflight_candidate(module_path)

            benchmark_id = f"{study_name}_{candidate_id}_{_timestamp()}"
            cmd = [
                str(PYTHON_BIN),
                str(BENCHMARK_SCRIPT),
                "--benchmark-id",
                benchmark_id,
                "--schedule-module",
                module_path,
                "--eval-seeds",
                eval_seeds,
                "--batch-size",
                str(batch_size),
                "--num-workers",
                str(num_workers),
                "--max-epochs",
                str(max_epochs),
                "--lr",
                str(lr),
                "--weight-decay",
                str(weight_decay),
                "--ema-alpha",
                str(ema_alpha),
                "--history-window",
                str(history_window),
                "--val-fraction",
                str(val_fraction),
                "--patience",
                str(patience),
                "--min-delta",
                str(min_delta),
                "--use-early-stopping" if use_early_stopping else "",
                "--accelerator",
                accelerator,
                "--devices",
                devices,
                "--precision",
                precision,
            ]
            cmd = [part for part in cmd if part != ""]
            result = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
            aggregate = json.loads(result.stdout)
            row.update(aggregate)
            row["benchmark_id"] = benchmark_id
            row["candidate_sha256"] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
            row["status"] = "ok"
        except Exception as exc:
            row["error"] = repr(exc)
        row["openai_input_tokens"] = usage.get("input_tokens", 0.0)
        row["openai_output_tokens"] = usage.get("output_tokens", 0.0)
        row["openai_total_tokens"] = usage.get("total_tokens", 0.0)
        _append_history(history_path, row)
        print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
