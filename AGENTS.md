# AGENTS.md — Project Context For Coding Assistants

> Read this first. It describes how this repository is actually used today.

## Project Snapshot

- Project: Dynamic Multi-Loss Curriculum Learning with Meta-Optimization
- Context: Master's thesis implementation (active research code)
- Code language: English
- Thesis language: Polish

Goal in one line:
- Compare `single` vs `curriculum` (and optionally `meta`) training on Gaussian-continuation
  targets for benchmark regression surfaces, with reproducible sweeps and MLflow tracking.

## What Is Core vs Optional

Core training pipeline (high priority):
- `ma_thesis/experiment.py` (orchestrator CLI)
- `ma_thesis/dataset.py` (data generation + train/test split)
- `ma_thesis/train.py` (single/curriculum training)
- `ma_thesis/sweep.py` (Optuna model HP sweep)
- `ma_thesis/schedule_sweep.py` (Optuna curriculum schedule-only sweep)
- `ma_thesis/models.py` (model factory + architectures)
- `ma_thesis/common.py` (shared training helpers, plotting helpers, parameter budgeting)
- `ma_thesis/io.py` (shared dataset loading + train/val split)
- `ma_thesis/mlflow_utils.py` (shared MLflow artifact/config logging)
- `ma_thesis/training_core.py` (shared optimizer/scheduler and minibatch loop primitives)

Used but secondary (research/analysis support):
- `ma_thesis/meta_train.py` (experimental meta-curriculum / bi-level optimization)
- `ma_thesis/optimization_metrics.py` (diagnostic metrics)
- `scripts/run_curriculum_benchmark.py`
- `scripts/analyze_curriculum_benchmark.py`

Archival / frozen results (touch only if needed):
- `docs/results/2026-03-14_sweep-pack-v1.md` + related frozen result artifacts

## Current Stack

- Python 3.10
- PyTorch (main training stack)
- Optuna (hyperparameter search)
- MLflow (local SQLite backend + filesystem artifacts)
- Polars, NumPy, SciPy, scikit-learn
- Typer CLIs
- Ruff lint/format
- MkDocs + LaTeX for docs/thesis

Note:
- `jax` / `flax` may appear in dependencies, but the active training pipeline is PyTorch-based.

## Main Commands

Environment:
```bash
make create_environment
source .venv/bin/activate
make requirements
```

Data:
```bash
make data
```

Run experiments through single entrypoint:
```bash
python -m ma_thesis.experiment run --method single --function ackley --sigma-level -1
python -m ma_thesis.experiment run --method curriculum --function ackley
python -m ma_thesis.experiment run --method meta --function eggholder
python -m ma_thesis.experiment run --method sweep --function levy --n-trials 80
python -m ma_thesis.experiment run --method schedule_sweep --function ackley
```

Run from YAML:
```bash
python -m ma_thesis.experiment run-config configs/sweeps/v3_ratio/ackley_sweep_v3_ratio.yaml
```

Batch sweep packs:
```bash
bash scripts/run_sweep_pack.sh
bash scripts/run_sweep_pack_v2_ratio.sh
bash scripts/run_sweep_pack_v3_ratio.sh
bash scripts/run_schedule_sweep_pack.sh
```

Quality:
```bash
make lint
make format
```

## MLflow Setup (Important)

Tracking defaults are repository-local:
- Backend DB: `sqlite:////home/adam/studies/ma/code/mlflow.db`
- Artifact root: `/home/adam/studies/ma/code/mlruns`

Start UI with explicit paths:
```bash
uv run mlflow ui \
  --backend-store-uri sqlite:////home/adam/studies/ma/code/mlflow.db \
  --default-artifact-root /home/adam/studies/ma/code/mlruns \
  --host 127.0.0.1 --port 5000
```

If runs seem missing:
1. Confirm the same backend URI is used by both training and UI.
2. Check experiment name exactly (for example `sweep-ackley-v3-ratio`).
3. Include nested runs in MLflow UI filters.
4. Verify process logs for MLflow startup warnings/errors.

## Repo Conventions

- Use paths from `ma_thesis.config` where possible.
- Keep generated datasets in `data/processed/*.parquet`.
- Keep reproducible plans in `configs/sweeps/*.yaml` and `configs/experiments/*.yaml`.
- Do not hardcode machine-specific paths outside central config.
- Ruff max line length is 99.

## Refactor Status

Completed:
- Shared IO and split handling centralized in `io.py`.
- Shared MLflow config/dataset logging centralized in `mlflow_utils.py`.
- Shared training loop primitives extracted to `training_core.py` and used by
  `train.py`, `sweep.py`, and `schedule_sweep.py`.
- Dead/non-essential code removed:
  - `ma_thesis/plots.py`
  - `notebooks/functions.py`
  - `scripts/zotero_organize.py`

Recommended next:
- Keep `meta_train.py` clearly marked as experimental unless it becomes a primary thesis track.
- Continue separating runtime code (`ma_thesis/`, `configs/`, `scripts/`) from archival thesis artifacts.

## Research Context

Design / method docs:
- `docs/design/2026-01-15-proposal.md`
- `docs/design/2026-03-02-implementation-plan.md`

Thesis sources:
- `thesis/main.tex`
- `thesis/appendices/*`
