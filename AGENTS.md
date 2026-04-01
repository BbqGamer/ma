# AGENTS.md — Project Context For Coding Assistants

> Use this file first when working in this repository.

## Project Snapshot

- **Project:** Dynamic Multi-Loss Curriculum Learning with Meta-Optimization
- **Author:** Adam Korba
- **Context:** Master's thesis (in progress)
- **Language split:** Polish thesis text, English code/docs

Core idea:
- Build Gaussian-continuation targets (`y_sigma_0 ... y_sigma_K`) for 2D benchmark functions.
- Train neural regressors using:
1. single-target training,
2. sequential curriculum over sigma levels,
3. meta-curriculum (bi-level optimization of loss weights),
4. Optuna sweeps (model HP or schedule-only HP).

The research goal is to show whether curriculum learning is useful for some cases,
either when it comes to better generalization or efficiency. All the experiments
should aim to show the gap or improvement between curriculum and single-target training.

The results should be a publishable paper obtaining signifiant results.

## Tech Stack (Current)

- Python 3.10
- PyTorch for training loops and models
- Optuna for sweeps
- MLflow for experiment tracking
- Polars / NumPy / SciPy / scikit-learn for data and utilities
- Typer CLIs
- Ruff formatting/linting
- MkDocs for docs
- LaTeX for thesis

Note:
- `jax` / `flax` are present in dependencies but the main training pipeline currently runs through the PyTorch modules under `ma_thesis/`.

## Repository Map

### `ma_thesis/` (main package)
- `experiment.py` — main orchestrator CLI (`run`, `prepare-data`, `run-config`)
- `dataset.py` — Gaussian continuation dataset generator and train/test split writer
- `train.py` — single/curriculum training, nested MLflow runs, checkpoints, plots
- `meta_train.py` — meta-curriculum training (learned multi-loss weights)
- `sweep.py` — Optuna model-hyperparameter sweep
- `schedule_sweep.py` — Optuna schedule-only sweep (fixed model/training HP)
- `models.py` — model definitions + factory (`mlp`, `siren`, `fourier`)
- `common.py` — shared split/plot helpers and model-parameter budgeting helpers
- `optimization_metrics.py` — gradient/noise/sharpness/spectral diagnostics
- `data.py` — benchmark function definitions
- `config.py` — project paths and constants

### `configs/`
- `configs/sweeps/*.yaml` — per-function sweep configs
- `configs/sweeps/v2_ratio/*.yaml`, `v3_ratio/*.yaml` — ratio-constrained sweep packs
- `configs/experiments/*.yaml` — reusable experiment templates

### `scripts/`
- Sweep pack runners and benchmark helper scripts.
- Important: `run_sweep_pack_v3_ratio.sh` is a long-running batch entrypoint.

### Outputs / Tracking
- `data/processed/` — generated parquet datasets (`*_train.parquet`, `*_test.parquet`)
- `reports/runs/` — per-run manifest JSON from orchestrator
- `reports/logs/` — batch script logs
- `reports/optuna/` — Optuna SQLite files for external study storage
- `models/` — saved checkpoints (`.pt`)
- `mlflow.db` + `mlruns/` — MLflow tracking backend + artifacts

## Run Flows (Preferred)

Single entrypoint for most work:
```bash
python -m ma_thesis.experiment run --method curriculum --function ackley
python -m ma_thesis.experiment run --method single --function ackley --sigma-level -1
python -m ma_thesis.experiment run --method meta --function eggholder
python -m ma_thesis.experiment run --method sweep --function levy --n-trials 80
python -m ma_thesis.experiment run --method schedule_sweep --function ackley
```

Run from YAML:
```bash
python -m ma_thesis.experiment run-config configs/sweeps/v3_ratio/ackley_sweep_v3_ratio.yaml
```

Batch sweep pack:
```bash
bash scripts/run_sweep_pack_v3_ratio.sh
```

## MLflow Behavior (Important)

Tracking defaults:
- backend: `sqlite:////<repo>/mlflow.db`
- artifacts: `<repo>/mlruns`

`train.py` logging structure:
- parent run: global metrics (`global/train_loss`, etc.) and final metrics
- child runs (nested): per-level epoch metrics (`train_loss`, `val_loss`, `hard_val_loss`)

If metrics appear “missing”, verify:
1. correct experiment selected,
2. status filter includes `RUNNING`/`FINISHED`,
3. nested runs are visible (or query by `mlflow.parentRunId`).

## Key Conventions

- Use `ma_thesis.config` paths; avoid hardcoded paths.
- Datasets are parquet and should include sigma columns.
- Keep CLI configs reproducible via YAML in `configs/`.
- Ruff line length is 99.
- Do not commit large generated artifacts unless explicitly intended.

## Common Commands

```bash
make create_environment
source .venv/bin/activate
make requirements

make data
make lint
make format

# MLflow UI using repo-local backend
uv run mlflow ui \
  --backend-store-uri sqlite:////home/adam/studies/ma/code/mlflow.db \
  --default-artifact-root /home/adam/studies/ma/code/mlruns \
  --host 127.0.0.1 --port 5000
```

## Research Context

Design notes for the meta-learning formulation are in:
- `docs/design/2026-01-15-proposal.md`
- `docs/design/2026-03-02-implementation-plan.md`

Thesis source is under `thesis/` (`main.tex`, bibliography, appendices, figures).
