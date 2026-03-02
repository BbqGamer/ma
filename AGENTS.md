# AGENTS.md — LLM Project Context

> This file helps AI assistants (Copilot, Cursor, Aider, etc.) quickly orient
> themselves in this repository. Read this first before exploring the codebase.

## Project Summary

**Title:** Dynamic Multi-Loss Curriculum Learning with Meta-Optimization  
**Author:** Adam Korba  
**Type:** Master's thesis (in progress)  
**Language:** Polish thesis, English code and docs  

The project implements curriculum learning for multi-loss regression. A
meta-learning outer loop learns loss weights that transition from smoothed
(crude) to original (detailed) targets — a Gaussian continuation scheme — while
training an MLP on benchmark optimisation functions.

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.10 |
| Deep Learning | PyTorch, JAX / Flax |
| Data | Polars, NumPy, scikit-learn |
| Experiment Tracking | MLflow (local SQLite backend `mlflow.db`, artefacts in `mlruns/`) |
| Notebooks | Marimo, Jupyter |
| Package Management | uv (lockfile `uv.lock`), flit build backend |
| Linting / Formatting | Ruff |
| CLI | Typer |
| Docs | MkDocs |
| Thesis | LaTeX (`thesis/main.tex`, `ppfcmthesis.cls`) |

## Key Directories

### `ma_thesis/` — main Python package
- `config.py` — project paths and constants (PROJ_ROOT, DATA_DIR, etc.)
- `data.py` — benchmark function definitions (Ackley, Levy, Eggholder, Bukin)
- `dataset.py` — Gaussian continuation data generator: generates multi-resolution
  smoothed labels via kernel density weighting. Entry point: `python ma_thesis/dataset.py`
- `train.py` — MLP model (ResBlock-based), training loop with MLflow logging,
  surface plotting utilities (~500 lines, the most substantial module)
- `plots.py` — visualisation of smoothed function surfaces
- `features.py` — feature engineering (scaffold / placeholder)
- `modeling/train.py`, `modeling/predict.py` — placeholder entry points

### `docs/` — project documentation (MkDocs)
- `docs/design/` — **design documents and proposals**
  - `2026-01-15-proposal.md` — mathematical framework for the meta-learning
    curriculum procedure (loss formulation, softmax bridge, monotonic gradient
    regularisation, inner/outer optimisation loops)
- `docs/journal/` — research journal / work log (may be empty early on)
- `docs/meets/` — **supervisor meeting notes and sprint task lists**
  - `2025-11-26.md` — Sprint 1: literature review tasks (curriculum learning,
    multi-task learning), ecosystem setup, data generator requirements
  - `2026-01-15.md` — Sprint 2: Gaussian continuation implementation details,
    meta-learning proposal review tasks

### `thesis/` — LaTeX source
- `main.tex` — thesis document
- `bibliografia.bib` — BibTeX references
- `figures/` — thesis figures (PRISMA flow diagram, university logos)
- `notebooks/` — notebook exports for inclusion in the thesis

### `notebooks/` — interactive exploration
- `gaussian.ipynb` — Gaussian smoothing experiments
- `functions.py` — shared helpers

### `data/processed/` — generated datasets
Parquet files with columns: `x1`, `x2`, `y_sigma_0`, `y_sigma_1`, …, `y_sigma_K`
(one file per benchmark function).

### `models/` — serialised checkpoints
- `data_mlp.pt` — trained MLP weights

### `reports/figures/` — generated plots
Surface plots and learning curves for each smoothing level.

## Important Conventions

- **CLI entry points** use Typer (`app = typer.Typer()`).
- **Paths** are constructed from `ma_thesis.config` (never hardcoded).
- **Experiments** are logged to MLflow — check `mlruns/` or run `mlflow ui`.
- **Data files** are `.parquet` (Polars).
- **Formatting**: Ruff, 99-char line length, isort-style imports.
- The project was scaffolded from Cookiecutter Data Science but has diverged.

## Common Tasks

```bash
make requirements     # install deps via uv sync
make data             # generate smoothed datasets
make lint             # ruff check
make format           # ruff format
mlflow ui             # browse experiment results
```

## Mathematical Context

The meta-learning procedure (documented in `docs/design/2026-01-15-proposal.md`):

1. **Weighted loss:** $L(w, \theta) = \sum_i w_i L_i(\theta)$
2. **Softmax bridge:** weights derived from unconstrained params $u$ via softmax
3. **Inner loop:** standard gradient descent on model params $\theta$
4. **Outer loop:** meta-update on $u$ with monotonic gradient regularisation
   to enforce curriculum schedule (crude weights ↓, detailed weights ↑)
