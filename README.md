# Dynamic Multi-Loss Curriculum Learning with Meta-Optimization

Master's thesis project by **Adam Korba**.

## Overview

This project explores **curriculum learning** applied to multi-loss regression via a
**meta-learning outer loop**. The core idea is to learn a schedule of loss weights
$w = (w_1, \dots, w_N)$ that smoothly transition from crude (heavily smoothed) to
detailed (original) approximations of the target function — a technique called
**Gaussian continuation**.

The data pipeline generates multi-resolution labels using Gaussian kernel smoothing
over benchmark functions from the [VLSE](https://www.sfu.ca/~ssurjano/) library
(Ackley, Levy, Eggholder). The training procedure optimises an MLP on the weighted
sum of per-resolution losses while a meta-loop adjusts the weights subject to
monotonic curriculum constraints.

## Quick Start

```bash
# create virtual environment
make create_environment
source .venv/bin/activate

# install dependencies
make requirements          # runs `uv sync`

# generate smoothed datasets (Ackley, Levy, Eggholder)
make data

# run one experiment (single script orchestrator)
python -m ma_thesis.experiment run --method curriculum --function ackley

# run from yaml config
python -m ma_thesis.experiment run-config configs/sweeps/ackley_sweep.yaml

# lint / format
make lint
make format
```

## Unified Experiment Runner

Use a single entrypoint for all experiment types:

```bash
# curriculum learning
python -m ma_thesis.experiment run --method curriculum --function ackley

# single sigma level
python -m ma_thesis.experiment run --method single --function ackley --sigma-level -1

# meta-curriculum
python -m ma_thesis.experiment run --method meta --function eggholder --model-arch fourier

# optuna sweep
python -m ma_thesis.experiment run --method sweep --function levy --n-trials 80
```

Each run writes:
- MLflow params/tags with explicit `strategy` + `model_arch`
- model artifacts (`.pt` checkpoint + MLflow PyTorch model for train/meta)
- run manifest JSON in `reports/runs/` with invoked arguments

Datasets are versioned by default, e.g.:
- `data/processed/ackley_n8000_k5_ss5_seed42.parquet`

## YAML Experiment Configs

You can commit reproducible experiment plans in YAML and run them with one command:

```bash
python -m ma_thesis.experiment run-config configs/sweeps/ackley_sweep.yaml
python -m ma_thesis.experiment run-config configs/sweeps/levy_sweep.yaml
python -m ma_thesis.experiment run-config configs/sweeps/eggholder_sweep.yaml
python -m ma_thesis.experiment run-config configs/sweeps/bukin_sweep.yaml
python -m ma_thesis.experiment run-config configs/sweeps/franke_sweep.yaml
python -m ma_thesis.experiment run-config configs/sweeps/peaks_sweep.yaml
python -m ma_thesis.experiment run-config configs/sweeps/friedman1_2d_sweep.yaml
python -m ma_thesis.experiment run-config configs/sweeps/friedman2_2d_sweep.yaml
python -m ma_thesis.experiment run-config configs/sweeps/schedule_sweep_eggholder.yaml
```

Typical workflow:
1. Run sweep config for a function.
2. Open MLflow, pick best trial hyperparameters.
3. Run curriculum/meta with those exact hyperparameters (or via `--from-sweep` and `--study-name` for curriculum).

Config templates for step 3:
- `configs/experiments/curriculum_from_sweep_template.yaml`
- `configs/experiments/meta_manual_template.yaml`

Start-and-leave sweep pack:
- `./scripts/run_sweep_pack.sh`
- Detailed logs: `reports/logs/sweep_pack_*.log`
- Summary: `reports/logs/sweep_pack_*_summary.log`
- Optuna DBs (resumable): `reports/optuna/*.db`

## Project Structure

```
├── Makefile               <- Convenience commands (make data, make lint, …)
├── pyproject.toml         <- Package metadata and dependencies (flit + uv)
├── uv.lock                <- Locked dependency versions
│
├── ma_thesis/             <- Main Python package
│   ├── config.py          <- Paths and project-wide constants
│   ├── data.py            <- Benchmark functions (Ackley, Levy, Eggholder, …)
│   ├── dataset.py         <- Gaussian continuation data generator (CLI via Typer)
│   ├── experiment.py      <- Single orchestrator CLI for data + train + sweep
│   ├── train.py           <- Curriculum/single training (PyTorch + MLflow)
│   ├── meta_train.py      <- Meta-curriculum training (bi-level optimization)
│   ├── sweep.py           <- Optuna hyperparameter sweeps
│   ├── models.py          <- Model architectures and factory
│   └── common.py          <- Shared data splitting + plotting utilities
│
├── notebooks/             <- Interactive notebooks (Marimo / Jupyter)
│   ├── gaussian.ipynb     <- Gaussian smoothing explorations
│   └── functions.py       <- Shared helper functions for notebooks
│
├── docs/                  <- Project documentation (MkDocs)
│   ├── mkdocs.yml         <- MkDocs configuration
│   ├── design/            <- Design documents and proposals
│   ├── journal/           <- Research journal / work log
│   └── meets/             <- Supervisor meeting notes and sprint plans
│
├── thesis/                <- LaTeX source for the thesis document
│   ├── main.tex           <- Main thesis file
│   ├── bibliografia.bib   <- BibTeX references
│   ├── ppfcmthesis.cls    <- University thesis class
│   ├── figures/           <- Thesis figures (EPS/PDF)
│   └── notebooks/         <- Notebook exports embedded in the thesis
│
├── data/
│   ├── raw/               <- Original, immutable data
│   ├── interim/           <- Intermediate transformed data
│   ├── processed/         <- Final datasets (.parquet) for training
│   └── external/          <- Third-party data
│
├── models/                <- Serialised model checkpoints (.pt)
├── mlruns/                <- MLflow experiment tracking data
├── mlflow.db              <- MLflow backend store (SQLite)
├── reports/
│   └── figures/           <- Generated plots and visualisations
└── references/            <- Papers, manuals, and other reference material
```

## Experiment Tracking

Experiments are tracked with [MLflow](https://mlflow.org/). The local backend store
is `mlflow.db` and artefacts live under `mlruns/`.

```bash
mlflow ui                  # browse experiments at http://localhost:5000
```

## Documentation

The `docs/` directory uses MkDocs. It contains design proposals, a research journal,
and supervisor meeting notes. See `docs/mkdocs.yml` for configuration.

## License

MIT — see [LICENSE](LICENSE) for details.
