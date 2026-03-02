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
make data                  # runs ma_thesis/dataset.py

# lint / format
make lint
make format
```

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
│   ├── features.py        <- Feature engineering (placeholder)
│   ├── plots.py           <- Visualisation of smoothed surfaces
│   ├── train.py           <- MLP model definition and training loop (PyTorch + MLflow)
│   └── modeling/
│       ├── train.py       <- Model training entry point (placeholder)
│       └── predict.py     <- Inference entry point (placeholder)
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


