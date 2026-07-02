# Dynamic Multi-Loss Curriculum Learning with Meta-Optimization

Master's thesis project by **Adam Korba**.

## Overview

This repository contains two main experimental tracks for the thesis.

1. **Gaussian-continuation regression** in `ma_thesis/` — the original track.
   It studies curriculum learning for non-convex benchmark surfaces by training
   on Gaussian-smoothed targets and gradually moving toward the original target.
   Experiments are tracked with MLflow.
2. **Coarse-to-fine image classification** in `coarse-to-fine-curriculum/` — the
   main second-stage thesis track. It is a PyTorch reproduction and extension of
   Stretcu et al.'s output-space curriculum learning idea, with W&B logging,
   Runpod/Docker execution, adaptive schedules, model-capacity studies,
   roughness diagnostics, and random-hierarchy ablations.

The Gaussian-continuation pipeline generates multi-resolution labels using
Gaussian kernel smoothing over benchmark functions from the
[VLSE](https://www.sfu.ca/~ssurjano/) library (Ackley, Levy, Eggholder, and
others). The coarse-to-fine pipeline tests whether learned label hierarchies help
vision models by first training on coarse labels and then fine labels.

## Quick Start

```bash
# create virtual environment
make create_environment
source .venv/bin/activate

# install dependencies
make requirements          # runs `uv sync`

# generate smoothed datasets for the Gaussian-continuation track
make data

# run one Gaussian-continuation experiment
python -m ma_thesis.experiment run --method curriculum --function ackley

# run from yaml config
python -m ma_thesis.experiment run-config configs/sweeps/ackley_sweep.yaml

# lint / format
make lint
make format
```

For the coarse-to-fine image-classification track:

```bash
cd coarse-to-fine-curriculum
python train_coarse_to_fine.py \
  --mode baseline \
  --dataset cifar100 \
  --model cnn \
  --output_dir ./runs \
  --run_id local-baseline

python train_coarse_to_fine.py \
  --mode curriculum \
  --dataset cifar100 \
  --model cnn \
  --reference_run_dir ./runs/local-baseline/cifar100_cnn_baseline \
  --output_dir ./runs \
  --run_id local-curriculum
```

See `coarse-to-fine-curriculum/README.md` for W&B, Docker, and Runpod workflows.

## Gaussian-Continuation Experiment Runner

Use a single entrypoint for the `ma_thesis/` Gaussian-continuation experiment types:

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

Each Gaussian-continuation run writes:
- MLflow params/tags with explicit `strategy` + `model_arch`
- model artifacts (`.pt` checkpoint + MLflow PyTorch model for train/meta)
- run manifest JSON in `reports/runs/` with invoked arguments

Datasets are versioned by default, e.g.:
- `data/processed/ackley_n8000_k5_ss5_seed42.parquet`

## YAML Experiment Configs

For the Gaussian-continuation track, you can commit reproducible experiment
plans in YAML and run them with one command:

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
3. Run curriculum/meta with those exact hyperparameters (or via `--from-sweep`
   and `--study-name` for curriculum).

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
├── ma_thesis/             <- Gaussian-continuation regression track
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
├── coarse-to-fine-curriculum/
│   ├── train_coarse_to_fine.py <- Main PyTorch CLI for output-space curricula
│   ├── ctf/                    <- Data, hierarchy, and model code
│   ├── entrypoint.sh           <- Runpod/Docker experiment entrypoint
│   ├── scripts/                <- W&B export and analysis scripts
│   └── README.md               <- Detailed workflow for the second thesis track
│
├── notebooks/             <- Interactive notebooks (Marimo / Jupyter)
│   ├── gaussian.ipynb     <- Gaussian smoothing explorations
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

The repository uses two tracking backends:

- **MLflow** for the Gaussian-continuation regression track in `ma_thesis/`.
  The local backend store is `mlflow.db` and artifacts live under `mlruns/`.
- **Weights & Biases** for the coarse-to-fine image-classification track in
  `coarse-to-fine-curriculum/`. W&B is the source of truth for the later
  CIFAR-100/Fashion-MNIST/CIFAR-10/Tiny ImageNet analyses.

```bash
mlflow ui                  # browse Gaussian-continuation experiments
```

## Documentation

The `docs/` directory uses MkDocs. It contains design proposals, a research journal,
and supervisor meeting notes. See `docs/mkdocs.yml` for configuration.

## License

MIT — see [LICENSE](LICENSE) for details.
