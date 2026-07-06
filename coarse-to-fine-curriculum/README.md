# Reproduction: Coarse-to-Fine Curriculum Learning

PyTorch rewrite of the code for the paper:
- Otilia Stretcu et al., *Coarse-to-Fine Curriculum Learning* (2021)

This directory is a **primary thesis experiment track**, not an auxiliary
sandbox. It contains the main second-stage experiments of the thesis: CIFAR-100
model-capacity studies, Fashion-MNIST/CIFAR-10/Tiny ImageNet controls,
adaptive plateau schedules, hierarchy inspection, roughness diagnostics, and the
random-vs-learned hierarchy ablation. W&B runs from this track are the source of
truth for Appendices F--H in `thesis/`.

## What is included

- paper-faithful **continuous** curriculum training by default
- hierarchy generation from either:
  - `classifier_weights` (default, closer to the 2021 paper), or
  - `confusion` (closer to the released TensorFlow code)
- datasets from the original repo scope:
  - `cifar10`
  - `cifar100`
  - `shapes`
  - `tiny-imagenet`
- models:
  - `cnn`
  - `resnet18`
  - `resnet50`
- Runpod-ready containerization modeled after `hcl-cifar100/`

## Important note

The released TensorFlow code in original codebase is closer to the **staged**
variant (reset classifier per hierarchy level, hierarchy from confusion), while
the main 2021 paper focuses on the **continuous** variant with label
marginalization and a fixed fine-class head.

This rewrite implements the **continuous variant** as the main reproduction target.

## Files

- `train_coarse_to_fine.py` — main training CLI
- `ctf/data.py` — dataset loading
- `ctf/hierarchy.py` — affinity clustering hierarchy builder
- `ctf/models.py` — CNN / ResNet models
- `Dockerfile`, `entrypoint.sh` — container workflow

## Local usage

From this folder:

```bash
python train_coarse_to_fine.py \
  --mode baseline \
  --dataset cifar100 \
  --model cnn \
  --output_dir ./runs \
  --run_id local-baseline
```

```bash
python train_coarse_to_fine.py \
  --mode curriculum \
  --dataset cifar100 \
  --model cnn \
  --output_dir ./runs \
  --run_id local-curriculum
```

If you already trained a baseline and want to reuse it for hierarchy construction:

```bash
python train_coarse_to_fine.py \
  --mode curriculum \
  --dataset cifar100 \
  --model cnn \
  --reference_run_dir ./runs/local-baseline/cifar100_cnn_baseline \
  --output_dir ./runs \
  --run_id local-curriculum
```

## Shapes dataset

The original shapes data already exists in this repository under:

```bash
coarse-to-fine-curriculum/data/shapes
```

Run with:

```bash
python train_coarse_to_fine.py \
  --mode curriculum \
  --dataset shapes \
  --model cnn \
  --shapes_path ../../coarse-to-fine-curriculum/data/shapes \
  --output_dir ./runs \
  --run_id shapes-curriculum
```

## Main CLI options

```bash
--mode baseline|curriculum
--dataset cifar10|cifar100|shapes|tiny-imagenet
--model cnn|resnet18|resnet50
--epochs INT
--curriculum_epochs INT
--distance_source classifier_weights|confusion|random_permutation|teacher_embeddings
--teacher_run_dir PATH
--teacher_checkpoint_path PATH
--teacher_model cnn|cifar_resnet8|...|resnet18|resnet50
--teacher_pretrained_source none|torchvision_imagenet
--teacher_embedding_split train|val|test
--curriculum_order easy_to_hard|hard_to_easy
--batch_size INT
--lr FLOAT
--weight_decay FLOAT
--val_ratio FLOAT
--patience INT
--download / --no-download
--amp
```

## Default behavior

- `cnn`
  - epochs: `400`
  - optimizer: Adam
  - lr: `1e-3`
  - batch size: `512`
  - no augmentation by default
- `resnet18` / `resnet50`
  - epochs: `200`
  - optimizer: SGD with momentum `0.9`
  - lr: `0.1`
  - weight decay: `5e-4`
  - batch size: `128`
  - basic image augmentation enabled by default

If `--curriculum_epochs` is omitted, it is chosen automatically as the **first baseline epoch reaching
`curriculum_target_fraction * best_val_acc`**, with default target fraction `0.9`.

## Outputs

Each run writes to:

```bash
<output_dir>/<run_id>/<dataset>_<model>_<mode>/
```

Artifacts include:

- `config.json`
- `history.json`
- `history.csv`
- `results.json`
- `summary.csv`
- `best_model.pt`
- `last_model.pt`
- `training_log_*.txt`
- `distance_matrix.npy` (curriculum)
- `hierarchy.json` (curriculum)
- `schedule.json` (curriculum)
- `confusion_val_counts.csv`
- `confusion_val_normalized.csv`
- `confusion_test_counts.csv`
- `confusion_test_normalized.csv`
- `class_metrics_val.csv`
- `class_metrics_test.csv`
- `difficulty_metrics_val.json`
- `difficulty_metrics_val.csv`
- `difficulty_metrics_test.json`
- `difficulty_metrics_test.csv`

`history.csv` is a tidy per-epoch table for plotting, and `summary.csv` is a one-row run summary
for easy aggregation across runs.

## Analysis

Analyze one downloaded run directory:

```bash
python analyze_run.py 2026-06-09_15-20-54
```

Or use the aggregate script on one run or a directory of many runs:

```bash
python scripts/analyze_results.py 2026-06-09_15-20-54
python scripts/analyze_results.py runs/
```

Per-run analysis writes:

- `analysis/comparison_summary.csv`
- `analysis/accuracy_curves.png`
- `analysis/loss_curves.png`
- `analysis/confusion_matrices_test.png`
- `analysis/per_class_accuracy_gain_test.png`
- `analysis/difficulty_summary_test.csv`
- `analysis/report.md`

Aggregate analysis writes:

- `analysis/aggregate_summary.csv`
- `analysis/aggregate_best_test_accuracy.png`
- `analysis/aggregate_gain.png`
- `analysis/gain_vs_curriculum_epochs.png`
- `analysis/gain_vs_curriculum_epochs.csv`
- `analysis/aggregate_report.md`

See also:

- `EXPERIMENTS.md` for the recommended reproduction suite
- `FIGURE11_RESNET18.md` for the Figure-11-style CIFAR-100 ResNet-18 sweep

## Docker / Runpod

Build locally:

```bash
docker build -t coarse-to-fine-curriculum .
```

Run baseline + curriculum in one container:

```bash
docker run --gpus all --rm \
  -e RUN_MODES=baseline,curriculum \
  -e DATASET=cifar100 \
  -e MODEL=cnn \
  -e RUN_ID=demo-cifar100 \
  -e DATA_DIR=/workspace/data \
  -e OUTPUT_DIR=/workspace/runs \
  -v $PWD/data:/workspace/data \
  -v $PWD/runs:/workspace/runs \
  coarse-to-fine-curriculum
```

### Automated Figure-11-style Runpod sweep

To run the ResNet-18 CIFAR-100 curriculum-length sweep automatically inside the container,
set:

```bash
EXPERIMENT=figure11_resnet18  # or figure11_cnn for the paper-like CNN sweep
DATA_DIR=/runpod-volume/data
OUTPUT_DIR=/runpod-volume/runs
SEED=42
EPOCHS=200
VAL_RATIO=0.1
OPTIMIZER=adam
SCHEDULER=none
LR=0.001
FIG11_METRIC=test_acc
SAVE_CHECKPOINTS=0
ARCHIVE_OUTPUTS=1
WANDB=1
WANDB_PROJECT=coarse-to-fine-curriculum
WANDB_ENTITY=
WANDB_GROUP=fig11-resnet18-cifar100-seed42
WANDB_TAGS=runpod,figure11,adam
```

You can either:

- use the default entrypoint with `EXPERIMENT=figure11_resnet18`, or
- run `scripts/run_figure11_resnet18.sh` directly

This will:

1. generate the sweep commands
2. run the baseline + curriculum-length sweep
3. plot the Figure-11-style curves
4. run aggregate analysis
5. stop the pod automatically

### Runpod-style env vars

- `RUN_MODES=baseline,curriculum`
- `DATASET=cifar100`
- `MODEL=cnn`
- `EPOCHS=400`
- `CURRICULUM_EPOCHS=40` (optional)
- `DISTANCE_SOURCE=classifier_weights`
- `DATA_DIR=/workspace/data`
- `OUTPUT_DIR=/workspace/runs`
- `AMP=1`

The entrypoint auto-selects `NUM_WORKERS` when unset as roughly half of available CPUs,
capped to `[2, 12]`. Set `NUM_WORKERS` explicitly to override.

The entrypoint always attempts to stop the Runpod pod at exit when `RUNPOD_POD_ID` is set,
using `runpodctl stop pod $RUNPOD_POD_ID`.

### Teacher hierarchy suite (external teacher + anti-curriculum control)

To compare:
- self-derived hierarchy,
- teacher-embedding hierarchy,
- random hierarchy,
- teacher anti-curriculum,

set:

```bash
EXPERIMENT=teacher_hierarchy_suite
TEACHER_RUN_DIR=/runpod-volume/teachers/cifar100_resnet18_strong
# or TEACHER_CHECKPOINT_PATH + TEACHER_MODEL
# or TEACHER_PRETRAINED_SOURCE=torchvision_imagenet with TEACHER_MODEL=resnet18
TEACHER_EMBEDDING_SPLIT=val
TEACHER_HIERARCHY_SPECS=cifar100:cnn:0.5:1.0:20:100,cifar100:cnn:1.0:1.0:10:100,cifar100:cifar_resnet8:1.0:1.0:20:100
TEACHER_HIERARCHY_SEEDS=42,43,44
TEACHER_HIERARCHY_RANDOM_SEEDS=1001,1002,1003
WANDB=1
WANDB_PROJECT=coarse-to-fine-curriculum
WANDB_GROUP=teacher-hierarchy-suite
WANDB_TAGS=runpod,teacher-hierarchy,anti-curriculum
```

For a one-off teacher probe where the container will stop immediately afterward,
you can auto-export the hierarchy directly inside the baseline run itself:

```bash
EXPORT_TEACHER_HIERARCHY=1
EXPORT_TEACHER_HIERARCHY_SPLIT=val
# optional explicit destination:
# EXPORT_TEACHER_HIERARCHY_DIR=/runpod-volume/teachers/cifar100_resnet18_teacher_hierarchy
```

This suite runs baseline, self hierarchy, teacher hierarchy, random hierarchy,
and teacher anti-curriculum for each configured spec.

For the strongest apples-to-apples follow-up against the previous weak-hierarchy
appendix, use the known CIFAR-100 `cnn_w0.5` setup:

```bash
python scripts/plan_teacher_hierarchy_suite.py
bash teacher_hierarchy_suite_w0_5.sh
```

This generates a 3-seed suite with:
- baseline,
- self hierarchy (previous weak hierarchy source),
- teacher hierarchy,
- teacher anti-curriculum,
- three random hierarchies per seed,

all on `cifar100:cnn:0.5:1.0:20:100`.

After the runs finish, write the comparison table with:

```bash
python scripts/analyze_teacher_hierarchy_suite.py /runpod-volume/runs
```

This produces `analysis/teacher_hierarchy_suite/comparison_table.csv` and a
matching `REPORT.md` with thesis-friendly best-accuracy / final-accuracy / AUC
comparisons.

If you do not already have a saved CIFAR-100 teacher checkpoint, use a
single-container bootstrap run:

```bash
EXPERIMENT=teacher_bootstrap_suite
TEACHER_BOOTSTRAP_RUN_ID=teacher-cifar100-resnet18-bootstrap
TEACHER_BOOTSTRAP_DATASET=cifar100
TEACHER_BOOTSTRAP_MODEL=resnet18
TEACHER_BOOTSTRAP_EPOCHS=30
TEACHER_BOOTSTRAP_PRETRAINED_BACKBONE=1
TEACHER_HIERARCHY_SPECS=cifar100:cnn:0.5:1.0:20:100
TEACHER_HIERARCHY_SEEDS=42,43,44
TEACHER_HIERARCHY_RANDOM_SEEDS=1001,1002,1003
```

This first trains and saves a ResNet-18 teacher baseline with checkpoints,
optionally exports its hierarchy, then immediately reuses that saved run as the
teacher source for the full multi-seed comparison suite. The whole workflow stays
inside `entrypoint.sh`, so it is suitable for a single Runpod env file.

If you do not have a CIFAR-100 teacher yet and want the simplest fallback
without a learned teacher run, use:

```bash
TEACHER_PRETRAINED_SOURCE=torchvision_imagenet
TEACHER_MODEL=resnet18
```

This uses an ImageNet-pretrained torchvision backbone as the teacher for
embedding-based hierarchy construction. It is convenient, but less direct than a
strong CIFAR-100 teacher, so treat it as a pragmatic baseline rather than an
ideal match to Weinshall.

For shapes:

- mount or copy the dataset
- set `DATASET=shapes`
- set `SHAPES_PATH=/workspace/data/shapes`

For Tiny ImageNet:

- set `DATASET=tiny-imagenet`
- optionally set `TINY_IMAGENET_PATH=/workspace/data/tiny-imagenet-200`

## Release workflow

Same pattern as `hcl-cifar100/`:

```bash
make release
```

Optional:

```bash
IMAGE=yourdockerhubuser/coarse-to-fine-curriculum make release
PLATFORM=linux/amd64 make release
```

## Current limits

This is a practical reproduction scaffold, not yet a full reimplementation of every table in the
paper. In particular:

- no WideResNet-28-10 yet
- no automatic full paper sweep scripts yet
- no direct comparison scripts against NBDT / SPL / multitask baselines yet

But the core coarse-to-fine curriculum algorithm, hierarchy construction, PyTorch training, and
Runpod container workflow are in place.
