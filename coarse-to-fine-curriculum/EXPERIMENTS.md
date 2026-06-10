# Recommended reproduction experiments

This reproduction currently supports:

- datasets: `cifar10`, `cifar100`, `shapes`, `tiny-imagenet`
- models: `cnn`, `resnet18`, `resnet50`
- curriculum type: continuous coarse-to-fine

## What is most interesting to reproduce

## Tier 1 — main paper-style result on CIFAR-100

This should be your first serious suite.

### Why

- CIFAR-100 is where coarse-to-fine should matter more than CIFAR-10.
- The paper reports stronger gains on problems with many labels.
- It lets you compare model scaling effects: `cnn` vs `resnet18` vs `resnet50`.
- It is much cheaper than ImageNet-subset experiments.

### Run

- dataset: `cifar100`
- models: `cnn`, `resnet18`, `resnet50`
- seeds: `42, 43, 44`
- curriculum lengths: `5, 10, 20, 30, 40, 50`
- modes: baseline + curriculum

### What you want to learn

- whether curriculum gain persists beyond the small CNN
- which curriculum length works best per model
- whether stronger backbones reduce or preserve the gain

---

## Tier 2 — CIFAR-10 as a negative / weak-control dataset

### Why

- CIFAR-10 has fewer classes and should usually benefit less.
- This gives you a useful contrast against CIFAR-100.

### Run

- dataset: `cifar10`
- models: `cnn`, `resnet18`
- seeds: `42, 43, 44`
- curriculum lengths: `5, 10, 20, 30, 40, 50`

### Expected outcome

- smaller gains than CIFAR-100
- possibly near-zero gain for stronger models

---

## Tier 3 — Tiny ImageNet for harder real-data validation

### Why

- more classes and harder data than CIFAR
- closer to the paper's "many classes help curriculum" story

### Run

- dataset: `tiny-imagenet`
- models: `cnn`, `resnet18`
- seeds: `42, 43, 44`
- curriculum lengths: `10, 20, 30, 40, 50`

### Caveat

- more expensive
- best after Tier 1 is working cleanly

---

## Tier 4 — Shapes sanity check

### Why

- useful for debugging hierarchy behavior and curriculum shape
- easy to interpret qualitatively

### Run

- dataset: `shapes`
- model: `cnn`
- seeds: `42, 43, 44`
- curriculum lengths: `5, 10, 20, 30`

---

## Suggested execution order

1. `cifar100` + `cnn`
2. `cifar100` + `resnet18`
3. `cifar100` + `resnet50`
4. `cifar10` control
5. `tiny-imagenet`
6. `shapes` sanity / interpretation

## Important implementation note

For the larger models, the paper effectively compares curriculum lengths from:

- `5, 10, 20, 30, 40, 50`

So this suite mirrors that part directly.

## Confusion and difficulty artifacts now saved

Each run now saves additional analysis artifacts for the best restored model:

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

These are meant to support:

- confusion-matrix inspection
- hardest/easiest class analysis
- variance-of-difficulty analysis
- coarse-group ANOVA on CIFAR-100

## Difficulty / landscape proxy metrics

These are not true loss-landscape measurements, but practical proxies:

- class accuracy mean / std / variance
- class accuracy gini of error concentration
- true-class probability mean / std
- top-1 margin mean / std
- top-1 margin on correct vs incorrect predictions
- confusion entropy mean
- off-diagonal confusion mass mean
- for CIFAR-100: coarse-group class-accuracy ANOVA F and eta^2

Interpretation:

- high class-accuracy variance → difficulty is unevenly distributed across classes
- high ANOVA eta^2 → difficulty aligns strongly with coarse semantic groups
- low margins + high confusion entropy → decision boundaries are less confident / more diffuse

## How to generate the recommended commands

```bash
cd coarse-to-fine-curriculum
python scripts/plan_reproduction_suite.py --suite core
```

This writes:

- `suite_commands.sh`
- `suite_manifest.csv`

Then run:

```bash
bash suite_commands.sh
```

## How to analyze many runs

```bash
python scripts/analyze_results.py runs/
```

This gives you:

- per-run analysis inside each run folder
- aggregate summary across runs
- aggregate gain plots
- gain-vs-curriculum-length plots

## Recommended thesis figures

If I were prioritizing figures for the thesis, I would make:

1. mean best-test gain vs curriculum length on `cifar100`, split by model
2. baseline vs curriculum best-test accuracy bars for `cnn`, `resnet18`, `resnet50`
3. representative confusion matrices on `cifar100`
4. per-class accuracy gain plot for one strong positive run
5. difficulty-summary table comparing baseline vs curriculum on `cifar100`
