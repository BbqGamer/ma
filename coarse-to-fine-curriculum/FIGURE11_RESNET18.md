# Figure-11-style experiment on CIFAR-100 with ResNet-18

This experiment is the closest practical analogue of Appendix Figure 11, but on a stronger model than the paper's small CNN.

## Goal

Measure how curriculum length affects CIFAR-100 performance for `resnet18`.

## Curriculum lengths

We test:

- `5`
- `10`
- `20`
- `30`
- `40`
- `50`

against a single baseline.

## Why ResNet-18

- much stronger than the small CNN
- already supported in this reproduction codebase
- still cheap enough to sweep on Runpod
- good compromise before adding WideResNet-28-10

## Optimizer choice

This sweep now defaults to:

- optimizer: `adam`
- scheduler: `none`
- learning rate: `1e-3`

The reason is to avoid the large synchronized jump caused by the step learning-rate schedule in the SGD setup, which made curriculum-length effects harder to interpret.

## Recommended settings

Use this cleaner analysis-oriented setting:

- dataset: `cifar100`
- model: `resnet18`
- optimizer: `adam`
- scheduler: `none`
- learning rate: `1e-3`
- epochs: `200`
- validation split: `0.1`
- seed: at least `42`

If you have budget, repeat with:

- seeds `42, 43, 44`

## Step 1: generate commands

```bash
cd coarse-to-fine-curriculum
python scripts/plan_figure11_resnet18.py --seed 42
```

This writes:

- `figure11_resnet18_commands.sh`
- `figure11_resnet18_manifest.csv`

## Step 2: run the sweep

```bash
bash figure11_resnet18_commands.sh
```

This runs:

- 1 baseline
- 6 curriculum runs

Total: **7 runs**.

## Step 3: make the plot

```bash
python scripts/plot_figure11_resnet18.py runs/ --seed 42 --metric test_acc
```

This writes:

```bash
runs/fig11-resnet18-cifar100-seed42-figure11-analysis/
```

with:

- `figure11_test_acc.png`
- `figure11_val_acc.png`
- `figure11_summary.csv`

## What else each run already contains

Each run also saves:

- confusion matrices
- per-class metrics
- difficulty metrics
- full history CSV

so you can inspect not only peak accuracy but also:

- which curriculum lengths change the confusion structure
- which classes gain the most
- whether difficulty becomes more/less concentrated

## Suggested analysis after the plot

After the length sweep, inspect:

1. **best curriculum length by best test accuracy**
2. **best curriculum length by earliest strong performance**
3. **confusion matrices** for:
   - baseline
   - best curriculum length
   - worst curriculum length
4. **difficulty metrics** for the same three runs

## If you have more budget

Repeat the exact same sweep for:

- `--seed 43`
- `--seed 44`

Then run:

```bash
python scripts/analyze_results.py runs/
```

This gives you aggregate plots and lets you check whether the preferred curriculum length is stable across seeds.

## Minimum result worth reporting

At minimum, report:

- the Figure-11-style test-accuracy plot
- the table from `figure11_summary.csv`
- one confusion-matrix comparison for baseline vs best curriculum length
- one difficulty-summary comparison for baseline vs best curriculum length
