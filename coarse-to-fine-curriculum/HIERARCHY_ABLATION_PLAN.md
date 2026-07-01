# Final hierarchy ablation plan

## Question

Does the curriculum benefit come from meaningful class grouping, or mostly from reducing the
number of labels during the early training stages?

The clean comparison is:

1. baseline, no curriculum,
2. fixed curriculum with the classifier-weight hierarchy,
3. fixed curriculum with random hierarchies that preserve the same hierarchy shape.

The random condition keeps the number of levels and the cluster sizes fixed by permuting class
labels inside the learned tree. This makes the ablation stronger than comparing against a fully
random tree, because label-count and schedule effects are controlled.

## Recommended primary run

Use CIFAR-100 with the weak CNN, width 0.5, and `curr20`.

Reason: this is the strongest and most thesis-relevant positive case from the previous appendices.
It is also cheap enough to run with three training seeds and three random hierarchy seeds per
training seed.

This produces 15 runs:

- 3 baselines,
- 3 learned-hierarchy curricula,
- 9 random-hierarchy curricula.

```bash
EXPERIMENT=hierarchy_ablation
HIERARCHY_ABLATION_SPECS=cifar100:cnn:0.5:1.0:20:100
HIERARCHY_ABLATION_SEEDS=42,43,44
HIERARCHY_ABLATION_RANDOM_SEEDS=1001,1002,1003
HIERARCHY_ABLATION_WANDB_GROUP=hierarchy-ablation-cifar100-cnn-w05-final
HIERARCHY_ABLATION_WANDB_TAGS=runpod,hierarchy-ablation,cifar100,cnn-w0.5,random-hierarchy
OPTIMIZER=adam
SCHEDULER=none
LR=0.001
BATCH_SIZE=128
VAL_RATIO=0.1
SAVE_CHECKPOINTS=0
WANDB=1
DOWNLOAD=1
```

## Optional cheap control

If the primary run finishes comfortably, add CIFAR-10 `cnn_w1` with `curr10`.
This checks whether the hierarchy effect is still visible on the smaller 10-class natural-image
control where previous curriculum gains were modest.

```bash
HIERARCHY_ABLATION_SPECS=cifar100:cnn:0.5:1.0:20:100,cifar10:cnn:1.0:1.0:10:100
```

This doubles the run count to 30. I would only run the CIFAR-10 control if the CIFAR-100 ablation
finishes without issues.

## Analysis after W&B export

```bash
python scripts/export_wandb_results.py \
  --entity curriculum-learning-ma \
  --project coarse-to-fine-curriculum \
  --group hierarchy-ablation-cifar100-cnn-w05-final \
  --output-dir wandb_export_hierarchy_ablation

python scripts/analyze_hierarchy_ablation.py \
  --summary-csv wandb_export_hierarchy_ablation/wandb_runs_summary.csv \
  --output-dir wandb_analysis_hierarchy_ablation
```

The analysis writes:

- `aggregate_hierarchy_ablation.csv`,
- `paired_hierarchy_deltas_by_seed.csv`,
- `hierarchy_ablation_gain.png`,
- `REPORT.md`.

For the thesis, the main comparison should be learned hierarchy versus random mean. The random-best
column is only an optimistic diagnostic showing what would happen if several random hierarchies
could be tried and the best one kept.
