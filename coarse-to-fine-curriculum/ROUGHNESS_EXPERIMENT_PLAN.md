# Focused roughness experiment for Appendix F

Goal: test whether the Appendix F curriculum gains correspond to smoother / less sharp optimization behavior, without rerunning the full model-size grid.

## Why this subset

Appendix F already identified representative cases:

- curriculum helps: `cnn_w1`, `cifar_resnet8`, `resnet18` weakly,
- curriculum does not help: `cnn_w4`, `cifar_resnet56`.

The roughness experiment compares baseline vs the best/most relevant curriculum for each of these models across several seeds.

Default pairs:

| model | curriculum | reason |
|---|---:|---|
| `cnn_w1` | `curr10` | strongest Appendix F gain |
| `cnn_w4` | `curr5` | larger CNN, no useful gain |
| `cifar_resnet8` | `curr20` | smallest CIFAR ResNet, positive gain |
| `cifar_resnet56` | `curr20` | deeper CIFAR ResNet, negative gain |
| `resnet18` | `curr20` | strongest absolute model, small gain |

Default seeds:

```text
42,43,44
```

This is 30 runs total: 5 models x 2 methods x 3 seeds.

## Roughness epochs

Default:

```text
1,5,6,10,11,20,21,40,50,75,100
```

Rationale:

- `1`: initial post-update behavior,
- `5/6`: immediately before/after `curr5` transition,
- `10/11`: immediately before/after `curr10` transition,
- `20/21`: immediately before/after `curr20` transition,
- `40`: medium training,
- `50`: same checkpoint used in the initial plan,
- `75`: late training,
- `100`: final model.

This is better than only `1,10,11,50,100` because it captures all curriculum transitions used in the selected subset.

## Recommended Runpod env

Use one Runpod pod first. It is operationally simpler and avoids creating many pods manually. The entrypoint skips completed runs if `results.json` already exists, so a persistent volume can resume partial progress.

```bash
IMAGE=bbqdocker/coarse-to-fine-curriculum:<new-tag-after-release>
EXPERIMENT=roughness_subset
DATASET=cifar100

# Core training protocol from Appendix F
ROUGHNESS_SUBSET_EPOCHS=100
VAL_RATIO=0.1
BATCH_SIZE=128
OPTIMIZER=adam
SCHEDULER=none
LR=0.001
WEIGHT_DECAY=0.0
SAVE_CHECKPOINTS=0
ARCHIVE_OUTPUTS=0
DOWNLOAD=1

# Speed/reproducibility tradeoff
# Use DETERMINISTIC=0 because this experiment is multi-seed and does not require bitwise repeatability.
DETERMINISTIC=0
AMP=1
NUM_WORKERS=4

# W&B
WANDB=true
WANDB_PROJECT=coarse-to-fine-curriculum
WANDB_ENTITY=curriculum-learning-ma
ROUGHNESS_SUBSET_WANDB_GROUP=roughness-cifar100-appendix-f
ROUGHNESS_SUBSET_WANDB_TAGS=runpod,roughness,cifar100,appendix-f

# Seeds and selected model/curriculum pairs
ROUGHNESS_SUBSET_SEEDS=42,43,44
ROUGHNESS_SUBSET_SPECS=cnn:1.0:1.0:10,cnn:4.0:1.0:5,cifar_resnet8:1.0:1.0:20,cifar_resnet56:1.0:1.0:20,resnet18:1.0:1.0:20

# Roughness probes
ROUGHNESS_SUBSET_EPOCH_LIST=1,5,6,10,11,20,21,40,50,75,100
ROUGHNESS_SUBSET_BATCHES=2
ROUGHNESS_SUBSET_HESSIAN_ITERS=2
ROUGHNESS_SUBSET_HESSIAN_SAMPLES=1
ROUGHNESS_SUBSET_SHARPNESS_RHO=0.05
```

## What gets logged

Each run logs the usual metrics plus roughness metrics at the selected epochs:

```text
rough_grad_norm_mean
rough_grad_norm_std
rough_grad_norm_cv
rough_grad_norm_skew
rough_gradient_noise_scale
rough_critical_sharpness
rough_relative_critical_sharpness
rough_hessian_top_eigenvalue
rough_hessian_frobenius
rough_hessian_trace
```

The key comparisons are:

1. final roughness: epoch 100 baseline vs curriculum,
2. transition behavior: epochs 5/6, 10/11, 20/21,
3. trajectory behavior: roughness AUC or mean over probe epochs,
4. relationship between accuracy gain and roughness change.

## One pod or many pods?

Recommended now: **one pod**.

Reason:

- only 30 runs,
- simpler operationally,
- fewer chances of W&B grouping mistakes,
- baseline and curriculum reference directories remain local,
- the script can skip completed runs if restarted on a persistent volume.

Use multiple pods only if this takes too long. If parallelizing later, split by seed or by model:

```text
pod 1: ROUGHNESS_SUBSET_SEEDS=42
pod 2: ROUGHNESS_SUBSET_SEEDS=43
pod 3: ROUGHNESS_SUBSET_SEEDS=44
```

or:

```text
pod 1: cnn models
pod 2: cifar_resnet models
pod 3: resnet18
```

Splitting by seed is cleaner for analysis.
