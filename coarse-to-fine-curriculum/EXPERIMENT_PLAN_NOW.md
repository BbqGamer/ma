# Experiment plan to run now

Image to use for all runs:

```text
bbqdocker/coarse-to-fine-curriculum:v0.5.3-402c511
```

The entrypoint always attempts to stop the Runpod pod at the end with:

```bash
runpodctl stop pod "$RUNPOD_POD_ID"
```

Do **not** set `RUNPOD_API_KEY` or `AUTO_STOP_POD`.

---

## Research questions

1. How does model size affect whether coarse-to-fine curriculum learning is useful?
2. Does curriculum learning find less sharp / smoother solutions?
3. Does the Figure-11-style result generalize to other datasets?
4. How do non-accuracy metrics behave during coarse-to-fine curriculum learning?

---

## Shared defaults

Use these for all first-round runs unless noted otherwise:

```text
EXPERIMENT=figure11_resnet18
DATASET=cifar100
SEED=42
EPOCHS=100
VAL_RATIO=0.1
BATCH_SIZE=128

CURRICULUM_LENGTHS=5,10,20,30,40

OPTIMIZER=adam
SCHEDULER=none
LR=0.001

SAVE_CHECKPOINTS=0
ARCHIVE_OUTPUTS=0

WANDB=true
WANDB_API_KEY=<your_wandb_key>
WANDB_PROJECT=coarse-to-fine-curriculum

DOWNLOAD=1
AMP=1
NUM_WORKERS=4
```

Each run automatically executes:

```text
baseline
curr5
curr10
curr20
curr30
curr40
analysis
pareto analysis
```

`curr50` is intentionally skipped.

---

## Roughness probe setting

Use light probes in the first model-size sweep:

```text
ROUGHNESS_PROBES=1
ROUGHNESS_EPOCHS=1,10,11,50,100
ROUGHNESS_BATCHES=1
HESSIAN_ITERS=1
HESSIAN_SAMPLES=1
SHARPNESS_RHO=0.05
```

This logs:

```text
rough_critical_sharpness
rough_relative_critical_sharpness
rough_hessian_top_eigenvalue
rough_hessian_frobenius
rough_hessian_trace
rough_gradient_noise_scale
rough_grad_norm_skew
```

If runtime is too slow, disable roughness for large models only:

```text
ROUGHNESS_PROBES=0
```

---

# Phase 1 — CIFAR-100 model-size sweep

Goal: answer whether curriculum benefit decreases as model capacity increases.

Run each block as a separate Runpod pod/job.

## 1. CNN width 0.5

```text
MODEL=cnn
CNN_WIDTH_MULTIPLIER=0.5
CIFAR_RESNET_WIDTH_MULTIPLIER=1.0
WANDB_GROUP=model-size-cifar100-seed42-cnn-w0.5
WANDB_TAGS=model-size,cifar100,cnn,w0.5,seed42,roughness-light
```

## 2. CNN width 1.0

```text
MODEL=cnn
CNN_WIDTH_MULTIPLIER=1.0
CIFAR_RESNET_WIDTH_MULTIPLIER=1.0
WANDB_GROUP=model-size-cifar100-seed42-cnn-w1
WANDB_TAGS=model-size,cifar100,cnn,w1,seed42,roughness-light
```

## 3. CNN width 2.0

```text
MODEL=cnn
CNN_WIDTH_MULTIPLIER=2.0
CIFAR_RESNET_WIDTH_MULTIPLIER=1.0
WANDB_GROUP=model-size-cifar100-seed42-cnn-w2
WANDB_TAGS=model-size,cifar100,cnn,w2,seed42,roughness-light
```

## 4. CNN width 4.0

```text
MODEL=cnn
CNN_WIDTH_MULTIPLIER=4.0
CIFAR_RESNET_WIDTH_MULTIPLIER=1.0
WANDB_GROUP=model-size-cifar100-seed42-cnn-w4
WANDB_TAGS=model-size,cifar100,cnn,w4,seed42,roughness-light
```

## 5. CIFAR ResNet-8

```text
MODEL=cifar_resnet8
CNN_WIDTH_MULTIPLIER=1.0
CIFAR_RESNET_WIDTH_MULTIPLIER=1.0
WANDB_GROUP=model-size-cifar100-seed42-cifar-resnet8
WANDB_TAGS=model-size,cifar100,cifar-resnet8,seed42,roughness-light
```

## 6. CIFAR ResNet-20

```text
MODEL=cifar_resnet20
CNN_WIDTH_MULTIPLIER=1.0
CIFAR_RESNET_WIDTH_MULTIPLIER=1.0
WANDB_GROUP=model-size-cifar100-seed42-cifar-resnet20
WANDB_TAGS=model-size,cifar100,cifar-resnet20,seed42,roughness-light
```

## 7. CIFAR ResNet-32

```text
MODEL=cifar_resnet32
CNN_WIDTH_MULTIPLIER=1.0
CIFAR_RESNET_WIDTH_MULTIPLIER=1.0
WANDB_GROUP=model-size-cifar100-seed42-cifar-resnet32
WANDB_TAGS=model-size,cifar100,cifar-resnet32,seed42,roughness-light
```

## 8. CIFAR ResNet-56

TODO
```text
MODEL=cifar_resnet56
CNN_WIDTH_MULTIPLIER=1.0
CIFAR_RESNET_WIDTH_MULTIPLIER=1.0
WANDB_GROUP=model-size-cifar100-seed42-cifar-resnet56
WANDB_TAGS=model-size,cifar100,cifar-resnet56,seed42,roughness-light
```

## 9. ResNet-18

```text
MODEL=resnet18
CNN_WIDTH_MULTIPLIER=1.0
CIFAR_RESNET_WIDTH_MULTIPLIER=1.0
WANDB_GROUP=model-size-cifar100-seed42-resnet18
WANDB_TAGS=model-size,cifar100,resnet18,seed42,roughness-light
```

---

## What to analyze after Phase 1

Primary W&B / CSV metrics:

```text
num_trainable_parameters
best_test_acc
gain_best_test_acc
auc_test_acc
epoch_to_95pct_baseline
pareto_accuracy_speed
```

Roughness metrics:

```text
rough_critical_sharpness
rough_hessian_top_eigenvalue
rough_hessian_frobenius
rough_gradient_noise_scale
rough_grad_norm_skew
```

Hierarchy and alternative metrics:

```text
test_f1_macro
test_top5_acc
test_hier_score_official
test_hier_score_learned
test_ece
```

Key plots:

```text
x = log10(num_trainable_parameters)
y = best curriculum gain over baseline
color = architecture family
```

and:

```text
x = curriculum length
y = best_test_acc or auc_test_acc
facet = model
```

---

# Phase 2 — repeat seeds for important models

Do this only after Phase 1. Suggested repeat set:

```text
cnn w0.5
cnn w1
cnn w4
cifar_resnet20
cifar_resnet56
resnet18
```

Run with:

```text
SEED=43
```

then:

```text
SEED=44
```

Keep all other settings the same.

This gives enough data for mean/std estimates without running the full grid immediately.

---

# Phase 3 — dataset generalization

Goal: test whether Figure-11-style curriculum behavior generalizes beyond CIFAR-100.

Use two representative models first:

```text
cnn w1
resnet18
```

Run on:

```text
cifar10
fashion-mnist
svhn
stl10
tiny-imagenet
```

Same defaults:

```text
SEED=42
EPOCHS=100
BATCH_SIZE=128
CURRICULUM_LENGTHS=5,10,20,30,40
OPTIMIZER=adam
SCHEDULER=none
LR=0.001
```

For these first dataset-generalization runs, roughness can be disabled to save time:

```text
ROUGHNESS_PROBES=0
```

Example CNN Fashion-MNIST env additions:

```text
DATASET=fashion-mnist
MODEL=cnn
CNN_WIDTH_MULTIPLIER=1.0
WANDB_GROUP=dataset-generalization-fashion-mnist-cnn-w1-seed42
WANDB_TAGS=dataset-generalization,fashion-mnist,cnn,w1,seed42
```

Example ResNet-18 SVHN env additions:

```text
DATASET=svhn
MODEL=resnet18
WANDB_GROUP=dataset-generalization-svhn-resnet18-seed42
WANDB_TAGS=dataset-generalization,svhn,resnet18,seed42
```

---

# Known problems / caveats

## Curriculum length tuning

We handle this by sweeping:

```text
5,10,20,30,40
```

and reporting both:

```text
best curriculum gain
best curriculum length
```

Do not claim one fixed length is universally best.

## Architecture vs size confound

Do not claim that CNN vs ResNet18 is purely a parameter-count comparison.

Better wording:

```text
Within CNNs and within CIFAR ResNets we test capacity scaling.
Across CNN/ResNet families we test architecture-capacity interaction.
```

## Roughness probes are expensive

The first sweep uses light probes. If runs are too slow, disable probes for the largest models and rerun roughness only on a subset later.

## Some datasets may be too easy

Fashion-MNIST, MNIST, SVHN, and CIFAR-10 may show little curriculum benefit. That is still informative as an easy-task control.
