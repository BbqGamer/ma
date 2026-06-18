#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${EXPERIMENT:-single}"
RUN_MODES="${RUN_MODES:-baseline,curriculum}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
DATASET="${DATASET:-cifar100}"
MODEL="${MODEL:-cnn}"
EPOCHS="${EPOCHS:-400}"
CURRICULUM_EPOCHS="${CURRICULUM_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
DROPOUT="${DROPOUT:-0.0}"
PATIENCE="${PATIENCE:-50}"
VAL_RATIO="${VAL_RATIO:-0.2}"
SHAPES_TEST_RATIO="${SHAPES_TEST_RATIO:-0.2}"
DISTANCE_SOURCE="${DISTANCE_SOURCE:-classifier_weights}"
CURRICULUM_TARGET_FRACTION="${CURRICULUM_TARGET_FRACTION:-0.9}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/runs}"
SHAPES_PATH="${SHAPES_PATH:-}"
TINY_IMAGENET_PATH="${TINY_IMAGENET_PATH:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
AMP="${AMP:-1}"
DOWNLOAD="${DOWNLOAD:-1}"
AUGMENTATION="${AUGMENTATION:-auto}"
AUTO_STOP_POD="${AUTO_STOP_POD:-1}"
FIG11_METRIC="${FIG11_METRIC:-test_acc}"
OPTIMIZER="${OPTIMIZER:-}"
SCHEDULER="${SCHEDULER:-}"

common_args=(
  --dataset "$DATASET"
  --model "$MODEL"
  --epochs "$EPOCHS"
  --dropout "$DROPOUT"
  --patience "$PATIENCE"
  --val_ratio "$VAL_RATIO"
  --shapes_test_ratio "$SHAPES_TEST_RATIO"
  --distance_source "$DISTANCE_SOURCE"
  --curriculum_target_fraction "$CURRICULUM_TARGET_FRACTION"
  --data_dir "$DATA_DIR"
  --output_dir "$OUTPUT_DIR"
  --run_id "$RUN_ID"
  --num_workers "$NUM_WORKERS"
  --seed "$SEED"
)

if [[ -n "$BATCH_SIZE" ]]; then
  common_args+=(--batch_size "$BATCH_SIZE")
fi
if [[ -n "$LR" ]]; then
  common_args+=(--lr "$LR")
fi
if [[ -n "$WEIGHT_DECAY" ]]; then
  common_args+=(--weight_decay "$WEIGHT_DECAY")
fi
if [[ -n "$CURRICULUM_EPOCHS" ]]; then
  common_args+=(--curriculum_epochs "$CURRICULUM_EPOCHS")
fi
if [[ -n "$SHAPES_PATH" ]]; then
  common_args+=(--shapes_path "$SHAPES_PATH")
fi
if [[ -n "$TINY_IMAGENET_PATH" ]]; then
  common_args+=(--tiny_imagenet_path "$TINY_IMAGENET_PATH")
fi
if [[ "$AMP" == "1" ]]; then
  common_args+=(--amp)
fi
if [[ "$DOWNLOAD" == "1" ]]; then
  common_args+=(--download)
else
  common_args+=(--no-download)
fi
if [[ "$AUGMENTATION" == "1" || "$AUGMENTATION" == "true" ]]; then
  common_args+=(--augmentation)
elif [[ "$AUGMENTATION" == "0" || "$AUGMENTATION" == "false" ]]; then
  common_args+=(--no-augmentation)
fi

run_single_experiment() {
  local reference_run_dir=""
  IFS=',' read -r -a modes <<< "$RUN_MODES"
  for mode in "${modes[@]}"; do
    mode="$(echo "$mode" | xargs)"
    [[ -z "$mode" ]] && continue
    echo "[entrypoint] Running mode: $mode"
    mode_args=(--mode "$mode" "${common_args[@]}")
    if [[ "$mode" == "curriculum" && -n "$reference_run_dir" ]]; then
      mode_args+=(--reference_run_dir "$reference_run_dir")
    fi
    python train_coarse_to_fine.py "${mode_args[@]}"
    if [[ "$mode" == "baseline" ]]; then
      reference_run_dir="$OUTPUT_DIR/$RUN_ID/${DATASET}_${MODEL}_baseline"
    fi
  done
}

run_figure11_resnet18() {
  echo "[entrypoint] Running Figure-11-style ResNet-18 sweep"
  local optimizer_arg=()
  local scheduler_arg=()
  local lr_arg=()
  if [[ -n "$OPTIMIZER" ]]; then
    optimizer_arg=(--optimizer "$OPTIMIZER")
  fi
  if [[ -n "$SCHEDULER" ]]; then
    scheduler_arg=(--scheduler "$SCHEDULER")
  fi
  if [[ -n "$LR" ]]; then
    lr_arg=(--lr "$LR")
  fi

  python scripts/plan_figure11_resnet18.py \
    --seed "$SEED" \
    --epochs "${EPOCHS:-200}" \
    --val-ratio "${VAL_RATIO:-0.1}" \
    "${optimizer_arg[@]}" \
    "${scheduler_arg[@]}" \
    "${lr_arg[@]}" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR"

  bash figure11_resnet18_commands.sh

  python scripts/plot_figure11_resnet18.py \
    "$OUTPUT_DIR" \
    --seed "$SEED" \
    --metric "$FIG11_METRIC"

  python scripts/analyze_results.py "$OUTPUT_DIR"
}

case "$EXPERIMENT" in
  single)
    run_single_experiment
    ;;
  figure11_resnet18)
    run_figure11_resnet18
    ;;
  *)
    echo "[entrypoint] Unknown EXPERIMENT=$EXPERIMENT"
    exit 1
    ;;
esac

if [[ "$AUTO_STOP_POD" == "1" && -n "${RUNPOD_POD_ID:-}" ]]; then
  echo "[entrypoint] Training finished; stopping Runpod pod ${RUNPOD_POD_ID}"
  if command -v runpodctl >/dev/null 2>&1; then
    runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl remove pod "$RUNPOD_POD_ID" || true
  else
    echo "[entrypoint] runpodctl not found; cannot auto-stop pod"
  fi
fi
