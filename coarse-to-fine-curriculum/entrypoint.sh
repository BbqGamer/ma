#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${EXPERIMENT:-single}"
RUN_MODES="${RUN_MODES:-baseline,curriculum}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
DATASET="${DATASET:-cifar100}"
MODEL="${MODEL:-}"
EPOCHS="${EPOCHS:-400}"
CURRICULUM_EPOCHS="${CURRICULUM_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
DROPOUT="${DROPOUT:-0.0}"
CNN_WIDTH_MULTIPLIER="${CNN_WIDTH_MULTIPLIER:-1.0}"
CIFAR_RESNET_WIDTH_MULTIPLIER="${CIFAR_RESNET_WIDTH_MULTIPLIER:-1.0}"
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
CURRICULUM_LENGTHS="${CURRICULUM_LENGTHS:-5,10,20,30,40}"
OPTIMIZER="${OPTIMIZER:-}"
SCHEDULER="${SCHEDULER:-}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
ARCHIVE_OUTPUTS="${ARCHIVE_OUTPUTS:-1}"
WANDB="${WANDB:-auto}"
WANDB_PROJECT="${WANDB_PROJECT:-coarse-to-fine-curriculum}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_TAGS="${WANDB_TAGS:-runpod,figure11}"
ROUGHNESS_PROBES="${ROUGHNESS_PROBES:-0}"
ROUGHNESS_EPOCHS="${ROUGHNESS_EPOCHS:-1,5,10,11,20,50,100}"
ROUGHNESS_BATCHES="${ROUGHNESS_BATCHES:-2}"
SHARPNESS_RHO="${SHARPNESS_RHO:-0.05}"
HESSIAN_ITERS="${HESSIAN_ITERS:-10}"
HESSIAN_SAMPLES="${HESSIAN_SAMPLES:-2}"

is_truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

wandb_enabled() {
  if is_truthy "$WANDB"; then
    return 0
  fi
  if [[ "${WANDB,,}" == "auto" && -n "${WANDB_API_KEY:-}" ]]; then
    return 0
  fi
  return 1
}

auto_stop_pod() {
  local status="${1:-0}"
  if ! is_truthy "$AUTO_STOP_POD"; then
    return 0
  fi
  if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    echo "[entrypoint] AUTO_STOP_POD is enabled but RUNPOD_POD_ID is not set"
    return 0
  fi

  echo "[entrypoint] Run finished with status $status; stopping Runpod pod ${RUNPOD_POD_ID}"
  if command -v runpodctl >/dev/null 2>&1; then
    if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
      runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1 || true
    fi
    runpodctl pod stop "$RUNPOD_POD_ID" && return 0
  fi

  if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
    python - "$RUNPOD_POD_ID" "$RUNPOD_API_KEY" <<'PY' || true
import sys
import urllib.request

pod_id, api_key = sys.argv[1], sys.argv[2]
request = urllib.request.Request(
    f"https://rest.runpod.io/v1/pods/{pod_id}/stop",
    method="POST",
    headers={"Authorization": f"Bearer {api_key}"},
)
with urllib.request.urlopen(request, timeout=30) as response:
    print(f"[entrypoint] Runpod REST stop returned HTTP {response.status}")
PY
  else
    echo "[entrypoint] Cannot auto-stop: provide RUNPOD_API_KEY or include configured runpodctl"
  fi
}

trap 'status=$?; auto_stop_pod "$status"' EXIT

echo "[entrypoint] image=bbqdocker/coarse-to-fine-curriculum:v0.3.x"
echo "[entrypoint] EXPERIMENT=$EXPERIMENT WANDB=$WANDB WANDB_PROJECT=$WANDB_PROJECT WANDB_GROUP=${WANDB_GROUP:-<auto>} WANDB_API_KEY_SET=$([[ -n "${WANDB_API_KEY:-}" ]] && echo yes || echo no) RUNPOD_API_KEY_SET=$([[ -n "${RUNPOD_API_KEY:-}" ]] && echo yes || echo no)"

common_args=(
  --dataset "$DATASET"
  --model "${MODEL:-cnn}"
  --epochs "$EPOCHS"
  --dropout "$DROPOUT"
  --cnn-width-multiplier "$CNN_WIDTH_MULTIPLIER"
  --cifar-resnet-width-multiplier "$CIFAR_RESNET_WIDTH_MULTIPLIER"
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
if is_truthy "$ROUGHNESS_PROBES"; then
  common_args+=(--roughness-probes)
  common_args+=(--roughness-epochs "$ROUGHNESS_EPOCHS")
  common_args+=(--roughness-batches "$ROUGHNESS_BATCHES")
  common_args+=(--sharpness-rho "$SHARPNESS_RHO")
  common_args+=(--hessian-iters "$HESSIAN_ITERS")
  common_args+=(--hessian-samples "$HESSIAN_SAMPLES")
fi

if [[ "$SAVE_CHECKPOINTS" == "1" ]]; then
  common_args+=(--save-checkpoints)
else
  common_args+=(--no-save-checkpoints)
fi

if wandb_enabled; then
  common_args+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-tags "$WANDB_TAGS")
  [[ -n "$WANDB_ENTITY" ]] && common_args+=(--wandb-entity "$WANDB_ENTITY")
  [[ -n "$WANDB_GROUP" ]] && common_args+=(--wandb-group "$WANDB_GROUP")
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

wandb_args_array() {
  local default_group="$1"
  local -n out_ref="$2"
  out_ref=()
  if wandb_enabled; then
    out_ref+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-tags "$WANDB_TAGS")
    [[ -n "$WANDB_ENTITY" ]] && out_ref+=(--wandb-entity "$WANDB_ENTITY")
    if [[ -n "$WANDB_GROUP" ]]; then
      out_ref+=(--wandb-group "$WANDB_GROUP")
    else
      out_ref+=(--wandb-group "$default_group")
    fi
  fi
}

run_figure11_sweep() {
  local fig11_model="resnet18"
  if [[ "$EXPERIMENT" == "figure11_cnn" ]]; then
    fig11_model="cnn"
  fi
  if [[ -n "$MODEL" ]]; then
    fig11_model="$MODEL"
  fi
  local fig11_model_token="$fig11_model"
  if [[ "$fig11_model" == "cnn" && "$CNN_WIDTH_MULTIPLIER" != "1.0" && "$CNN_WIDTH_MULTIPLIER" != "1" ]]; then
    fig11_model_token="cnn-w${CNN_WIDTH_MULTIPLIER}"
  fi
  if [[ "$fig11_model" == cifar_resnet* && "$CIFAR_RESNET_WIDTH_MULTIPLIER" != "1.0" && "$CIFAR_RESNET_WIDTH_MULTIPLIER" != "1" ]]; then
    fig11_model_token="${fig11_model}-w${CIFAR_RESNET_WIDTH_MULTIPLIER}"
  fi
  echo "[entrypoint] Running Figure-11-style CIFAR-100 sweep with model=$fig11_model token=$fig11_model_token"
  local optimizer_arg=()
  local scheduler_arg=()
  local lr_arg=()
  local batch_size_arg=()
  local width_arg=(--cnn-width-multiplier "$CNN_WIDTH_MULTIPLIER" --cifar-resnet-width-multiplier "$CIFAR_RESNET_WIDTH_MULTIPLIER")
  local roughness_arg=()
  if [[ -n "$OPTIMIZER" ]]; then
    optimizer_arg=(--optimizer "$OPTIMIZER")
  fi
  if [[ -n "$SCHEDULER" ]]; then
    scheduler_arg=(--scheduler "$SCHEDULER")
  fi
  if [[ -n "$LR" ]]; then
    lr_arg=(--lr "$LR")
  fi
  if [[ -n "$BATCH_SIZE" ]]; then
    batch_size_arg=(--batch-size "$BATCH_SIZE")
  fi
  if is_truthy "$ROUGHNESS_PROBES"; then
    roughness_arg+=(--roughness-probes)
    roughness_arg+=(--roughness-epochs "$ROUGHNESS_EPOCHS")
    roughness_arg+=(--roughness-batches "$ROUGHNESS_BATCHES")
    roughness_arg+=(--sharpness-rho "$SHARPNESS_RHO")
    roughness_arg+=(--hessian-iters "$HESSIAN_ITERS")
    roughness_arg+=(--hessian-samples "$HESSIAN_SAMPLES")
  fi

  local wandb_arg=()
  wandb_args_array "fig11-${fig11_model_token}-${DATASET}-seed${SEED}" wandb_arg

  python scripts/plan_figure11_resnet18.py \
    --seed "$SEED" \
    --dataset "$DATASET" \
    --model "$fig11_model" \
    --epochs "${EPOCHS:-200}" \
    --val-ratio "${VAL_RATIO:-0.1}" \
    --curriculum-lengths "$CURRICULUM_LENGTHS" \
    "${optimizer_arg[@]}" \
    "${scheduler_arg[@]}" \
    "${lr_arg[@]}" \
    "${batch_size_arg[@]}" \
    "${width_arg[@]}" \
    "${roughness_arg[@]}" \
    "${wandb_arg[@]}" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR"

  bash figure11_resnet18_commands.sh

  python scripts/plot_figure11_resnet18.py \
    "$OUTPUT_DIR" \
    --seed "$SEED" \
    --model "$fig11_model" \
    --dataset "$DATASET" \
    --run-prefix "fig11-${fig11_model_token}-${DATASET}" \
    --metric "$FIG11_METRIC" \
    --curriculum-lengths "$CURRICULUM_LENGTHS"

  python scripts/analyze_results.py "$OUTPUT_DIR"
  python scripts/analyze_pareto.py "$OUTPUT_DIR"
}

run_cnn_multiloss() {
  echo "[entrypoint] Running CNN multiloss weighting comparison"
  local group="cnn-multiloss-cifar100-seed${SEED}-epochs${EPOCHS}-bs${BATCH_SIZE:-128}"
  local wandb_arg=()
  wandb_args_array "$group" wandb_arg
  local base_common=(
    --dataset cifar100
    --model cnn
    --epochs "$EPOCHS"
    --val_ratio "$VAL_RATIO"
    --optimizer "${OPTIMIZER:-adam}"
    --scheduler "${SCHEDULER:-none}"
    --lr "${LR:-0.001}"
    --batch_size "${BATCH_SIZE:-128}"
    --cnn-width-multiplier "$CNN_WIDTH_MULTIPLIER"
    --cifar-resnet-width-multiplier "$CIFAR_RESNET_WIDTH_MULTIPLIER"
    --data_dir "$DATA_DIR"
    --output_dir "$OUTPUT_DIR"
    --seed "$SEED"
    --num_workers "$NUM_WORKERS"
  )
  if [[ "$AMP" == "1" ]]; then
    base_common+=(--amp)
  fi
  if is_truthy "$ROUGHNESS_PROBES"; then
    base_common+=(--roughness-probes)
    base_common+=(--roughness-epochs "$ROUGHNESS_EPOCHS")
    base_common+=(--roughness-batches "$ROUGHNESS_BATCHES")
    base_common+=(--sharpness-rho "$SHARPNESS_RHO")
    base_common+=(--hessian-iters "$HESSIAN_ITERS")
    base_common+=(--hessian-samples "$HESSIAN_SAMPLES")
  fi
  if [[ "$DOWNLOAD" == "1" ]]; then
    base_common+=(--download)
  else
    base_common+=(--no-download)
  fi
  if [[ "$SAVE_CHECKPOINTS" == "1" ]]; then
    base_common+=(--save-checkpoints)
  else
    base_common+=(--no-save-checkpoints)
  fi

  local prefix="cnn-multiloss-cifar100-seed${SEED}-epochs${EPOCHS}-bs${BATCH_SIZE:-128}"
  local baseline_id="${prefix}-baseline"
  local reference_dir="$OUTPUT_DIR/$baseline_id/cifar100_cnn_baseline"

  python train_coarse_to_fine.py --mode baseline "${base_common[@]}" --run_id "$baseline_id" "${wandb_arg[@]}"
  python train_coarse_to_fine.py --mode curriculum "${base_common[@]}" --run_id "${prefix}-hard-curr10" --curriculum_epochs 10 --reference_run_dir "$reference_dir" "${wandb_arg[@]}"
  python train_coarse_to_fine.py --mode multiloss "${base_common[@]}" --run_id "${prefix}-uncertainty" --multi-weighting uncertainty --multi-initial-weights 1,1,1,1 --reference_run_dir "$reference_dir" "${wandb_arg[@]}"
  python train_coarse_to_fine.py --mode multiloss "${base_common[@]}" --run_id "${prefix}-gradnorm" --multi-weighting gradnorm --multi-initial-weights 1,1,1,1 --gradnorm-alpha 0.5 --reference_run_dir "$reference_dir" "${wandb_arg[@]}"

  python scripts/analyze_results.py "$OUTPUT_DIR"
  python scripts/analyze_pareto.py "$OUTPUT_DIR"
}

case "$EXPERIMENT" in
  single)
    run_single_experiment
    ;;
  figure11_resnet18|figure11_cnn)
    run_figure11_sweep
    ;;
  cnn_multiloss)
    run_cnn_multiloss
    ;;
  *)
    echo "[entrypoint] Unknown EXPERIMENT=$EXPERIMENT"
    exit 1
    ;;
esac

if [[ "$ARCHIVE_OUTPUTS" == "1" ]]; then
  archive_base="${RUN_ID}"
  archive_members=()
  if [[ "$EXPERIMENT" == "figure11_resnet18" || "$EXPERIMENT" == "figure11_cnn" ]]; then
    archive_model="resnet18"
    [[ "$EXPERIMENT" == "figure11_cnn" ]] && archive_model="cnn"
    [[ -n "$MODEL" ]] && archive_model="$MODEL"
    archive_model_token="$archive_model"
    if [[ "$archive_model" == "cnn" && "$CNN_WIDTH_MULTIPLIER" != "1.0" && "$CNN_WIDTH_MULTIPLIER" != "1" ]]; then
      archive_model_token="cnn-w${CNN_WIDTH_MULTIPLIER}"
    fi
    if [[ "$archive_model" == cifar_resnet* && "$CIFAR_RESNET_WIDTH_MULTIPLIER" != "1.0" && "$CIFAR_RESNET_WIDTH_MULTIPLIER" != "1" ]]; then
      archive_model_token="${archive_model}-w${CIFAR_RESNET_WIDTH_MULTIPLIER}"
    fi
    archive_base="figure11_${archive_model_token}_${DATASET}-seed${SEED}"
    archive_members+=("fig11-${archive_model_token}-${DATASET}-seed${SEED}-baseline")
    IFS=',' read -r -a archive_curriculum_lengths <<< "$CURRICULUM_LENGTHS"
    for n in "${archive_curriculum_lengths[@]}"; do
      n="$(echo "$n" | xargs)"
      [[ -z "$n" ]] && continue
      archive_members+=("fig11-${archive_model_token}-${DATASET}-seed${SEED}-curr${n}")
    done
    archive_members+=("fig11-${archive_model_token}-${DATASET}-seed${SEED}-figure11-analysis")
    [[ -d "$OUTPUT_DIR/analysis" ]] && archive_members+=("analysis")
  else
    archive_members+=("$RUN_ID")
  fi
  archive_path="$OUTPUT_DIR/${archive_base}.tar.gz"
  echo "[entrypoint] Creating archive: $archive_path"
  existing_members=()
  for member in "${archive_members[@]}"; do
    [[ -e "$OUTPUT_DIR/$member" ]] && existing_members+=("$member")
  done
  if [[ ${#existing_members[@]} -gt 0 ]]; then
    tar -czf "$archive_path" -C "$OUTPUT_DIR" "${existing_members[@]}"
  else
    echo "[entrypoint] No matching output directories found to archive"
  fi
fi

