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
if [[ -z "${NUM_WORKERS:-}" ]]; then
  CPU_COUNT="${RUNPOD_CPU_COUNT:-$(nproc 2>/dev/null || echo 4)}"
  NUM_WORKERS="$(( CPU_COUNT / 2 ))"
  (( NUM_WORKERS < 2 )) && NUM_WORKERS=2
  (( NUM_WORKERS > 12 )) && NUM_WORKERS=12
  echo "[entrypoint] Auto-selected NUM_WORKERS=$NUM_WORKERS from CPU_COUNT=$CPU_COUNT"
else
  echo "[entrypoint] Using explicit NUM_WORKERS=$NUM_WORKERS"
fi
SEED="${SEED:-42}"
DETERMINISTIC="${DETERMINISTIC:-1}"
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
AMP="${AMP:-1}"
DOWNLOAD="${DOWNLOAD:-1}"
AUGMENTATION="${AUGMENTATION:-auto}"
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
ROUGHNESS_SUBSET_SEEDS="${ROUGHNESS_SUBSET_SEEDS:-42,43,44}"
ROUGHNESS_SUBSET_SPECS="${ROUGHNESS_SUBSET_SPECS:-cnn:1.0:1.0:10,cnn:4.0:1.0:5,cifar_resnet8:1.0:1.0:20,cifar_resnet56:1.0:1.0:20,resnet18:1.0:1.0:20}"
ROUGHNESS_SUBSET_EPOCHS="${ROUGHNESS_SUBSET_EPOCHS:-100}"
ROUGHNESS_SUBSET_PATIENCE="${ROUGHNESS_SUBSET_PATIENCE:-0}"
ROUGHNESS_SUBSET_EPOCH_LIST="${ROUGHNESS_SUBSET_EPOCH_LIST:-1,5,6,10,11,20,21,40,50,75,100}"
ROUGHNESS_SUBSET_BATCHES="${ROUGHNESS_SUBSET_BATCHES:-2}"
ROUGHNESS_SUBSET_HESSIAN_ITERS="${ROUGHNESS_SUBSET_HESSIAN_ITERS:-2}"
ROUGHNESS_SUBSET_HESSIAN_SAMPLES="${ROUGHNESS_SUBSET_HESSIAN_SAMPLES:-1}"
ROUGHNESS_SUBSET_SHARPNESS_RHO="${ROUGHNESS_SUBSET_SHARPNESS_RHO:-0.05}"
ROUGHNESS_SUBSET_WANDB_GROUP="${ROUGHNESS_SUBSET_WANDB_GROUP:-roughness-cifar100-appendix-f}"
ROUGHNESS_SUBSET_WANDB_TAGS="${ROUGHNESS_SUBSET_WANDB_TAGS:-runpod,roughness,cifar100,appendix-f}"

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
  if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    echo "[entrypoint] RUNPOD_POD_ID is not set; skipping Runpod stop command"
    return 0
  fi
  if ! command -v runpodctl >/dev/null 2>&1; then
    echo "[entrypoint] runpodctl not found; cannot stop Runpod pod ${RUNPOD_POD_ID}"
    return 0
  fi

  echo "[entrypoint] Run finished with status $status; stopping Runpod pod ${RUNPOD_POD_ID}"
  echo "[entrypoint] Executing: runpodctl stop pod ${RUNPOD_POD_ID}"
  runpodctl stop pod "$RUNPOD_POD_ID" || true
}

trap 'status=$?; auto_stop_pod "$status"' EXIT

echo "[entrypoint] image=bbqdocker/coarse-to-fine-curriculum:v0.5.x"
echo "[entrypoint] EXPERIMENT=$EXPERIMENT WANDB=$WANDB WANDB_PROJECT=$WANDB_PROJECT WANDB_GROUP=${WANDB_GROUP:-<auto>} WANDB_API_KEY_SET=$([[ -n "${WANDB_API_KEY:-}" ]] && echo yes || echo no)"
echo "[entrypoint] SEED=$SEED DETERMINISTIC=$DETERMINISTIC PYTHONHASHSEED=$PYTHONHASHSEED CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"

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
if is_truthy "$DETERMINISTIC"; then
  common_args+=(--deterministic)
else
  common_args+=(--no-deterministic)
fi

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
if is_truthy "$DOWNLOAD"; then
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
  local deterministic_arg=()
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
  if is_truthy "$DETERMINISTIC"; then
    deterministic_arg=(--deterministic)
  else
    deterministic_arg=(--no-deterministic)
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
    "${deterministic_arg[@]}" \
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

model_token_for_spec() {
  local model="$1"
  local cnn_width="$2"
  local cifar_width="$3"
  local token="$model"
  if [[ "$model" == "cnn" && "$cnn_width" != "1.0" && "$cnn_width" != "1" ]]; then
    token="cnn-w${cnn_width}"
  fi
  if [[ "$model" == cifar_resnet* && "$cifar_width" != "1.0" && "$cifar_width" != "1" ]]; then
    token="${model}-w${cifar_width}"
  fi
  echo "$token"
}

run_roughness_subset() {
  echo "[entrypoint] Running focused roughness subset"
  echo "[entrypoint] specs=$ROUGHNESS_SUBSET_SPECS seeds=$ROUGHNESS_SUBSET_SEEDS"
  echo "[entrypoint] roughness_epochs=$ROUGHNESS_SUBSET_EPOCH_LIST roughness_batches=$ROUGHNESS_SUBSET_BATCHES hessian_iters=$ROUGHNESS_SUBSET_HESSIAN_ITERS hessian_samples=$ROUGHNESS_SUBSET_HESSIAN_SAMPLES"

  local optimizer_value="${OPTIMIZER:-adam}"
  local scheduler_value="${SCHEDULER:-none}"
  local lr_value="${LR:-0.001}"
  local batch_size_value="${BATCH_SIZE:-128}"
  local val_ratio_value="${VAL_RATIO:-0.1}"
  local weight_decay_value="${WEIGHT_DECAY:-0.0}"

  local wandb_arg=()
  if wandb_enabled; then
    wandb_arg+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$ROUGHNESS_SUBSET_WANDB_GROUP")
    wandb_arg+=(--wandb-tags "$ROUGHNESS_SUBSET_WANDB_TAGS")
    [[ -n "$WANDB_ENTITY" ]] && wandb_arg+=(--wandb-entity "$WANDB_ENTITY")
  fi

  local deterministic_arg=()
  if is_truthy "$DETERMINISTIC"; then
    deterministic_arg=(--deterministic)
  else
    deterministic_arg=(--no-deterministic)
  fi

  local download_arg=(--download)
  is_truthy "$DOWNLOAD" || download_arg=(--no-download)

  local checkpoint_arg=(--no-save-checkpoints)
  [[ "$SAVE_CHECKPOINTS" == "1" ]] && checkpoint_arg=(--save-checkpoints)

  local amp_arg=()
  [[ "$AMP" == "1" ]] && amp_arg=(--amp)

  IFS=',' read -r -a seeds <<< "$ROUGHNESS_SUBSET_SEEDS"
  IFS=',' read -r -a specs <<< "$ROUGHNESS_SUBSET_SPECS"
  for seed in "${seeds[@]}"; do
    seed="$(echo "$seed" | xargs)"
    [[ -z "$seed" ]] && continue
    for spec in "${specs[@]}"; do
      spec="$(echo "$spec" | xargs)"
      [[ -z "$spec" ]] && continue
      IFS=':' read -r spec_model spec_cnn_width spec_cifar_width spec_curr_epochs <<< "$spec"
      if [[ -z "${spec_model:-}" || -z "${spec_cnn_width:-}" || -z "${spec_cifar_width:-}" || -z "${spec_curr_epochs:-}" ]]; then
        echo "[entrypoint] Invalid ROUGHNESS_SUBSET_SPECS item: $spec"
        exit 1
      fi

      local token
      token="$(model_token_for_spec "$spec_model" "$spec_cnn_width" "$spec_cifar_width")"
      local prefix="rough-${DATASET}-${token}-seed${seed}"
      local baseline_id="${prefix}-baseline"
      local curriculum_id="${prefix}-curr${spec_curr_epochs}"
      local baseline_dir="$OUTPUT_DIR/$baseline_id/${DATASET}_${spec_model}_baseline"

      local base_args=(
        --dataset "$DATASET"
        --model "$spec_model"
        --epochs "$ROUGHNESS_SUBSET_EPOCHS"
        --val_ratio "$val_ratio_value"
        --optimizer "$optimizer_value"
        --scheduler "$scheduler_value"
        --lr "$lr_value"
        --weight_decay "$weight_decay_value"
        --batch_size "$batch_size_value"
        --dropout "$DROPOUT"
        --cnn-width-multiplier "$spec_cnn_width"
        --cifar-resnet-width-multiplier "$spec_cifar_width"
        --patience "$ROUGHNESS_SUBSET_PATIENCE"
        --distance_source "$DISTANCE_SOURCE"
        --data_dir "$DATA_DIR"
        --output_dir "$OUTPUT_DIR"
        --num_workers "$NUM_WORKERS"
        --seed "$seed"
        --roughness-probes
        --roughness-epochs "$ROUGHNESS_SUBSET_EPOCH_LIST"
        --roughness-batches "$ROUGHNESS_SUBSET_BATCHES"
        --sharpness-rho "$ROUGHNESS_SUBSET_SHARPNESS_RHO"
        --hessian-iters "$ROUGHNESS_SUBSET_HESSIAN_ITERS"
        --hessian-samples "$ROUGHNESS_SUBSET_HESSIAN_SAMPLES"
        "${deterministic_arg[@]}"
        "${download_arg[@]}"
        "${checkpoint_arg[@]}"
        "${amp_arg[@]}"
        "${wandb_arg[@]}"
      )

      echo "[entrypoint] Roughness baseline: seed=$seed model=$spec_model token=$token"
      if [[ -f "$baseline_dir/results.json" ]]; then
        echo "[entrypoint] Skipping completed baseline: $baseline_id"
      else
        python train_coarse_to_fine.py --mode baseline --run_id "$baseline_id" "${base_args[@]}"
      fi

      echo "[entrypoint] Roughness curriculum: seed=$seed model=$spec_model token=$token curr=$spec_curr_epochs"
      local curriculum_dir="$OUTPUT_DIR/$curriculum_id/${DATASET}_${spec_model}_curriculum"
      if [[ -f "$curriculum_dir/results.json" ]]; then
        echo "[entrypoint] Skipping completed curriculum: $curriculum_id"
      else
        python train_coarse_to_fine.py \
          --mode curriculum \
          --run_id "$curriculum_id" \
          --curriculum_epochs "$spec_curr_epochs" \
          --reference_run_dir "$baseline_dir" \
          "${base_args[@]}"
      fi
    done
  done

  python scripts/analyze_results.py "$OUTPUT_DIR" || true
  python scripts/analyze_pareto.py "$OUTPUT_DIR" || true
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
  if is_truthy "$DETERMINISTIC"; then
    base_common+=(--deterministic)
  else
    base_common+=(--no-deterministic)
  fi
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
  if is_truthy "$DOWNLOAD"; then
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
  roughness_subset)
    run_roughness_subset
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

