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
RANDOM_HIERARCHY_SEED="${RANDOM_HIERARCHY_SEED:-}"
CURRICULUM_TARGET_FRACTION="${CURRICULUM_TARGET_FRACTION:-0.9}"
CURRICULUM_POLICY="${CURRICULUM_POLICY:-fixed}"
CURRICULUM_ORDER="${CURRICULUM_ORDER:-easy_to_hard}"
CURRICULUM_MIN_CLUSTERS="${CURRICULUM_MIN_CLUSTERS:-0}"
CURRICULUM_MAX_LEVELS="${CURRICULUM_MAX_LEVELS:-0}"
CURRICULUM_STAGE_MIN_EPOCHS="${CURRICULUM_STAGE_MIN_EPOCHS:-10}"
CURRICULUM_STAGE_MAX_EPOCHS="${CURRICULUM_STAGE_MAX_EPOCHS:-50}"
CURRICULUM_STAGE_PATIENCE="${CURRICULUM_STAGE_PATIENCE:-5}"
CURRICULUM_STAGE_MIN_DELTA="${CURRICULUM_STAGE_MIN_DELTA:-0.002}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/runs}"
SHAPES_PATH="${SHAPES_PATH:-}"
TINY_IMAGENET_PATH="${TINY_IMAGENET_PATH:-}"
TEACHER_RUN_DIR="${TEACHER_RUN_DIR:-}"
TEACHER_CHECKPOINT_PATH="${TEACHER_CHECKPOINT_PATH:-}"
TEACHER_MODEL="${TEACHER_MODEL:-}"
TEACHER_CNN_WIDTH_MULTIPLIER="${TEACHER_CNN_WIDTH_MULTIPLIER:-}"
TEACHER_CIFAR_RESNET_WIDTH_MULTIPLIER="${TEACHER_CIFAR_RESNET_WIDTH_MULTIPLIER:-}"
TEACHER_EMBEDDING_SPLIT="${TEACHER_EMBEDDING_SPLIT:-val}"
TEACHER_PRETRAINED_SOURCE="${TEACHER_PRETRAINED_SOURCE:-none}"
EXPORT_TEACHER_HIERARCHY="${EXPORT_TEACHER_HIERARCHY:-0}"
EXPORT_TEACHER_HIERARCHY_SPLIT="${EXPORT_TEACHER_HIERARCHY_SPLIT:-val}"
EXPORT_TEACHER_HIERARCHY_DIR="${EXPORT_TEACHER_HIERARCHY_DIR:-}"
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
PRETRAINED_BACKBONE="${PRETRAINED_BACKBONE:-0}"
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
HIERARCHY_ABLATION_SEEDS="${HIERARCHY_ABLATION_SEEDS:-42,43,44}"
HIERARCHY_ABLATION_RANDOM_SEEDS="${HIERARCHY_ABLATION_RANDOM_SEEDS:-1001,1002,1003}"
HIERARCHY_ABLATION_SPECS="${HIERARCHY_ABLATION_SPECS:-cifar100:cnn:0.5:1.0:20:100}"
HIERARCHY_ABLATION_PATIENCE="${HIERARCHY_ABLATION_PATIENCE:-0}"
HIERARCHY_ABLATION_WANDB_GROUP="${HIERARCHY_ABLATION_WANDB_GROUP:-hierarchy-ablation-final}"
HIERARCHY_ABLATION_WANDB_TAGS="${HIERARCHY_ABLATION_WANDB_TAGS:-runpod,hierarchy-ablation,random-hierarchy}"
TEACHER_HIERARCHY_SPECS="${TEACHER_HIERARCHY_SPECS:-cifar100:cnn:0.5:1.0:20:100,cifar100:cnn:1.0:1.0:10:100,cifar100:cifar_resnet8:1.0:1.0:20:100}"
TEACHER_HIERARCHY_SEEDS="${TEACHER_HIERARCHY_SEEDS:-42,43,44}"
TEACHER_HIERARCHY_RANDOM_SEEDS="${TEACHER_HIERARCHY_RANDOM_SEEDS:-1001,1002,1003}"
TEACHER_HIERARCHY_WANDB_GROUP="${TEACHER_HIERARCHY_WANDB_GROUP:-teacher-hierarchy-suite}"
TEACHER_HIERARCHY_WANDB_TAGS="${TEACHER_HIERARCHY_WANDB_TAGS:-runpod,teacher-hierarchy,anti-curriculum}"
TEACHER_HIERARCHY_OUTPUT_PREFIX="${TEACHER_HIERARCHY_OUTPUT_PREFIX:-teacher}"
TEACHER_HIERARCHY_REFERENCE_PREFIX="${TEACHER_HIERARCHY_REFERENCE_PREFIX:-teacher}"
TEACHER_HIERARCHY_RUN_CONDITIONS="${TEACHER_HIERARCHY_RUN_CONDITIONS:-baseline,self,teacher,teacher_anti,random}"
TEACHER_BOOTSTRAP_RUN_ID="${TEACHER_BOOTSTRAP_RUN_ID:-teacher-cifar100-resnet18-bootstrap}"
TEACHER_BOOTSTRAP_DATASET="${TEACHER_BOOTSTRAP_DATASET:-cifar100}"
TEACHER_BOOTSTRAP_MODEL="${TEACHER_BOOTSTRAP_MODEL:-resnet18}"
TEACHER_BOOTSTRAP_EPOCHS="${TEACHER_BOOTSTRAP_EPOCHS:-30}"
TEACHER_BOOTSTRAP_BATCH_SIZE="${TEACHER_BOOTSTRAP_BATCH_SIZE:-128}"
TEACHER_BOOTSTRAP_OPTIMIZER="${TEACHER_BOOTSTRAP_OPTIMIZER:-sgd}"
TEACHER_BOOTSTRAP_SCHEDULER="${TEACHER_BOOTSTRAP_SCHEDULER:-step}"
TEACHER_BOOTSTRAP_LR="${TEACHER_BOOTSTRAP_LR:-0.1}"
TEACHER_BOOTSTRAP_WEIGHT_DECAY="${TEACHER_BOOTSTRAP_WEIGHT_DECAY:-0.0005}"
TEACHER_BOOTSTRAP_VAL_RATIO="${TEACHER_BOOTSTRAP_VAL_RATIO:-0.1}"
TEACHER_BOOTSTRAP_SEED="${TEACHER_BOOTSTRAP_SEED:-42}"
TEACHER_BOOTSTRAP_PRETRAINED_BACKBONE="${TEACHER_BOOTSTRAP_PRETRAINED_BACKBONE:-1}"
TEACHER_BOOTSTRAP_EXPORT_HIERARCHY="${TEACHER_BOOTSTRAP_EXPORT_HIERARCHY:-1}"
TEACHER_BOOTSTRAP_EXPORT_SPLIT="${TEACHER_BOOTSTRAP_EXPORT_SPLIT:-val}"
TEACHER_BOOTSTRAP_WANDB_GROUP="${TEACHER_BOOTSTRAP_WANDB_GROUP:-teacher-bootstrap}"
TEACHER_BOOTSTRAP_WANDB_TAGS="${TEACHER_BOOTSTRAP_WANDB_TAGS:-runpod,teacher-bootstrap,cifar100,resnet18}"

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

teacher_condition_enabled() {
  local condition="$1"
  case ",${TEACHER_HIERARCHY_RUN_CONDITIONS}," in
    *,"$condition",*) return 0 ;;
    *) return 1 ;;
  esac
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
  --curriculum-policy "$CURRICULUM_POLICY"
  --curriculum-order "$CURRICULUM_ORDER"
  --curriculum-min-clusters "$CURRICULUM_MIN_CLUSTERS"
  --curriculum-max-levels "$CURRICULUM_MAX_LEVELS"
  --curriculum-stage-min-epochs "$CURRICULUM_STAGE_MIN_EPOCHS"
  --curriculum-stage-max-epochs "$CURRICULUM_STAGE_MAX_EPOCHS"
  --curriculum-stage-patience "$CURRICULUM_STAGE_PATIENCE"
  --curriculum-stage-min-delta "$CURRICULUM_STAGE_MIN_DELTA"
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
if [[ -n "$RANDOM_HIERARCHY_SEED" ]]; then
  common_args+=(--random-hierarchy-seed "$RANDOM_HIERARCHY_SEED")
fi
if [[ -n "$SHAPES_PATH" ]]; then
  common_args+=(--shapes_path "$SHAPES_PATH")
fi
if [[ -n "$TINY_IMAGENET_PATH" ]]; then
  common_args+=(--tiny_imagenet_path "$TINY_IMAGENET_PATH")
fi
if [[ -n "$TEACHER_RUN_DIR" ]]; then
  common_args+=(--teacher_run_dir "$TEACHER_RUN_DIR")
fi
if [[ -n "$TEACHER_CHECKPOINT_PATH" ]]; then
  common_args+=(--teacher_checkpoint_path "$TEACHER_CHECKPOINT_PATH")
fi
if [[ -n "$TEACHER_MODEL" ]]; then
  common_args+=(--teacher_model "$TEACHER_MODEL")
fi
common_args+=(--teacher_pretrained_source "$TEACHER_PRETRAINED_SOURCE")
if [[ -n "$TEACHER_CNN_WIDTH_MULTIPLIER" ]]; then
  common_args+=(--teacher_cnn_width_multiplier "$TEACHER_CNN_WIDTH_MULTIPLIER")
fi
if [[ -n "$TEACHER_CIFAR_RESNET_WIDTH_MULTIPLIER" ]]; then
  common_args+=(--teacher_cifar_resnet_width_multiplier "$TEACHER_CIFAR_RESNET_WIDTH_MULTIPLIER")
fi
common_args+=(--teacher_embedding_split "$TEACHER_EMBEDDING_SPLIT")
if [[ "$AMP" == "1" ]]; then
  common_args+=(--amp)
fi
if is_truthy "$PRETRAINED_BACKBONE"; then
  common_args+=(--pretrained-backbone)
fi
if is_truthy "$EXPORT_TEACHER_HIERARCHY"; then
  common_args+=(--export-teacher-hierarchy)
  common_args+=(--export-teacher-hierarchy-split "$EXPORT_TEACHER_HIERARCHY_SPLIT")
  [[ -n "$EXPORT_TEACHER_HIERARCHY_DIR" ]] && common_args+=(--export-teacher-hierarchy-dir "$EXPORT_TEACHER_HIERARCHY_DIR")
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
  local curriculum_policy_arg=(
    --curriculum-policy "$CURRICULUM_POLICY"
    --curriculum-min-clusters "$CURRICULUM_MIN_CLUSTERS"
    --curriculum-max-levels "$CURRICULUM_MAX_LEVELS"
    --curriculum-stage-min-epochs "$CURRICULUM_STAGE_MIN_EPOCHS"
    --curriculum-stage-max-epochs "$CURRICULUM_STAGE_MAX_EPOCHS"
    --curriculum-stage-patience "$CURRICULUM_STAGE_PATIENCE"
    --curriculum-stage-min-delta "$CURRICULUM_STAGE_MIN_DELTA"
  )
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
    "${curriculum_policy_arg[@]}" \
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
        --curriculum-policy "$CURRICULUM_POLICY"
        --curriculum-min-clusters "$CURRICULUM_MIN_CLUSTERS"
        --curriculum-max-levels "$CURRICULUM_MAX_LEVELS"
        --curriculum-stage-min-epochs "$CURRICULUM_STAGE_MIN_EPOCHS"
        --curriculum-stage-max-epochs "$CURRICULUM_STAGE_MAX_EPOCHS"
        --curriculum-stage-patience "$CURRICULUM_STAGE_PATIENCE"
        --curriculum-stage-min-delta "$CURRICULUM_STAGE_MIN_DELTA"
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

run_hierarchy_ablation() {
  echo "[entrypoint] Running hierarchy-source ablation"
  echo "[entrypoint] specs=$HIERARCHY_ABLATION_SPECS seeds=$HIERARCHY_ABLATION_SEEDS random_hierarchy_seeds=$HIERARCHY_ABLATION_RANDOM_SEEDS"

  local optimizer_value="${OPTIMIZER:-adam}"
  local scheduler_value="${SCHEDULER:-none}"
  local lr_value="${LR:-0.001}"
  local batch_size_value="${BATCH_SIZE:-128}"
  local val_ratio_value="${VAL_RATIO:-0.1}"
  local weight_decay_value="${WEIGHT_DECAY:-0.0}"

  local wandb_arg=()
  if wandb_enabled; then
    wandb_arg+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$HIERARCHY_ABLATION_WANDB_GROUP")
    wandb_arg+=(--wandb-tags "$HIERARCHY_ABLATION_WANDB_TAGS")
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

  IFS=',' read -r -a seeds <<< "$HIERARCHY_ABLATION_SEEDS"
  IFS=',' read -r -a random_seeds <<< "$HIERARCHY_ABLATION_RANDOM_SEEDS"
  IFS=',' read -r -a specs <<< "$HIERARCHY_ABLATION_SPECS"
  for spec in "${specs[@]}"; do
    spec="$(echo "$spec" | xargs)"
    [[ -z "$spec" ]] && continue
    IFS=':' read -r spec_dataset spec_model spec_cnn_width spec_cifar_width spec_curr_epochs spec_epochs <<< "$spec"
    if [[ -z "${spec_dataset:-}" || -z "${spec_model:-}" || -z "${spec_cnn_width:-}" || -z "${spec_cifar_width:-}" || -z "${spec_curr_epochs:-}" || -z "${spec_epochs:-}" ]]; then
      echo "[entrypoint] Invalid HIERARCHY_ABLATION_SPECS item: $spec"
      echo "[entrypoint] Expected dataset:model:cnn_width:cifar_width:curriculum_epochs:epochs"
      exit 1
    fi

    local token
    token="$(model_token_for_spec "$spec_model" "$spec_cnn_width" "$spec_cifar_width")"

    for seed in "${seeds[@]}"; do
      seed="$(echo "$seed" | xargs)"
      [[ -z "$seed" ]] && continue
      local prefix="hier-${spec_dataset}-${token}-seed${seed}"
      local baseline_id="${prefix}-baseline"
      local learned_id="${prefix}-learned-curr${spec_curr_epochs}"
      local baseline_dir="$OUTPUT_DIR/$baseline_id/${spec_dataset}_${spec_model}_baseline"
      local learned_dir="$OUTPUT_DIR/$learned_id/${spec_dataset}_${spec_model}_curriculum"

      local base_args=(
        --dataset "$spec_dataset"
        --model "$spec_model"
        --epochs "$spec_epochs"
        --val_ratio "$val_ratio_value"
        --optimizer "$optimizer_value"
        --scheduler "$scheduler_value"
        --lr "$lr_value"
        --weight_decay "$weight_decay_value"
        --batch_size "$batch_size_value"
        --dropout "$DROPOUT"
        --cnn-width-multiplier "$spec_cnn_width"
        --cifar-resnet-width-multiplier "$spec_cifar_width"
        --patience "$HIERARCHY_ABLATION_PATIENCE"
        --curriculum-policy fixed
        --curriculum-min-clusters "$CURRICULUM_MIN_CLUSTERS"
        --curriculum-max-levels "$CURRICULUM_MAX_LEVELS"
        --data_dir "$DATA_DIR"
        --output_dir "$OUTPUT_DIR"
        --num_workers "$NUM_WORKERS"
        --seed "$seed"
        "${deterministic_arg[@]}"
        "${download_arg[@]}"
        "${checkpoint_arg[@]}"
        "${amp_arg[@]}"
        "${wandb_arg[@]}"
      )
      [[ -n "$SHAPES_PATH" ]] && base_args+=(--shapes_path "$SHAPES_PATH")
      [[ -n "$TINY_IMAGENET_PATH" ]] && base_args+=(--tiny_imagenet_path "$TINY_IMAGENET_PATH")

      echo "[entrypoint] Hierarchy ablation baseline: dataset=$spec_dataset seed=$seed model=$token"
      if [[ -f "$baseline_dir/results.json" ]]; then
        echo "[entrypoint] Skipping completed baseline: $baseline_id"
      else
        python train_coarse_to_fine.py \
          --mode baseline \
          --run_id "$baseline_id" \
          --distance_source classifier_weights \
          "${base_args[@]}"
      fi

      echo "[entrypoint] Hierarchy ablation learned curriculum: dataset=$spec_dataset seed=$seed model=$token curr=$spec_curr_epochs"
      if [[ -f "$learned_dir/results.json" ]]; then
        echo "[entrypoint] Skipping completed learned curriculum: $learned_id"
      else
        python train_coarse_to_fine.py \
          --mode curriculum \
          --run_id "$learned_id" \
          --curriculum_epochs "$spec_curr_epochs" \
          --distance_source classifier_weights \
          --reference_run_dir "$baseline_dir" \
          "${base_args[@]}"
      fi

      for random_seed in "${random_seeds[@]}"; do
        random_seed="$(echo "$random_seed" | xargs)"
        [[ -z "$random_seed" ]] && continue
        local random_id="${prefix}-random${random_seed}-curr${spec_curr_epochs}"
        local random_dir="$OUTPUT_DIR/$random_id/${spec_dataset}_${spec_model}_curriculum"
        echo "[entrypoint] Hierarchy ablation random curriculum: dataset=$spec_dataset seed=$seed model=$token curr=$spec_curr_epochs random_seed=$random_seed"
        if [[ -f "$random_dir/results.json" ]]; then
          echo "[entrypoint] Skipping completed random curriculum: $random_id"
        else
          python train_coarse_to_fine.py \
            --mode curriculum \
            --run_id "$random_id" \
            --curriculum_epochs "$spec_curr_epochs" \
            --distance_source random_permutation \
            --random-hierarchy-seed "$random_seed" \
            --reference_run_dir "$baseline_dir" \
            "${base_args[@]}"
        fi
      done
    done
  done

  python scripts/analyze_results.py "$OUTPUT_DIR" || true
  python scripts/analyze_pareto.py "$OUTPUT_DIR" || true
}

run_teacher_hierarchy_suite() {
  echo "[entrypoint] Running teacher hierarchy suite"
  if [[ -z "$TEACHER_RUN_DIR" && -z "$TEACHER_CHECKPOINT_PATH" && "$TEACHER_PRETRAINED_SOURCE" != "torchvision_imagenet" ]]; then
    echo "[entrypoint] TEACHER_RUN_DIR, TEACHER_CHECKPOINT_PATH, or TEACHER_PRETRAINED_SOURCE=torchvision_imagenet is required for EXPERIMENT=teacher_hierarchy_suite"
    exit 1
  fi

  local optimizer_value="${OPTIMIZER:-adam}"
  local scheduler_value="${SCHEDULER:-none}"
  local lr_value="${LR:-0.001}"
  local batch_size_value="${BATCH_SIZE:-128}"
  local val_ratio_value="${VAL_RATIO:-0.1}"
  local weight_decay_value="${WEIGHT_DECAY:-0.0}"

  local wandb_arg=()
  if wandb_enabled; then
    wandb_arg+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$TEACHER_HIERARCHY_WANDB_GROUP")
    wandb_arg+=(--wandb-tags "$TEACHER_HIERARCHY_WANDB_TAGS")
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

  IFS=',' read -r -a seeds <<< "$TEACHER_HIERARCHY_SEEDS"
  IFS=',' read -r -a random_seeds <<< "$TEACHER_HIERARCHY_RANDOM_SEEDS"
  IFS=',' read -r -a specs <<< "$TEACHER_HIERARCHY_SPECS"
  for spec in "${specs[@]}"; do
    spec="$(echo "$spec" | xargs)"
    [[ -z "$spec" ]] && continue
    IFS=':' read -r spec_dataset spec_model spec_cnn_width spec_cifar_width spec_curr_epochs spec_epochs <<< "$spec"
    if [[ -z "${spec_dataset:-}" || -z "${spec_model:-}" || -z "${spec_cnn_width:-}" || -z "${spec_cifar_width:-}" || -z "${spec_curr_epochs:-}" || -z "${spec_epochs:-}" ]]; then
      echo "[entrypoint] Invalid TEACHER_HIERARCHY_SPECS item: $spec"
      echo "[entrypoint] Expected dataset:model:cnn_width:cifar_width:curriculum_epochs:epochs"
      exit 1
    fi

    local token
    token="$(model_token_for_spec "$spec_model" "$spec_cnn_width" "$spec_cifar_width")"

    for seed in "${seeds[@]}"; do
      seed="$(echo "$seed" | xargs)"
      [[ -z "$seed" ]] && continue
      local reference_prefix="${TEACHER_HIERARCHY_REFERENCE_PREFIX}-${spec_dataset}-${token}-seed${seed}"
      local output_prefix="${TEACHER_HIERARCHY_OUTPUT_PREFIX}-${spec_dataset}-${token}-seed${seed}"
      local reference_baseline_id="${reference_prefix}-baseline"
      local baseline_id="${output_prefix}-baseline"
      local learned_id="${output_prefix}-self-curr${spec_curr_epochs}"
      local teacher_id="${output_prefix}-teacher-curr${spec_curr_epochs}"
      local anti_id="${output_prefix}-teacher-anti-curr${spec_curr_epochs}"
      local baseline_output_dir="$OUTPUT_DIR/$baseline_id/${spec_dataset}_${spec_model}_baseline"
      local baseline_dir="$OUTPUT_DIR/$reference_baseline_id/${spec_dataset}_${spec_model}_baseline"
      local learned_dir="$OUTPUT_DIR/$learned_id/${spec_dataset}_${spec_model}_curriculum"
      local teacher_dir="$OUTPUT_DIR/$teacher_id/${spec_dataset}_${spec_model}_curriculum"
      local anti_dir="$OUTPUT_DIR/$anti_id/${spec_dataset}_${spec_model}_curriculum"

      local base_args=(
        --dataset "$spec_dataset"
        --model "$spec_model"
        --epochs "$spec_epochs"
        --val_ratio "$val_ratio_value"
        --optimizer "$optimizer_value"
        --scheduler "$scheduler_value"
        --lr "$lr_value"
        --weight_decay "$weight_decay_value"
        --batch_size "$batch_size_value"
        --dropout "$DROPOUT"
        --patience "$PATIENCE"
        --cnn-width-multiplier "$spec_cnn_width"
        --cifar-resnet-width-multiplier "$spec_cifar_width"
        --curriculum-policy fixed
        --curriculum-min-clusters "$CURRICULUM_MIN_CLUSTERS"
        --curriculum-max-levels "$CURRICULUM_MAX_LEVELS"
        --curriculum-stage-min-epochs "$CURRICULUM_STAGE_MIN_EPOCHS"
        --curriculum-stage-max-epochs "$CURRICULUM_STAGE_MAX_EPOCHS"
        --curriculum-stage-patience "$CURRICULUM_STAGE_PATIENCE"
        --curriculum-stage-min-delta "$CURRICULUM_STAGE_MIN_DELTA"
        --data_dir "$DATA_DIR"
        --output_dir "$OUTPUT_DIR"
        --num_workers "$NUM_WORKERS"
        --seed "$seed"
        --teacher_embedding_split "$TEACHER_EMBEDDING_SPLIT"
        "${deterministic_arg[@]}"
        "${download_arg[@]}"
        "${checkpoint_arg[@]}"
        "${amp_arg[@]}"
        "${wandb_arg[@]}"
      )
      [[ -n "$SHAPES_PATH" ]] && base_args+=(--shapes_path "$SHAPES_PATH")
      [[ -n "$TINY_IMAGENET_PATH" ]] && base_args+=(--tiny_imagenet_path "$TINY_IMAGENET_PATH")
      [[ -n "$TEACHER_RUN_DIR" ]] && base_args+=(--teacher_run_dir "$TEACHER_RUN_DIR")
      [[ -n "$TEACHER_CHECKPOINT_PATH" ]] && base_args+=(--teacher_checkpoint_path "$TEACHER_CHECKPOINT_PATH")
      [[ -n "$TEACHER_MODEL" ]] && base_args+=(--teacher_model "$TEACHER_MODEL")
      base_args+=(--teacher_pretrained_source "$TEACHER_PRETRAINED_SOURCE")
      [[ -n "$TEACHER_CNN_WIDTH_MULTIPLIER" ]] && base_args+=(--teacher_cnn_width_multiplier "$TEACHER_CNN_WIDTH_MULTIPLIER")
      [[ -n "$TEACHER_CIFAR_RESNET_WIDTH_MULTIPLIER" ]] && base_args+=(--teacher_cifar_resnet_width_multiplier "$TEACHER_CIFAR_RESNET_WIDTH_MULTIPLIER")

      if teacher_condition_enabled baseline; then
        echo "[entrypoint] Teacher suite baseline: dataset=$spec_dataset seed=$seed model=$token"
        if [[ -f "$baseline_output_dir/results.json" ]]; then
          echo "[entrypoint] Skipping completed baseline: $baseline_id"
        else
          python train_coarse_to_fine.py --mode baseline --run_id "$baseline_id" --distance_source classifier_weights "${base_args[@]}"
        fi
        baseline_dir="$baseline_output_dir"
      else
        echo "[entrypoint] Teacher suite baseline disabled; using reference baseline: $baseline_dir"
        if [[ ! -f "$baseline_dir/history.json" ]]; then
          echo "[entrypoint] Missing reference baseline history: $baseline_dir/history.json"
          exit 1
        fi
      fi

      if teacher_condition_enabled self; then
        echo "[entrypoint] Teacher suite self hierarchy curriculum: dataset=$spec_dataset seed=$seed model=$token curr=$spec_curr_epochs"
        if [[ -f "$learned_dir/results.json" ]]; then
          echo "[entrypoint] Skipping completed self hierarchy curriculum: $learned_id"
        else
          python train_coarse_to_fine.py \
            --mode curriculum \
            --run_id "$learned_id" \
            --curriculum_epochs "$spec_curr_epochs" \
            --distance_source classifier_weights \
            --reference_run_dir "$baseline_dir" \
            --curriculum-order easy_to_hard \
            "${base_args[@]}"
        fi
      fi

      if teacher_condition_enabled teacher; then
        echo "[entrypoint] Teacher suite teacher hierarchy curriculum: dataset=$spec_dataset seed=$seed model=$token curr=$spec_curr_epochs"
        if [[ -f "$teacher_dir/results.json" ]]; then
          echo "[entrypoint] Skipping completed teacher hierarchy curriculum: $teacher_id"
        else
          python train_coarse_to_fine.py \
            --mode curriculum \
            --run_id "$teacher_id" \
            --curriculum_epochs "$spec_curr_epochs" \
            --distance_source teacher_embeddings \
            --reference_run_dir "$baseline_dir" \
            --curriculum-order easy_to_hard \
            "${base_args[@]}"
        fi
      fi

      if teacher_condition_enabled teacher_anti; then
        echo "[entrypoint] Teacher suite anti-curriculum: dataset=$spec_dataset seed=$seed model=$token curr=$spec_curr_epochs"
        if [[ -f "$anti_dir/results.json" ]]; then
          echo "[entrypoint] Skipping completed anti-curriculum: $anti_id"
        else
          python train_coarse_to_fine.py \
            --mode curriculum \
            --run_id "$anti_id" \
            --curriculum_epochs "$spec_curr_epochs" \
            --distance_source teacher_embeddings \
            --reference_run_dir "$baseline_dir" \
            --curriculum-order hard_to_easy \
            "${base_args[@]}"
        fi
      fi

      if teacher_condition_enabled random; then
        for random_seed in "${random_seeds[@]}"; do
          random_seed="$(echo "$random_seed" | xargs)"
          [[ -z "$random_seed" ]] && continue
          local random_id="${output_prefix}-random${random_seed}-curr${spec_curr_epochs}"
          local random_dir="$OUTPUT_DIR/$random_id/${spec_dataset}_${spec_model}_curriculum"
          echo "[entrypoint] Teacher suite random hierarchy curriculum: dataset=$spec_dataset seed=$seed model=$token curr=$spec_curr_epochs random_seed=$random_seed"
          if [[ -f "$random_dir/results.json" ]]; then
            echo "[entrypoint] Skipping completed random hierarchy curriculum: $random_id"
          else
            python train_coarse_to_fine.py \
              --mode curriculum \
              --run_id "$random_id" \
              --curriculum_epochs "$spec_curr_epochs" \
              --distance_source random_permutation \
              --random-hierarchy-seed "$random_seed" \
              --reference_run_dir "$baseline_dir" \
              --curriculum-order easy_to_hard \
              "${base_args[@]}"
          fi
        done
      fi
    done
  done

  python scripts/analyze_pareto.py "$OUTPUT_DIR" || true
  python scripts/analyze_teacher_hierarchy_suite.py "$OUTPUT_DIR" || true
}

run_teacher_bootstrap_suite() {
  echo "[entrypoint] Running teacher bootstrap + hierarchy suite"

  local bootstrap_dir="$OUTPUT_DIR/$TEACHER_BOOTSTRAP_RUN_ID/${TEACHER_BOOTSTRAP_DATASET}_${TEACHER_BOOTSTRAP_MODEL}_baseline"
  local bootstrap_wandb_arg=()
  if wandb_enabled; then
    bootstrap_wandb_arg+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$TEACHER_BOOTSTRAP_WANDB_GROUP")
    bootstrap_wandb_arg+=(--wandb-tags "$TEACHER_BOOTSTRAP_WANDB_TAGS")
    [[ -n "$WANDB_ENTITY" ]] && bootstrap_wandb_arg+=(--wandb-entity "$WANDB_ENTITY")
  fi

  local deterministic_arg=()
  if is_truthy "$DETERMINISTIC"; then
    deterministic_arg=(--deterministic)
  else
    deterministic_arg=(--no-deterministic)
  fi

  local download_arg=(--download)
  is_truthy "$DOWNLOAD" || download_arg=(--no-download)

  local amp_arg=()
  [[ "$AMP" == "1" ]] && amp_arg=(--amp)

  local pretrained_arg=()
  if is_truthy "$TEACHER_BOOTSTRAP_PRETRAINED_BACKBONE"; then
    pretrained_arg=(--pretrained-backbone)
  fi

  local export_teacher_args=()
  if is_truthy "$TEACHER_BOOTSTRAP_EXPORT_HIERARCHY"; then
    export_teacher_args=(--export-teacher-hierarchy --export-teacher-hierarchy-split "$TEACHER_BOOTSTRAP_EXPORT_SPLIT")
  fi

  local dataset_path_args=()
  [[ -n "$SHAPES_PATH" ]] && dataset_path_args+=(--shapes_path "$SHAPES_PATH")
  [[ -n "$TINY_IMAGENET_PATH" ]] && dataset_path_args+=(--tiny_imagenet_path "$TINY_IMAGENET_PATH")

  if [[ -f "$bootstrap_dir/config.json" && -f "$bootstrap_dir/best_model.pt" ]]; then
    echo "[entrypoint] Skipping completed teacher bootstrap: $bootstrap_dir"
  else
    python train_coarse_to_fine.py \
      --mode baseline \
      --dataset "$TEACHER_BOOTSTRAP_DATASET" \
      --model "$TEACHER_BOOTSTRAP_MODEL" \
      --epochs "$TEACHER_BOOTSTRAP_EPOCHS" \
      --batch_size "$TEACHER_BOOTSTRAP_BATCH_SIZE" \
      --optimizer "$TEACHER_BOOTSTRAP_OPTIMIZER" \
      --scheduler "$TEACHER_BOOTSTRAP_SCHEDULER" \
      --lr "$TEACHER_BOOTSTRAP_LR" \
      --weight_decay "$TEACHER_BOOTSTRAP_WEIGHT_DECAY" \
      --val_ratio "$TEACHER_BOOTSTRAP_VAL_RATIO" \
      --data_dir "$DATA_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --run_id "$TEACHER_BOOTSTRAP_RUN_ID" \
      --num_workers "$NUM_WORKERS" \
      --seed "$TEACHER_BOOTSTRAP_SEED" \
      --distance_source classifier_weights \
      --save-checkpoints \
      "${deterministic_arg[@]}" \
      "${download_arg[@]}" \
      "${amp_arg[@]}" \
      "${pretrained_arg[@]}" \
      "${export_teacher_args[@]}" \
      "${dataset_path_args[@]}" \
      "${bootstrap_wandb_arg[@]}"
  fi

  export TEACHER_RUN_DIR="$bootstrap_dir"
  export TEACHER_CHECKPOINT_PATH=""
  export TEACHER_MODEL=""
  export TEACHER_PRETRAINED_SOURCE="none"
  run_teacher_hierarchy_suite
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
  if is_truthy "$PRETRAINED_BACKBONE"; then
    base_common+=(--pretrained-backbone)
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
  hierarchy_ablation)
    run_hierarchy_ablation
    ;;
  teacher_hierarchy_suite)
    run_teacher_hierarchy_suite
    ;;
  teacher_bootstrap_suite)
    run_teacher_bootstrap_suite
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
  elif [[ "$EXPERIMENT" == "hierarchy_ablation" ]]; then
    archive_base="hierarchy_ablation"
    IFS=',' read -r -a archive_hier_specs <<< "$HIERARCHY_ABLATION_SPECS"
    IFS=',' read -r -a archive_hier_seeds <<< "$HIERARCHY_ABLATION_SEEDS"
    IFS=',' read -r -a archive_hier_random_seeds <<< "$HIERARCHY_ABLATION_RANDOM_SEEDS"
    for spec in "${archive_hier_specs[@]}"; do
      spec="$(echo "$spec" | xargs)"
      [[ -z "$spec" ]] && continue
      IFS=':' read -r spec_dataset spec_model spec_cnn_width spec_cifar_width spec_curr_epochs _spec_epochs <<< "$spec"
      token="$(model_token_for_spec "$spec_model" "$spec_cnn_width" "$spec_cifar_width")"
      for seed in "${archive_hier_seeds[@]}"; do
        seed="$(echo "$seed" | xargs)"
        [[ -z "$seed" ]] && continue
        prefix="hier-${spec_dataset}-${token}-seed${seed}"
        archive_members+=("${prefix}-baseline")
        archive_members+=("${prefix}-learned-curr${spec_curr_epochs}")
        for random_seed in "${archive_hier_random_seeds[@]}"; do
          random_seed="$(echo "$random_seed" | xargs)"
          [[ -z "$random_seed" ]] && continue
          archive_members+=("${prefix}-random${random_seed}-curr${spec_curr_epochs}")
        done
      done
    done
    [[ -d "$OUTPUT_DIR/analysis" ]] && archive_members+=("analysis")
  elif [[ "$EXPERIMENT" == "teacher_hierarchy_suite" || "$EXPERIMENT" == "teacher_bootstrap_suite" ]]; then
    archive_base="$EXPERIMENT"
    if [[ "$EXPERIMENT" == "teacher_bootstrap_suite" ]]; then
      archive_members+=("$TEACHER_BOOTSTRAP_RUN_ID")
    fi
    IFS=',' read -r -a archive_teacher_specs <<< "$TEACHER_HIERARCHY_SPECS"
    IFS=',' read -r -a archive_teacher_seeds <<< "$TEACHER_HIERARCHY_SEEDS"
    IFS=',' read -r -a archive_teacher_random_seeds <<< "$TEACHER_HIERARCHY_RANDOM_SEEDS"
    for spec in "${archive_teacher_specs[@]}"; do
      spec="$(echo "$spec" | xargs)"
      [[ -z "$spec" ]] && continue
      IFS=':' read -r spec_dataset spec_model spec_cnn_width spec_cifar_width spec_curr_epochs _spec_epochs <<< "$spec"
      token="$(model_token_for_spec "$spec_model" "$spec_cnn_width" "$spec_cifar_width")"
      for seed in "${archive_teacher_seeds[@]}"; do
        seed="$(echo "$seed" | xargs)"
        [[ -z "$seed" ]] && continue
        prefix="${TEACHER_HIERARCHY_OUTPUT_PREFIX}-${spec_dataset}-${token}-seed${seed}"
        teacher_condition_enabled baseline && archive_members+=("${prefix}-baseline")
        teacher_condition_enabled self && archive_members+=("${prefix}-self-curr${spec_curr_epochs}")
        teacher_condition_enabled teacher && archive_members+=("${prefix}-teacher-curr${spec_curr_epochs}")
        teacher_condition_enabled teacher_anti && archive_members+=("${prefix}-teacher-anti-curr${spec_curr_epochs}")
        if teacher_condition_enabled random; then
          for random_seed in "${archive_teacher_random_seeds[@]}"; do
            random_seed="$(echo "$random_seed" | xargs)"
            [[ -z "$random_seed" ]] && continue
            archive_members+=("${prefix}-random${random_seed}-curr${spec_curr_epochs}")
          done
        fi
      done
    done
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

