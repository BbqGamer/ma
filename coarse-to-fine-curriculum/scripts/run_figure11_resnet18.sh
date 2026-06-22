#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-200}"
VAL_RATIO="${VAL_RATIO:-0.1}"
OPTIMIZER="${OPTIMIZER:-adam}"
SCHEDULER="${SCHEDULER:-none}"
LR="${LR:-0.001}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/runs}"
FIG11_METRIC="${FIG11_METRIC:-test_acc}"
AUTO_STOP_POD="${AUTO_STOP_POD:-1}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
ARCHIVE_OUTPUTS="${ARCHIVE_OUTPUTS:-1}"
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-coarse-to-fine-curriculum}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-fig11-resnet18-cifar100-seed${SEED}}"
WANDB_TAGS="${WANDB_TAGS:-runpod,figure11}"

cd /workspace

wandb_args=()
if [[ "$WANDB" == "1" ]]; then
  wandb_args+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP")
  wandb_args+=(--wandb-tags "$WANDB_TAGS")
  [[ -n "$WANDB_ENTITY" ]] && wandb_args+=(--wandb-entity "$WANDB_ENTITY")
fi

python scripts/plan_figure11_resnet18.py \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --val-ratio "$VAL_RATIO" \
  --optimizer "$OPTIMIZER" \
  --scheduler "$SCHEDULER" \
  --lr "$LR" \
  "${wandb_args[@]}" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR"

if [[ "$SAVE_CHECKPOINTS" != "1" ]]; then
  python - <<'PY'
from pathlib import Path
p = Path('figure11_resnet18_commands.sh')
text = p.read_text()
text = text.replace('python train_coarse_to_fine.py', 'python train_coarse_to_fine.py --no-save-checkpoints')
p.write_text(text)
PY
fi

bash figure11_resnet18_commands.sh

python scripts/plot_figure11_resnet18.py \
  "$OUTPUT_DIR" \
  --seed "$SEED" \
  --metric "$FIG11_METRIC"

python scripts/analyze_results.py "$OUTPUT_DIR"

if [[ "$ARCHIVE_OUTPUTS" == "1" ]]; then
  archive_path="$OUTPUT_DIR/figure11_resnet18-seed${SEED}.tar.gz"
  members=(
    "fig11-resnet18-cifar100-seed${SEED}-baseline"
    "fig11-resnet18-cifar100-seed${SEED}-curr5"
    "fig11-resnet18-cifar100-seed${SEED}-curr10"
    "fig11-resnet18-cifar100-seed${SEED}-curr20"
    "fig11-resnet18-cifar100-seed${SEED}-curr30"
    "fig11-resnet18-cifar100-seed${SEED}-curr40"
    "fig11-resnet18-cifar100-seed${SEED}-curr50"
    "fig11-resnet18-cifar100-seed${SEED}-figure11-analysis"
  )
  existing=()
  for member in "${members[@]}"; do
    [[ -e "$OUTPUT_DIR/$member" ]] && existing+=("$member")
  done
  if [[ ${#existing[@]} -gt 0 ]]; then
    tar -czf "$archive_path" -C "$OUTPUT_DIR" "${existing[@]}"
    echo "[run_figure11_resnet18] Created archive $archive_path"
  fi
fi

if [[ "$AUTO_STOP_POD" == "1" && -n "${RUNPOD_POD_ID:-}" ]]; then
  echo "[run_figure11_resnet18] Sweep finished; stopping Runpod pod ${RUNPOD_POD_ID}"
  if command -v runpodctl >/dev/null 2>&1; then
    runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl remove pod "$RUNPOD_POD_ID" || true
  else
    echo "[run_figure11_resnet18] runpodctl not found; cannot auto-stop pod"
  fi
fi
