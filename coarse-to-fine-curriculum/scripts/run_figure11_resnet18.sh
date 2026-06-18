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

cd /workspace

python scripts/plan_figure11_resnet18.py \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --val-ratio "$VAL_RATIO" \
  --optimizer "$OPTIMIZER" \
  --scheduler "$SCHEDULER" \
  --lr "$LR" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR"

bash figure11_resnet18_commands.sh

python scripts/plot_figure11_resnet18.py \
  "$OUTPUT_DIR" \
  --seed "$SEED" \
  --metric "$FIG11_METRIC"

python scripts/analyze_results.py "$OUTPUT_DIR"

if [[ "$AUTO_STOP_POD" == "1" && -n "${RUNPOD_POD_ID:-}" ]]; then
  echo "[run_figure11_resnet18] Sweep finished; stopping Runpod pod ${RUNPOD_POD_ID}"
  if command -v runpodctl >/dev/null 2>&1; then
    runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl remove pod "$RUNPOD_POD_ID" || true
  else
    echo "[run_figure11_resnet18] runpodctl not found; cannot auto-stop pod"
  fi
fi
