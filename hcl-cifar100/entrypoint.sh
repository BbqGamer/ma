#!/usr/bin/env bash
set -euo pipefail

RUN_MODES="${RUN_MODES:-baseline,hcl}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.1}"
THRESH="${THRESH:-50.0}"
SELECT_FRAC="${SELECT_FRAC:-0.9}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/runs}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
AMP="${AMP:-1}"
DOWNLOAD="${DOWNLOAD:-1}"

common_args=(
  --run_id "$RUN_ID"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --thresh "$THRESH"
  --select_frac "$SELECT_FRAC"
  --data_dir "$DATA_DIR"
  --output_dir "$OUTPUT_DIR"
  --num_workers "$NUM_WORKERS"
  --seed "$SEED"
)

if [[ "$AMP" == "1" ]]; then
  common_args+=(--amp)
fi

if [[ "$DOWNLOAD" == "1" ]]; then
  common_args+=(--download)
else
  common_args+=(--no-download)
fi

IFS=',' read -r -a modes <<< "$RUN_MODES"
for mode in "${modes[@]}"; do
  mode="$(echo "$mode" | xargs)"
  [[ -z "$mode" ]] && continue
  echo "[entrypoint] Running mode: $mode"
  python train_hcl.py --mode "$mode" "${common_args[@]}"
done
