#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
LOG_DIR="${ROOT_DIR}/reports/logs"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_LOG="${LOG_DIR}/meta_weighting_eggholder_${TIMESTAMP}.log"
SUMMARY_LOG="${LOG_DIR}/meta_weighting_eggholder_${TIMESTAMP}_summary.log"

SEEDS=(42 777 999)
NOISES=(0 0.05 0.2)

mkdir -p "${LOG_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${PYTHON_BIN}. Run 'make requirements' first."
  exit 1
fi

echo "Meta-weighting eggholder pack started at $(date -Iseconds)" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"
echo "Logs: ${RUN_LOG}" | tee -a "${SUMMARY_LOG}"

for seed in "${SEEDS[@]}"; do
  for noise in "${NOISES[@]}"; do
    noise_token="${noise/./p}"
    input_path="data/processed/eggholder_n12000_k5_ss5_nr${noise_token}_seed${seed}_train.parquet"
    run_name="meta_weighting_eggholder_u1_uniforminit_${TIMESTAMP}__seed${seed}__eggholder__losses5__lrm0.01__lru0.01__unroll1__noise${noise}"

    echo "" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"
    echo "=== Running seed=${seed} noise=${noise} ===" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"

    if "${PYTHON_BIN}" -m ma_thesis.experiment run \
      --method meta \
      --function eggholder \
      --input-path "${input_path}" \
      --num-samples 12000 \
      --num-sigmas 5 \
      --sigma-scale 5 \
      --noise-ratio "${noise}" \
      --epochs 250 \
      --batch-size 128 \
      --model-arch fourier \
      --hidden-dim 16 \
      --num-blocks 1 \
      --activation silu \
      --num-fourier 16 \
      --min-train-per-param 10 \
      --lr-model 0.01 \
      --lr-meta 0.01 \
      --momentum 0.9 \
      --lr-decay-gamma 0.999 \
      --grad-clip-norm 1.0 \
      --inner-steps 10 \
      --meta-unroll-steps 1 \
      --meta-num-losses 5 \
      --meta-val-samples 2000 \
      --seed "${seed}" \
      --experiment-name meta-weighting-v1-eggholder \
      --run-name "${run_name}" \
      >>"${RUN_LOG}" 2>&1; then
      echo "OK  seed=${seed} noise=${noise}" | tee -a "${SUMMARY_LOG}"
    else
      echo "FAIL seed=${seed} noise=${noise} (continuing)" | tee -a "${SUMMARY_LOG}"
    fi
  done
done

echo "" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"
echo "Meta-weighting eggholder pack finished at $(date -Iseconds)" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"
echo "Summary: ${SUMMARY_LOG}" | tee -a "${RUN_LOG}"
