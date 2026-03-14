#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
LOG_DIR="${ROOT_DIR}/reports/logs"
OPTUNA_DIR="${ROOT_DIR}/reports/optuna"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_LOG="${LOG_DIR}/sweep_pack_${TIMESTAMP}.log"
SUMMARY_LOG="${LOG_DIR}/sweep_pack_${TIMESTAMP}_summary.log"

CONFIGS=(
  "configs/sweeps/ackley_sweep.yaml"
  "configs/sweeps/levy_sweep.yaml"
  "configs/sweeps/bukin_sweep.yaml"
  "configs/sweeps/eggholder_sweep.yaml"
  "configs/sweeps/franke_sweep.yaml"
  "configs/sweeps/peaks_sweep.yaml"
  "configs/sweeps/friedman1_2d_sweep.yaml"
  "configs/sweeps/friedman2_2d_sweep.yaml"
)

mkdir -p "${LOG_DIR}" "${OPTUNA_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${PYTHON_BIN}. Run 'make requirements' first."
  exit 1
fi

echo "Sweep pack started at $(date -Iseconds)" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"
echo "Logs: ${RUN_LOG}" | tee -a "${SUMMARY_LOG}"

for cfg in "${CONFIGS[@]}"; do
  echo "" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"
  echo "=== Running ${cfg} ===" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"

  if "${PYTHON_BIN}" -m ma_thesis.experiment run-config "${ROOT_DIR}/${cfg}" >>"${RUN_LOG}" 2>&1; then
    echo "OK  ${cfg}" | tee -a "${SUMMARY_LOG}"
  else
    echo "FAIL ${cfg} (continuing)" | tee -a "${SUMMARY_LOG}"
  fi
done

echo "" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"
echo "Sweep pack finished at $(date -Iseconds)" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"
echo "Summary: ${SUMMARY_LOG}" | tee -a "${RUN_LOG}"
