#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
LOG_DIR="${ROOT_DIR}/reports/logs"
BENCHMARK_DIR="${ROOT_DIR}/reports/benchmarks"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
BENCHMARK_ID="curr_vs_single_${TIMESTAMP}"
RUN_LOG="${LOG_DIR}/benchmark_pack_${TIMESTAMP}.log"
SUMMARY_LOG="${LOG_DIR}/benchmark_pack_${TIMESTAMP}_summary.log"

mkdir -p "${LOG_DIR}" "${BENCHMARK_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${PYTHON_BIN}. Run 'make requirements' first."
  exit 1
fi

echo "Curriculum benchmark pack started at $(date -Iseconds)" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"
echo "Benchmark ID: ${BENCHMARK_ID}" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"
echo "Logs: ${RUN_LOG}" | tee -a "${SUMMARY_LOG}"

echo "" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"
echo "=== Running benchmark matrix (single vs curriculum; equal_epochs + equal_time) ===" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"

if "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_curriculum_benchmark.py" --benchmark-id "${BENCHMARK_ID}" "$@" >>"${RUN_LOG}" 2>&1; then
  echo "OK  benchmark matrix" | tee -a "${SUMMARY_LOG}"
else
  echo "FAIL benchmark matrix" | tee -a "${SUMMARY_LOG}"
  echo "Benchmark pack finished at $(date -Iseconds)" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"
  exit 1
fi

echo "" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"
echo "=== Analyzing benchmark results ===" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"

if "${PYTHON_BIN}" "${ROOT_DIR}/scripts/analyze_curriculum_benchmark.py" --benchmark-id "${BENCHMARK_ID}" >>"${RUN_LOG}" 2>&1; then
  echo "OK  benchmark analysis" | tee -a "${SUMMARY_LOG}"
else
  echo "FAIL benchmark analysis" | tee -a "${SUMMARY_LOG}"
  echo "Benchmark pack finished at $(date -Iseconds)" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"
  exit 1
fi

echo "" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"
echo "Benchmark pack finished at $(date -Iseconds)" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"
echo "Summary: ${SUMMARY_LOG}" | tee -a "${RUN_LOG}"
echo "Outputs: ${BENCHMARK_DIR}/${BENCHMARK_ID}" | tee -a "${RUN_LOG}" "${SUMMARY_LOG}"
