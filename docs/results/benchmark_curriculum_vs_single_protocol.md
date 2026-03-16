# Curriculum vs Single Benchmark Protocol

This benchmark compares `single` vs `curriculum` training on all sweep-supported
functions under two fairness regimes:

- `equal_epochs`: same epoch budget for both methods.
- `equal_time`: same wall-clock timeout for both methods.

## Methods and Scope

- Methods: `single` (`sigma_level=-1`) vs `curriculum`.
- Functions: loaded from `configs/sweeps/*_sweep.yaml`.
- Seeds (default): `42,43,44`.
- Hyperparameters: loaded from per-function sweep best trial using:
  - `--from-sweep <reports/optuna/sweep_..._v1.db>`
  - `--study-name <sweep-...-v1>`

## Run

```bash
./scripts/run_curriculum_benchmark_pack.sh
```

Optional manual run:

```bash
.venv/bin/python scripts/run_curriculum_benchmark.py \
  --benchmark-id curr_vs_single_YYYYMMDD_HHMMSS \
  --equal-time-cap-seconds 1800
```

Outputs:

- `reports/benchmarks/<benchmark_id>_manifest.json`
- `reports/benchmarks/<benchmark_id>_runs.csv`
- `reports/logs/benchmark_<benchmark_id>.log`

## Analyze

```bash
.venv/bin/python scripts/analyze_curriculum_benchmark.py --benchmark-id <benchmark_id>
```

Outputs:

- `reports/benchmarks/<benchmark_id>/runs_joined.csv`
- `reports/benchmarks/<benchmark_id>/aggregate.csv`
- `reports/benchmarks/<benchmark_id>/delta.csv`
- `reports/benchmarks/<benchmark_id>/time_to_target.csv`
- `reports/benchmarks/<benchmark_id>/summary.md`

## Decision Metrics

Primary quality metric:

- `final_hard_val_loss`

Efficiency metrics:

- `runtime_sec` (runner wall-clock, fallback to MLflow start/end)
- `time_to_target` where target is single-method median quality per function/regime

Key comparison columns in `delta.csv`:

- `delta_quality_single_minus_curriculum` (positive favors curriculum)
- `delta_runtime_sec_single_minus_curriculum` (positive means curriculum slower)
