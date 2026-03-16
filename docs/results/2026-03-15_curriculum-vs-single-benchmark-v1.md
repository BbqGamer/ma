# Curriculum vs Single Benchmark v1 (2026-03-15/16)

Canonical record for the full benchmark pack comparing `single` and `curriculum`.

## Run Identity

- Benchmark ID: `curr_vs_single_20260315_175826`
- Start: `2026-03-15T17:58:26+01:00`
- End: `2026-03-16T01:10:36+01:00`
- Runner status: `96/96` OK
- MLflow status: `96/96` FINISHED

## Inputs

- Methods: `single` vs `curriculum`
- Regimes: `equal_epochs`, `equal_time`
- Seeds: `42,43,44`
- Functions: `ackley, levy, bukin, eggholder, franke, peaks, friedman1_2d, friedman2_2d`
- Hyperparameters source: per-function best sweep trial (`from_sweep`)

## Artifact Paths

- Run table: `reports/benchmarks/curr_vs_single_20260315_175826_runs.csv`
- Joined metrics: `reports/benchmarks/curr_vs_single_20260315_175826/runs_joined.csv`
- Aggregate: `reports/benchmarks/curr_vs_single_20260315_175826/aggregate.csv`
- Delta: `reports/benchmarks/curr_vs_single_20260315_175826/delta.csv`
- Time-to-target: `reports/benchmarks/curr_vs_single_20260315_175826/time_to_target.csv`
- Summary log: `reports/logs/benchmark_pack_20260315_175826_summary.log`

## Headline Results

| Regime | Quality winner count (curriculum) | Quality winner count (single) | Runtime winner count (single) | Runtime winner count (curriculum) |
|---|---:|---:|---:|---:|
| equal_epochs | 5 | 3 | 8 | 0 |
| equal_time | 3 | 5 | 8 | 0 |

- Overall quality wins: curriculum `8/16`, single `8/16` (mixed).
- Overall runtime wins: single `16/16` (curriculum slower in every function/regime cell).

## Quality Winner by Function

| Function | equal_epochs | equal_time | Runtime winner (both regimes) |
|---|---|---|---|
| ackley | single | single | single |
| bukin | curriculum | curriculum | single |
| eggholder | single | single | single |
| franke | curriculum | curriculum | single |
| friedman1_2d | curriculum | single | single |
| friedman2_2d | curriculum | single | single |
| levy | curriculum | curriculum | single |
| peaks | single | single | single |

## Interpretation

- Curriculum is not a global default winner: it improves quality for selected functions but loses on others.
- Runtime penalty is consistent and large across all tasks.
- Recommended strategy: apply curriculum selectively (`bukin`, `franke`, `levy`), keep single as default elsewhere.
- Important caveat: `equal_time` used a high timeout cap, so in many runs it did not become a strict time-budget constraint.
