# Sweep Pack v1 Archive (2026-03-14/15)

Canonical, immutable record for the sweep pack executed via `scripts/run_sweep_pack.sh`.

## Run Identity

- Archive ID: `2026-03-14_sweep-pack-v1`
- Sweep start: `2026-03-14T15:15:21+01:00`
- Sweep end: `2026-03-15T02:41:45+01:00`
- Git commit (pre-archive): `bf8f54969c16dd7f16ee57e87313f20fb0f9c99c`
- Branch: `master`
- Total trials: `630`
- Finished: `336`, Pruned/KILLED: `294`

## Inputs

- `configs/sweeps/ackley_sweep.yaml`
- `configs/sweeps/levy_sweep.yaml`
- `configs/sweeps/bukin_sweep.yaml`
- `configs/sweeps/eggholder_sweep.yaml`
- `configs/sweeps/franke_sweep.yaml`
- `configs/sweeps/peaks_sweep.yaml`
- `configs/sweeps/friedman1_2d_sweep.yaml`
- `configs/sweeps/friedman2_2d_sweep.yaml`

## Artifact Paths

- Summary log: `reports/logs/sweep_pack_20260314_151521_summary.log`
- Full log: `reports/logs/sweep_pack_20260314_151521.log`
- Sweep manifests: `reports/runs/*_sweep.json`
- Optuna DBs: `reports/optuna/sweep_*_v1.db`
- Best runs CSV: `reports/results/2026-03-14_sweep-pack-v1_best_runs.csv`
- Trial stats CSV: `reports/results/2026-03-14_sweep-pack-v1_trial_stats.csv`
- Metadata JSON: `reports/results/2026-03-14_sweep-pack-v1_metadata.json`
- Environment snapshot: `reports/results/2026-03-14_sweep-pack-v1_env.txt`

## Best Trial per Function

| function | best_val_loss | median_best_val_loss | best_model_arch | killed_rate | best_run_uuid |
|---|---:|---:|---|---:|---|
| franke_n10000_k5_ss4_seed42 | 2.52614e-06 | 1.16191e-05 | mlp | 41.2% | `d9db287a52eb45038616af04e6c076bf` |
| friedman2_2d_n12000_k6_ss5_seed42 | 3.90145e-06 | 5.08304e-05 | mlp | 48.9% | `b3c062f08ded4849a2aa999023309e8f` |
| friedman1_2d_n12000_k5_ss4p5_seed42 | 1.0775e-05 | 0.000128475 | mlp | 40.0% | `5fb09dbd944947bda13f4501a9ab451f` |
| peaks_n10000_k5_ss5_seed42 | 3.63529e-05 | 0.000394481 | siren | 51.2% | `a145e516d6cf40c18dd32b1807ced76f` |
| bukin_n10000_k6_ss8_seed42 | 7.62519e-05 | 0.00744365 | mlp | 35.7% | `7bb6d59c06fd48b28c5ba87538593a38` |
| ackley_n8000_k5_ss5_seed42 | 0.000202372 | 0.288809 | mlp | 50.0% | `844714d91bcd4350ae06084bd33cb0f3` |
| levy_n9000_k5_ss6_seed42 | 0.00619497 | 1.03325 | mlp | 57.1% | `eb0d401b23594cc98e1aa1b1d037f80f` |
| eggholder_n12000_k6_ss7_seed42 | 60.2292 | 1458.07 | fourier | 48.9% | `21db0de20c274e8184876f56ce0a7eee` |

## Key Findings

- MLP is the best architecture for most functions in this pack (6/8).
- Best non-MLP outcomes: `peaks` (SIREN) and `eggholder` (Fourier).
- `eggholder` remains the hardest case by a wide margin in best and median loss.
- Pruning rate is substantial across tasks (~36% to ~57%), consistent with Optuna pruning behavior.

## Reproduction Notes

- Recompute analysis from `mlflow.db` and compare to archived CSV files.
- Validate environment against `reports/results/2026-03-14_sweep-pack-v1_env.txt` and `uv.lock` hash in metadata JSON.
