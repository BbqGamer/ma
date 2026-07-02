# Experiment Configs

These YAML configs are for the `ma_thesis/` Gaussian-continuation regression
track. The second-stage image-classification experiments live in
`coarse-to-fine-curriculum/` and are configured mainly through CLI flags and
Runpod/Docker environment variables documented in that directory's README.

Run a config with:

```bash
python -m ma_thesis.experiment run-config <path-to-yaml>
```

Current sweep configs:
- `configs/sweeps/ackley_sweep.yaml`
- `configs/sweeps/levy_sweep.yaml`
- `configs/sweeps/eggholder_sweep.yaml`
- `configs/sweeps/bukin_sweep.yaml`
- `configs/sweeps/franke_sweep.yaml`
- `configs/sweeps/peaks_sweep.yaml`
- `configs/sweeps/friedman1_2d_sweep.yaml`
- `configs/sweeps/friedman2_2d_sweep.yaml`
- `configs/sweeps/schedule_sweep_eggholder.yaml` (schedule-only curriculum study)

Recommendation for schedule-only studies:
- Keep `schedule_num_losses` fixed across functions (e.g. always `4`) for fair comparisons.

Templates for non-sweep runs:
- `configs/experiments/curriculum_from_sweep_template.yaml`
- `configs/experiments/meta_manual_template.yaml`

To run the full sweep pack overnight:

```bash
./scripts/run_sweep_pack.sh
```

Minimal schema:

```yaml
run:
  method: sweep            # single | curriculum | meta | sweep
  function: ackley         # ackley | levy | eggholder | bukin
  regenerate_data: true
  num_samples: 8000
  num_sigmas: 5
  sigma_scale: 5.0
  seed: 42
  n_trials: 120
  epochs: 800
```
