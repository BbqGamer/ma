# CIFAR-100 easy-vs-hard Runpod plan

## Recommended pod

- Preferred: **RTX 4090 24GB**
- Good fallback: **A5000 24GB** or **A10 24GB**

Why:
- enough VRAM for mixed-precision ResNet18 runs
- strong perf / dollar
- suitable for repeated 10-trial search jobs

## Experiment defaults used here

Search defaults are tuned for fair and cheap comparison:
- full CIFAR-100
- ResNet18-style model
- mixed precision
- **fixed 30 epochs**
- **early stopping disabled by default** for search
- one seed during search unless you override `--eval-seeds`

This is intentional: fixed-length runs make the schedule comparison cleaner and easier for the LLM to learn from.

## One-time setup on Runpod

```bash
cd /workspace

git clone <YOUR_REPO_URL> ma-code
cd ma-code

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install lightning torchvision
```

Optional sanity check:

```bash
python - <<'PY'
import torch, torchvision, lightning
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
PY
```

## Environment variables

Only needed for the LLM-guided search:

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
```

## Run 10-trial Optuna search

```bash
source .venv/bin/activate

python scripts/run_cifar100_optuna_search.py \
  --study-name cifar100_optuna_v1 \
  --n-trials 10 \
  --eval-seeds 42 \
  --batch-size 256 \
  --num-workers 8 \
  --max-epochs 30 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --history-window 5 \
  --ema-alpha 0.3 \
  --accelerator gpu \
  --devices 1 \
  --precision 16-mixed
```

## Run 10-candidate LLM-guided search

```bash
source .venv/bin/activate

python scripts/run_cifar100_llm_search.py \
  --study-name cifar100_llm_v1 \
  --iterations 10 \
  --eval-seeds 42 \
  --batch-size 256 \
  --num-workers 8 \
  --max-epochs 30 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --history-window 5 \
  --ema-alpha 0.3 \
  --accelerator gpu \
  --devices 1 \
  --precision 16-mixed
```

## Optional: stronger final validation after search

After selecting the best LLM schedule module and the best Optuna parameter set, evaluate them with more seeds and a longer budget.

For an LLM winner:

```bash
python scripts/run_cifar100_policy_benchmark.py \
  --benchmark-id cifar100_llm_best_eval_v1 \
  --schedule-module llm_schedules.generated.cifar100_llm_v1.candidate_0007 \
  --eval-seeds 42,43,44,45,46 \
  --batch-size 256 \
  --num-workers 8 \
  --max-epochs 100 \
  --accelerator gpu \
  --devices 1 \
  --precision 16-mixed
```

If you want fixed-budget final validation too, leave early stopping off.

## Where results are stored

Main outputs:
- `reports/cifar100_easy_hard/optuna/<study_name>/...`
- `reports/cifar100_easy_hard/llm_search/<study_name>/...`
- `reports/cifar100_easy_hard/policy/<benchmark_id>/...`
- `llm_schedules/generated/<llm_study_name>/...`

Useful files inside runs:
- `aggregate.json`
- `per_seed_results.jsonl`
- `trajectory.csv`
- `weights_and_hard_val.png`
- generated LLM policy `.py` files

## Bundle everything for download

```bash
source .venv/bin/activate

python scripts/bundle_cifar100_results.py \
  --bundle-name cifar100_v1_bundle \
  --llm-study-name cifar100_llm_v1 \
  --optuna-study-name cifar100_optuna_v1
```

This creates:
- `reports/artifacts/cifar100_v1_bundle.tar.gz`

## Download from Runpod

If you have SSH access to the pod:

```bash
scp root@<RUNPOD_HOST>:/workspace/ma-code/reports/artifacts/cifar100_v1_bundle.tar.gz .
```

Or download it directly from the Runpod file browser.

## Suggested workflow

1. Run Optuna search with 10 trials.
2. Run LLM search with 10 candidates.
3. Bundle and download results.
4. Inspect:
   - best hard validation loss
   - final hard validation accuracy
   - weight trajectories
   - generated LLM policy code
5. Optionally run a stronger 5-seed / 100-epoch final validation for the winners.
