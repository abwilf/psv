# Propose, Solve, Verify: Self-Play Through Formal Verification

PSV is a self-improving code generation system for verified code generation in Verus.

## Installation

```bash
conda create -n psv python=3.11 -y && conda activate psv
pip install -r requirements.txt
wandb login
```

## Running Experiments

All experiments are defined in `sweep.yml` and run using W&B sweeps:

### Step 1: Set up evaluation server
See `eval_server/SETUP.md` for instructions. You'll need:
- Verus installation (version 0.2025.06.05.d617bea recommended)
- Singularity container (build with `build_container.sh`)

### Step 2: Run experiments with W&B
```bash
# Create a sweep from sweep.yml
wandb sweep sweep.yml

# Edit train.sbatch with your cluster paths and wandb agent command from above, then submit it
sbatch train.sbatch
```

## Code Structure

- `train.py` - Main training loop: inference -> evaluation -> SFT -> proposal -> repeat
- `inference.py` - SGLang-based inference with LoRA support
- `sft.py` - TRL-based supervised fine-tuning (LoRA, completion-only loss)
- `src/core/` - Core infrastructure (server management, evaluation, caching)
- `src/proposal/` - Problem proposal strategies
  - Note: The algorithm is called "AlphaVerus-Zero" or "AV0" in the code (early name)
- `eval_server/` - Verus verification server
- `data/` - Dataset files (dafny2verus, humaneval, mbpp)

## Pre-trained Models

We provide trained LoRA weights from our main experiments in `model_runs/`:
- `AV0-seed0` through `AV0-seed4` - Five seeds of our main PSV training runs

Each contains `models/iter5/` with the final LoRA adapter (compatible with Qwen/Qwen2.5-Coder-7B-Instruct).

## Reproducing Paper Results

If you uncomment everything in `sweep.yml` and run it with the commands above, you will reproduce all the results for the paper.

Analysis scripts in `analysis/` reproduce paper figures:
- `ttt/` - Test-time training results
- `scaling_max_n_qs/` - Scaling experiments
- `io/` - Input/output strategy analysis
- `budget/` - Budget analysis
- `verification/` - Verification ablations
- `ablations/` - General ablations

Run each with `python main.py` from the respective directory.
