#!/bin/bash
#SBATCH -p general
#SBATCH --constrain="A6000|L40|L40S"
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300GB
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# Environment setup - adjust these paths for your system
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export TORCH_COMPILE_CACHE_DIR=${TORCH_COMPILE_CACHE_DIR:-~/.cache/torch}
export NCCL_P2P_DISABLE=1

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate psv

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Auto-detect physical GPU index to prevent port conflicts and set ports based on it
export PHYSICAL_GPU=$(python3 -c "from src.utils.gpu_utils import get_physical_gpu_index; print(get_physical_gpu_index(0))")
echo "Detected Physical GPU: ${PHYSICAL_GPU}"
export EVAL_SERVER_BASE_PORT=5000
export EVAL_SERVER_PORT=$((EVAL_SERVER_BASE_PORT + PHYSICAL_GPU))
export SGLANG_BASE_PORT=3002
export SGLANG_PORT=$((SGLANG_BASE_PORT + PHYSICAL_GPU))
echo "Eval Server Port: ${EVAL_SERVER_BASE_PORT} + ${PHYSICAL_GPU} = ${EVAL_SERVER_PORT}"
echo "SGLang Port: ${SGLANG_BASE_PORT} + ${PHYSICAL_GPU} = ${SGLANG_PORT}"

# Start eval server in background
cd eval_server
./run_server.sh &
cd ..

# Run training (uncomment and configure wandb sweep)
# wandb agent YOUR_ENTITY/YOUR_PROJECT/SWEEP_ID
