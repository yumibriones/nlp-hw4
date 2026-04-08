#!/bin/bash
#SBATCH --job-name=4_eval_original
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# Move to part-1 regardless of where sbatch is launched from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

mkdir -p logs

# Activate your conda env.
source ~/.bashrc
conda activate nlp_hw4_clean

MODEL_DIR="${MODEL_DIR:-./out_augmented}"

python3 main.py --eval --model_dir "${MODEL_DIR}"
