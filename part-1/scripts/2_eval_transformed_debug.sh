#!/bin/bash
#SBATCH --partition=gl40s_dev,gl40s_short,gl40s_long,gpu8_short,gpu8_medium,gpu8_long,gpu4_dev,gpu4_short,gpu4_medium,gpu4_long,a100_dev,a100_short,a100_long
#SBATCH --job-name=2_eval_transformed
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G

# activate conda env
source ~/.bashrc
conda activate nlp_hw4_clean

# exit immediately if a command exits with a non-zero status, if an undefined variable is used, or if any command in a pipeline fails
set -euo pipefail

# move to part-1 folder
cd /gpfs/scratch/yb2612/classes/nlp/hw4/nlp-hw4/part-1

# run script
python3 main.py --eval_transformed --debug_transformation
