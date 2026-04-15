#!/bin/bash
#SBATCH --partition=gl40s_short,gl40s_long,gpu8_short,gpu8_medium,gpu8_long,gpu4_short,gpu4_medium,gpu4_long,a100_short,a100_long
#SBATCH --job-name=length_512_prefix_2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G

# activate conda env
source ~/.bashrc
conda activate nlp-hw4-2

# exit immediately if a command exits with a non-zero status, if an undefined variable is used, or if any command in a pipeline fails
set -euo pipefail

# move to part-2 folder
cd /gpfs/scratch/yb2612/classes/nlp/hw4/nlp-hw4/part-2

# train configuration
EXPERIMENT_NAME="length_512_prefix_2"  # name for logging and checkpointing
MODEL_TYPE="pretrained"  # choices: pretrained, scratch

# run training + built-in dev/test evaluation
python3 train_t5.py \
    --finetune \
    --model_type "${MODEL_TYPE}" \
    --num_beams 1 \
    --optimizer_type AdamW \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --scheduler_type cosine \
    --num_warmup_epochs 0 \
    --max_n_epochs 80 \
    --patience_epochs 15 \
    --batch_size 16 \
    --test_batch_size 16 \
    --use_wandb \
    --experiment_name "${EXPERIMENT_NAME}" \
    --calc_dataset_stats \
    --max_source_length 96 \
    --max_target_length 512 \
    --max_new_tokens 512 \
    --finetune_scope full \
    --normalize_whitespace \
    --add_task_prefix