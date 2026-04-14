#!/bin/bash
#SBATCH --partition=gl40s_short,gl40s_long,gpu8_short,gpu8_medium,gpu8_long,gpu4_short,gpu4_medium,gpu4_long,a100_short,a100_long
#SBATCH --job-name=1_train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G

# activate conda env
source ~/.bashrc
conda activate nlp-hw4-2

# exit immediately if a command exits with a non-zero status, if an undefined variable is used, or if any command in a pipeline fails
set -euo pipefail

# move to part-2 folder
cd /gpfs/scratch/yb2612/classes/nlp/hw4/nlp-hw4/part-2

# train configuration
EXPERIMENT_NAME="beam_4_prefix"  # name for logging and checkpointing
MODEL_TYPE="pretrained"  # choices: pretrained, scratch

# run training + built-in dev/test evaluation
python3 train_t5.py \
    --finetune \
    --model_type "${MODEL_TYPE}" \
    --num_beams 4 \
    --optimizer_type AdamW \
    --learning_rate 1e-4 \
    --weight_decay 0.001 \
    --scheduler_type cosine \
    --num_warmup_epochs 3 \
    --max_n_epochs 30 \
    --patience_epochs 5 \
    --batch_size 16 \
    --test_batch_size 16 \
    --use_wandb \
    --experiment_name "${EXPERIMENT_NAME}" \
    --calc_dataset_stats \
    --max_source_length 96 \
    --max_target_length 384 \
    --max_new_tokens 384 \
    --add_task_prefix