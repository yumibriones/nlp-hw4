#!/bin/bash
#SBATCH --partition=gl40s_short,gl40s_long,gpu8_short,gpu8_medium,gpu8_long,gpu4_short,gpu4_medium,gpu4_long,a100_short,a100_long
#SBATCH --job-name=eval_defaults
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G

# # activate conda env
# source ~/.bashrc
# conda activate nlp-hw4-2

# # exit immediately if a command exits with a non-zero status, if an undefined variable is used, or if any command in a pipeline fails
# set -euo pipefail

# # move to part-2 folder
# cd /gpfs/scratch/yb2612/classes/nlp/hw4/nlp-hw4/part-2

# run script
EXPERIMENT_NAME="length_512_prefix_lr"
ERROR_TABLE_CSV="results/${EXPERIMENT_NAME}_error_table.csv"

python3 evaluate.py \
    --predicted_sql results/t5_ft_${EXPERIMENT_NAME}_dev.sql \
    --predicted_records records/t5_ft_${EXPERIMENT_NAME}_dev.pkl \
    --development_sql data/dev.sql \
    --development_records records/ground_truth_dev.pkl \
    --error_table_csv ${ERROR_TABLE_CSV}
