#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=REW
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=output/success/out-REW_prompting.out
#SBATCH --error=output/error/out-REW_prompting.err


python chat_prompt_rewrite.py \
--open_ai_key_id=0 \
--test_file_path="./datasets/cast19_test.json" \
--demo_file_path="./demonstrations.json" \
--qrel_file_path="./datasets/cast19_qrel.tsv" \
--work_dir="./results/new/cast19/REW" \
--n_generation=5 \