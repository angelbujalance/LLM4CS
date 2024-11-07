#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=REW_CoT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=output/success/out-REW_prompting_CoT_%x.%A.out
#SBATCH --error=output/error/out-REW_prompting_CoT_%x.%A.err

python chat_prompt_cot_rewrite.py \
--open_ai_key_id=0 \
--test_file_path="./datasets/cast19_test.json" \
--demo_file_path="./demonstrations.json" \
--qrel_file_path="./datasets/cast19_qrel.tsv" \
--work_dir="./results/new/cast19/COT_REW" \
--n_generation=5 \