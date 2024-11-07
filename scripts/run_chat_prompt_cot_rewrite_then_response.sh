#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=RTR_CoT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=output/success/out-RTR_cot_prompting.out
#SBATCH --error=output/error/out-RTR_cot_prompting.err

python chat_prompt_rewrite_then_response.py \
--open_ai_key_id=0 \
--test_file_path="./datasets/cast19_test.json" \
--qrel_file_path="./datasets/cast19_qrel.tsv" \
--work_dir="./results/new/cast19/RTR" \
--demo_file_path="./demonstrations.json" \
--rewrite_file_path="./results/new/cast19/COT_RTR/rewrites.jsonl" \
--n_generation=5 \