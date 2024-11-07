#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=InstallEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=output/success/out-RAR_prompting_CoT_%x.%A.out
#SBATCH --error=output/error/out-RAR_prompting_CoT_%x.%A.err

python chat_prompt_cot_rewrite_and_response.py \
--open_ai_key_id=0 \
--qrel_file_path="./datasets/cast19_qrel.tsv" \
--test_file_path="./datasets/cast19_test.json" \
--demo_file_path="./demonstrations.json" \
--work_dir="./results/new/cast19/COT_RAR" \
--n_generation=5 \