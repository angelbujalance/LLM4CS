#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=Evaluation_llmcs2_final
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=output/success/out-eval_dense_%x.%A.out
#SBATCH --error=output/error/out-eval_dense_%x.%A.err

module purge
module load 2022
module load Anaconda3/2022.05

# Initialize Conda
source /sw/arch/RHEL8/EB_production/2022/software/Anaconda3/2022.05/etc/profile.d/conda.sh

# Activate the correct Conda environment
conda activate llmcs2


# Change to the evaluation directory
cd $HOME/LLM4CS/evaluation


eval_field_name="predicted_rewrite"
work_dir="../results/new/cast19/REW"    # set your the folder containing your `rewrites.jsonl`file

eval_file_path="$work_dir/rewrites.jsonl" 
# index_path="/scratch-shared/llm4cs/new_embeddings/" # set the pre-built index which contains all candidate passage emebddings from ConvDR. 
index_path="/scratch-shared/dense_llm4cs/embeds"                                               
# qrel_file_path="../datasets/cast19_qrel.tsv" # set the qrel file path
qrel_file_path="/scratch-shared/dense_llm4cs/cast19_qrel.tsv"
retrieval_output_path="$work_dir/ance/+q+r+sc" # set your expected output folder path

#----------(disabled)-----------------------------------
# --include_response \ # enable `include_response` if you test RTR or RAR prompting.
# --use_gpu_in_faiss \
# --n_gpu_for_faiss=1 \
#---------------------------------------------                                                                                   
export CUDA_VISIBLE_DEVICES=0
# python eval_dense_retrieval.py \
python wrapper_eval_dense_retrieval.py \
--eval_file_path=$eval_file_path \
--eval_field_name=$eval_field_name \
--qrel_file_path=$qrel_file_path \
--index_path=$index_path \
--retriever_path="/scratch-shared/ad-hoc-ance-msmarco/" \
--top_n=1000 \
--rel_threshold=1 \
--retrieval_output_path=$retrieval_output_path \
--include_query \
--aggregation_method="sc"