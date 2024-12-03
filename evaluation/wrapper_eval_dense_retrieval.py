import sys
import numpy as np

# Map numpy._core.multiarray to numpy.core.multiarray !!!
sys.modules["numpy._core"] = sys.modules["numpy.core"]
sys.modules["numpy._core.multiarray"] = np.core.multiarray

# Print debug info
# print("Adjusted sys.modules keys:", sys.modules.keys())

# Import the main evaluation script as a module
import eval_dense_retrieval

if __name__ == "__main__":
    import argparse

    # Parse arguments and forward them to the main script
    parser = argparse.ArgumentParser(description="Wrapper for evaluation script with adjusted sys.modules.")
    parser.add_argument("--eval_file_path", required=True)
    parser.add_argument("--eval_field_name", required=True)
    parser.add_argument("--qrel_file_path", required=True)
    parser.add_argument("--index_path", required=True)
    # parser.add_argument("--processed_data_dir", required=True)
    parser.add_argument("--retriever_path", required=True)
    # parser.add_argument("--use_gpu_in_faiss", action="store_true")
    # parser.add_argument("--n_gpu_for_faiss", type=int, default=1)
    parser.add_argument("--top_n", type=int, default=1000)
    parser.add_argument("--rel_threshold", type=int, default=1)
    parser.add_argument("--retrieval_output_path", required=True)
    parser.add_argument("--include_query", action="store_true")
    parser.add_argument("--aggregation_method", default="sc")

    args = parser.parse_args()

    # Execute the eval_dense_retrieval script
    script_path = "eval_dense_retrieval.py"
    # script_path = "eval_dense_retrieval_v2.py"
    with open(script_path, "r") as script_file:
        code = script_file.read()
    exec(code, globals())
