import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



def aggregate_topics(input_file, output_file, aggregation_method, qrel_file=None):
    """Aggregate topics for sparse retrieval."""
    has_qrel_label_sample_ids = set()
    if qrel_file:
        with open(qrel_file, "r") as f:
            for line in f:
                sample_id = line.strip().split()[0]
                has_qrel_label_sample_ids.add(sample_id)

    with open(input_file, "r") as f, open(output_file, "w") as out_f:
        for line in f:
            record = json.loads(line)
            sample_id = record["sample_id"]
            rewrites = record["predicted_rewrite"]

            if has_qrel_label_sample_ids and sample_id not in has_qrel_label_sample_ids:
                continue

            # Apply aggregation
            if aggregation_method == "maxprob":
                aggregated_query = rewrites[0]  # Use the first rewrite
            elif aggregation_method == "mean":
                aggregated_query = " ".join(rewrites)  # Combine terms from all rewrites
            else:
                raise NotImplementedError(f"Aggregation method {aggregation_method} not supported.")

            # Write a single query per topic
            out_f.write(f"{sample_id}\t{aggregated_query}\n")

    print(f"Aggregated topics saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate topics for sparse retrieval.")
    parser.add_argument("--input", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output", required=True, help="Path to the output topics file.")
    parser.add_argument("--aggregation-method", choices=["maxprob", "mean"], default="maxprob",
                        help="Aggregation method to use (default: maxprob).")
    parser.add_argument("--qrel", help="Path to QRel file (optional).")
    args = parser.parse_args()

    aggregate_topics(args.input, args.output, args.aggregation_method, args.qrel)
