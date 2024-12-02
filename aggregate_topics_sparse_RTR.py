import json
import argparse

def aggregate_rtr_topics(input_file, output_file, aggregation_method, qrel_file=None):
    """Aggregate topics for RTR input."""
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
            rewrite = record["predicted_rewrite"]  # Single rewrite in RTR
            responses = record["predicted_response"]

            if has_qrel_label_sample_ids and sample_id not in has_qrel_label_sample_ids:
                continue

            # Apply aggregation based on the method
            if aggregation_method == "maxprob":
                # Sanitize the rewrite to remove \n and replace with space
                sanitized_rewrite = rewrite[0].replace("\n", " ")
                # Sanitize the first response as well
                sanitized_response = responses[0].replace("\n", " ")
                # Concatenate the sanitized rewrite with the sanitized response
                aggregated_text = f"{sanitized_rewrite} {sanitized_response}"

            elif aggregation_method == "mean":
                # Concatenate rewrite with all responses
                aggregated_text = " ".join([r.replace("\n", " ") for r in rewrite]) + " " + " ".join([response.replace("\n", " ") for response in responses])
            else:
                raise NotImplementedError(f"Aggregation method {aggregation_method} not supported.")

            # Write a single aggregated entry per topic
            out_f.write(f"{sample_id}\t{aggregated_text}\n")

    print(f"Aggregated RTR topics saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate topics for RTR input.")
    parser.add_argument("--input", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output", required=True, help="Path to the output topics file.")
    parser.add_argument("--aggregation-method", choices=["maxprob", "mean"], default="maxprob",
                        help="Aggregation method to use (default: maxprob).")
    parser.add_argument("--qrel", help="Path to QRel file (optional).")
    args = parser.parse_args()

    aggregate_rtr_topics(args.input, args.output, args.aggregation_method, args.qrel)
