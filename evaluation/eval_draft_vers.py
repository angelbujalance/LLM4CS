def parse_qrel(file_path, threshold=2):
    qrel = {}
    with open(file_path, 'r') as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split()
            if int(relevance) >= threshold:
                if query_id not in qrel:
                    qrel[query_id] = set()
                qrel[query_id].add(doc_id)
    return qrel

def parse_run(file_path):
    run = {}
    with open(file_path, 'r') as f:
        for line in f:
            query_id, _, doc_id, _, _, _,_ = line.strip().split()
            if query_id not in run:
                run[query_id] = []
            run[query_id].append(doc_id)
    return run

def compute_recall(qrel, run):
    recall_scores = {}
    for query_id in qrel:
        relevant_docs = qrel[query_id]
        retrieved_docs = run[query_id] if query_id in run else []
        retrieved_relevant = relevant_docs & set(retrieved_docs)
        recall = len(retrieved_relevant) / len(relevant_docs) if relevant_docs else 0
        recall_scores[query_id] = recall
    return recall_scores

# File paths
qrel_path = "/scratch-shared/dense_llm4cs/cast19_qrel.tsv"
run_path = "/home/scur2853/LLM4CS/results/new/cast19/REW/ance/+q+r+sc/res.trec"

# Process files
qrel = parse_qrel(qrel_path, threshold=2)
run = parse_run(run_path)

# Compute recall
recall_scores = compute_recall(qrel, run)

# Output recall
for query_id, recall in recall_scores.items():
    print(f"Query: {query_id}, Recall: {recall:.4f}")