# ------------------------------------------------------------------------------
# query_classifier/hybrid_extractor.py - Extracting "Hybrid"-labeled Queries
# ------------------------------------------------------------------------------
"""
Loads `large_labeled_sampled_queries.jsonl` and extracts the queries
labeled "hybrid" into a seperate file, For future assessment and classification.
"""

import json

def extract_hybrid(input_path: str, output_path: str):
    hybrid_queries = []

    with open(input_path, 'r', encoding = 'utf-8') as infile:
        for line in infile:
            entry = json.loads(line)
            if entry.get("label") == "hybrid":
                hybrid_queries.append({"id": entry["id"], "query": entry["query"]})

    with open(output_path, 'w', encoding = 'utf-8') as outfile:
        for item in hybrid_queries:
            outfile.write(json.dumps(item, ensure_ascii = False) + "\n")
        
    print(f"✅ Extracted {len(hybrid_queries)} hybrid queries to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description = "Extract Hybrid queries from the large labeled sampled queries.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input sampled queries JSONL file - knowledgebase/large_labeled_sampled_queries.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Path to the output labeled sampled queries JSONL file - knowledgebase/hybrid_queries.jsonl")
    args = parser.parse_args()

    extract_hybrid(args.input, args.output)