# ------------------------------------------------------------------------------
# query_classifier/hybrid_extractor.py - Extracting "Hybrid"-labeled Queries
# ------------------------------------------------------------------------------
"""
Loads `large_labeled_sampled_queries.jsonl` and extracts the queries
labeled "hybrid" into a seperate file, For future assessment and classification.
"""

import json

def extract_hybrid(input_path: str, output_path: str, existing_path: str):
    existing_ids = set()

    with open (existing_path, 'r', encoding = 'utf-8') as existing_file:
        for line in existing_file:
            entry = json.loads(line)
            existing_ids.add(entry["id"])

    new_hybrids = []

    with open(input_path, 'r', encoding = 'utf-8') as infile:
        for line in infile:
            entry = json.loads(line)
            if entry.get("label") == "hybrid" and entry["id"] not in existing_ids:
                new_hybrids.append({"id": entry["id"], "query": entry["query"]})
                existing_ids.add(entry["id"])

    if new_hybrids:
        with open(output_path, 'a', encoding = 'utf-8') as outfile:
            for item in new_hybrids:
                outfile.write(json.dumps(item, ensure_ascii = False) + "\n")
        
    print(f"✅ Appended {len(new_hybrids)} new hybrid queries to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description = "Extract Hybrid queries from the large labeled sampled queries.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input sampled queries JSONL file - knowledgebase/additional_labeled_queries.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Path to the output labeled sampled hybrid queries JSONL file - knowledgebase/hybrid_queries.jsonl")
    parser.add_argument("--existing", type=str, required=True, help="Path to the existing hybrid queries JSONL file - knowledgebase/hybrid_queries.jsonl")
    args = parser.parse_args()

    extract_hybrid(args.input, args.output, args.existing)