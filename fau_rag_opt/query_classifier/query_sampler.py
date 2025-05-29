# ------------------------------------------------------------------------------
# query_classifier/query_sampler.py - Sampling random queries for labelling.
# ------------------------------------------------------------------------------
"""
Loads `extracted_queries.jsonl` and samples Random queries from
the extracted FAU user queries.
"""

import json
import random

def sample_queries(input_path: str, output_path: str, num_samples: int):
    try:
        queries = []

        with open(input_path, 'r', encoding = 'utf-8') as infile:
            for line in infile:
                queries.append(json.loads(line))

        sampled = random.sample(queries, min(num_samples, len(queries)))

        with open(output_path, 'w', encoding = 'utf-8') as outfile:
            for item in sampled:
                outfile.write(json.dumps(item, ensure_ascii = False) + "\n")

        print(f"✅ Sampled {len(sampled)} queries to {output_path}")

    except Exception as e:
        print(f"An unexpected error occurred while reading '{input_path}': {e}")
        return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description = "Sample random queries from extraced FAU user queries.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input extracted queries JSONL file - knowledgebase/extracted_queries.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Path to the output sampled queries JSONL file - knowledgebase/sampled_queries.jsonl")
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of queries to sample")
    args = parser.parse_args()

    sample_queries(args.input, args.output, args.num_samples)