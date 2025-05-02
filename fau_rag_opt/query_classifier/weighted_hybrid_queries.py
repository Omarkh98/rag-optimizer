# ------------------------------------------------------------------------------
# query_classifier/weighted_hybrid_queries.py - Weighting Hybrid queries: Dense Vs. Sparse
# ------------------------------------------------------------------------------
"""
Smart Dynamic Hybrid Retrieval - Loads `hybrid_queries.jsonl` and estimate how much a query should rely on
dense vs sparse and returns the weights for each. End Goal - Dynamically adapt the Alpha
"""

from ..query_classifier.labeler_config import (LabelerConfig,
                                               labeler_config)

from ..query_classifier.llm_labeler import llm_labeler

from ..constants.my_constants import SAVE_INTERVAL

from ..helpers.labelling import LabelSetup

import json

class HybridWeighting:
    def __init__(self, config: LabelerConfig):
        self.config = config
        self.base_url = self.config.base_url
        self.api_key = self.config.api_key
        self.model = self.config.model

    async def weight_hybrid_queries(self, input_path: str, output_path: str):
        try:
            Lsetup = LabelSetup()

            samples = Lsetup.load_samples(input_path)

            existing_labels = Lsetup.load_existing_samples(output_path)
            existing_ids = set(existing_labels.keys())

            unlabeled_samples = [s for s in samples if s["id"] not in existing_ids]

            total_samples = len(samples)
            labeled_samples = len(existing_labels)

            print(f"Loaded {total_samples} hybrid samples, {labeled_samples} already weighted. {len(unlabeled_samples)} left to weight.\n")
            
            labels = existing_labels.copy()
            count_since_last_save = 0

            for i, sample in enumerate(unlabeled_samples, start=labeled_samples + 1):
                print(f"\n[{i}/{total_samples}] Query:")
                print(sample['query'])

                prompt = (
                    "You are a retrieval optimization expert. Given the following user query (in English or German), "
                    "assign weights to two retrieval methods — dense and sparse — such that:\n"
                    "- 'dense' is for conceptual or semantic retrieval\n"
                    "- 'sparse' is for keyword/exact match retrieval\n\n"
                    "Your task is to estimate how much each method should contribute to answering the query. "
                    "Return ONLY two float values: 'dense_weight' and 'sparse_weight'. "
                    "Make sure they sum to exactly 1.0.\n\n"
                    f'Query: "{sample["query"]}"\n\n'
                    'Example output: {"dense_weight": 0.7, "sparse_weight": 0.3}'
                )

                response = await llm_labeler(self.api_key, self.model, self.base_url, prompt)
                raw_response = response.strip()

                if raw_response.startswith("```json"):
                    raw_response = raw_response.replace("```json", "").strip()
                if raw_response.endswith("```"):
                    raw_response = raw_response[:-3].strip()

                weight_data = json.loads(raw_response)

                labels[sample["id"]] = {
                    "id": sample["id"],
                    "query": sample["query"],
                    "dense_weight": weight_data["dense_weight"],
                    "sparse_weight": weight_data["sparse_weight"]
                }

                count_since_last_save += 1

                if count_since_last_save >= SAVE_INTERVAL:
                    Lsetup.save_labels(labels, output_path)
                    print(f"💾 Progress saved! ({len(labels)}/{total_samples} labeled)")
                    count_since_last_save = 0

            # Final save after loop
            if count_since_last_save > 0:
                Lsetup.save_labels(labels, output_path)
                print(f"\n✅ Final progress saved! ({len(labels)}/{total_samples} labeled)")

        except Exception as e:
            print(f"An unexpected error occurred while reading '{input_path}': {e}")
            return
        
if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Weight hybrid queries with their respective dense/sparse weights.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input sampled queries JSONL file - knowledgebase/hybrid_queries.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Path to the output labeled sampled queries JSONL file - knowledgebase/weighted_hybrid_queries.jsonl")
    args = parser.parse_args()


    hybrid_weighting = HybridWeighting(labeler_config)
    asyncio.run(hybrid_weighting.weight_hybrid_queries(args.input, args.output))