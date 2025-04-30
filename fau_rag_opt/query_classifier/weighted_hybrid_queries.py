# ------------------------------------------------------------------------------
# query_classifier/weighted_hybrid_queries.py - Weighting Hybrid queries: Dense Vs. Sparse
# ------------------------------------------------------------------------------
"""
Smart Dynamic Hybrid Retrieval - Loads `hybrid_queries.jsonl` and estimate how much a query should rely on
dense vs sparse and returns the weights for each. End Goal - Dynamically adapt the Alpha
"""

from ..query_classifier.labeler_config import (LabelerConfig,
                                               labeler_config)

from ..constants.my_constants import SAVE_INTERVAL

from ..helpers.labelling import LabelSetup

import aiohttp
import time
import json

class HybridWeighting:
    def __init__(self, config: LabelerConfig):
        self.config = config
        self.base_url = self.config.base_url
        self.api_key = self.config.api_key
        self.model = self.config.model

    async def llm_hybrid_weighting(self, prompt: str) -> str:
        start_time = time.time()
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }

        print(f"\n📨 Sending prompt to LLM...")

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.base_url, headers=headers, json=data) as response:
                    print(f"🔄 Received response status: {response.status}")
                    if response.status != 200:
                        error_message = await response.text()
                        print(f"❗ Error message: {error_message}")
                        raise ValueError(f"❗ LLM endpoint returned {response.status}")

                    response_data = await response.json()

                    if 'choices' not in response_data or len(response_data['choices']) == 0:
                        print(f"❗ No 'choices' found in the LLM response!")
                        raise ValueError("❗ LLM response missing 'choices'")

                    answer = response_data['choices'][0]['message']['content']
                    elapsed_time = time.time() - start_time
                    print(f"✅ LLM answered in {elapsed_time:.2f} seconds. Label: {answer.strip()}")
                    return answer.strip()
            except Exception as e:
                print(f"❗ Exception during LLM call: {str(e)}")
                raise

    async def weight_queries(self, input_path: str, output_path: str):
        try:
            Lsetup = LabelSetup()

            samples = Lsetup.load_samples(input_path)
            existing_labels = Lsetup.load_existing_samples(output_path)

            labels = existing_labels.copy()
            unlabeled_samples = [s for s in samples if s["id"] not in labels]

            total_samples = len(samples)
            labeled_samples = len(labels)

            print(f"Loaded {total_samples} hybrid samples, {labeled_samples} already weighted. {len(unlabeled_samples)} left to weight.\n")

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

                response = await self.llm_hybrid_weighting(prompt)

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
    asyncio.run(hybrid_weighting.weight_queries(args.input, args.output))