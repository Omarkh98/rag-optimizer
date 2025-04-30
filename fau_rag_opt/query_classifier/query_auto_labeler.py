# ------------------------------------------------------------------------------
# query_classifier/query_auto_labeler.py - 
# ------------------------------------------------------------------------------
"""

"""

from ..query_classifier.labeler_config import (LabelerConfig,
                                               labeler_config)

from ..constants.my_constants import SAVE_INTERVAL

from ..helpers.labelling import LabelSetup

import aiohttp
import time

class AutoLabeler:
    def __init__(self, config: LabelerConfig):
        self.config = config
        self.base_url = self.config.base_url
        self.api_key = self.config.api_key
        self.model = self.config.model

    async def fau_llm_labeler(self, prompt: str) -> str:
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
            
    async def label_queries(self, input_path: str, output_path: str):
        try:
            Lsetup = LabelSetup()

            samples = Lsetup.load_samples(input_path)

            existing_labels = Lsetup.load_existing_samples(output_path)

            labels = existing_labels.copy()

            unlabeled_samples = [s for s in samples if s["id"] not in labels]

            total_samples = len(samples)
            labeled_samples = len(labels)

            print(f"Loaded {total_samples} samples, {labeled_samples} already labeled. {len(unlabeled_samples)} left to auto-label.\n")

            count_since_last_save = 0

            for i, sample in enumerate(unlabeled_samples, start=labeled_samples + 1):
                print(f"\n[{i}/{total_samples}] Query:")
                print(sample['query'])

                prompt = (
                    "You are a retrieval expert classifying queries based on the type of information retrieval they require.\n"
                    "The query may be in English or German. Please analyze the query in its own language and classify it strictly as:\n\n"
                    "- **Dense**: For conceptual queries needing semantic understanding.\n"
                    "  Example EN: 'Explain how renewable energy policies impact economies.'\n"
                    "  Example DE: 'Erkläre, wie erneuerbare Energiepolitik die Wirtschaft beeinflusst.'\n\n"
                    "- **Sparse**: For exact-term or keyword-based queries.\n"
                    "  Example EN: '2024 FAU Physics PhD requirements.'\n"
                    "  Example DE: 'Zulassungsvoraussetzungen FAU Physik Promotion 2024.'\n\n"
                    "- **Hybrid**: When both exact terms and semantic understanding are needed.\n"
                    "  Example EN: 'How does FAU support AI startups in medical imaging?'\n"
                    "  Example DE: 'Wie unterstützt die FAU KI-Startups im Bereich medizinischer Bildgebung?'\n\n"
                    "Return ONLY the most relevant category as a single lowercase word: dense, sparse, or hybrid.\n\n"
                    f'Query: "{sample["query"]}"'
                )

                label = (await self.fau_llm_labeler(prompt)).strip().lower()
                
                labels[sample["id"]] = {
                    "id": sample["id"],
                    "query": sample["query"],
                    "label": label
                }
            
                count_since_last_save += 1

                if count_since_last_save >= SAVE_INTERVAL:
                    Lsetup.save_labels(labels, output_path)
                    print(f"💾 Progress saved! ({len(labels)}/{total_samples} labeled)")
                    count_since_last_save = 0
                    
            if count_since_last_save > 0:
                Lsetup.save_labels(labels, output_path)
                print(f"\n✅ Final progress saved! ({len(labels)}/{total_samples} labeled)")

        except Exception as e:
            print(f"An unexpected error occurred while reading '{input_path}': {e}")
            return
        
if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Automatically label FAU sample queries using LLM.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input sampled queries JSONL file - knowledgebase/large_sampled_queries.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Path to the output labeled sampled queries JSONL file - knowledgebase/large_labeled_sampled_queries.jsonl")
    args = parser.parse_args()


    labeler = AutoLabeler(labeler_config)
    asyncio.run(labeler.label_queries(args.input, args.output))