# ------------------------------------------------------------------------------
# query_classifier/query_auto_labeler.py - Automatic LLM-based labeling.
# ------------------------------------------------------------------------------
"""
Loads `extracted_queries.jsonl`, parses through them and labels them as:
dense, sparse, or hybrid.
"""

from ..query_classifier.labeler_config import (LabelerConfig,
                                               labeler_config)

from ..query_classifier.llm_labeler import llm_labeler

from ..constants.my_constants import SAVE_INTERVAL

from ..helpers.labelling import LabelSetup

class AutoLabeler:
    def __init__(self, config: LabelerConfig):
        self.config = config
        self.base_url = self.config.base_url
        self.api_key = self.config.api_key
        self.model = self.config.model
            
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

                label = (await llm_labeler(self.api_key, self.model, self.base_url, prompt)).strip().lower()
                
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
        
    async def smart_label_queries(self, input_path: str, output_path: str, existing_labels_path: str,
                                  hybrid_target: int = 4000, sparse_target = 750):
        try:
            Lsetup = LabelSetup()

            existing_labels = Lsetup.load_existing_samples(existing_labels_path)
            seen_ids = set(existing_labels.keys())

            hybrid_count = sum(1 for v in existing_labels.values() if v['label'] == 'hybrid')
            sparse_count = sum(1 for v in existing_labels.values() if v['label'] == 'sparse')

            print(f"Existing: hybrid={hybrid_count}, sparse={sparse_count}")

            samples = Lsetup.load_samples(input_path)

            new_labels = {}

            for i, sample in enumerate(samples, 1):
                query_id = sample['id']
                query = sample['query']

                if query_id in seen_ids:
                    continue

                print(f"\n[{i}] Query: {query}")

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
                        f'Query: "{query}"'
                    )
                
                label = (await llm_labeler(self.api_key, self.model, self.base_url, prompt)).strip().lower()
                print(f"🏷️ LLM label: {label}")

                if label == "hybrid" and hybrid_count < hybrid_target:
                    hybrid_count += 1
                elif label == "sparse" and sparse_count < sparse_target:
                    sparse_count += 1
                else:
                    continue
                    
                new_labels[query_id] = {
                    "id": query_id,
                    "query": query,
                    "label": label
                }

                seen_ids.add(query_id)

                # Save progress every 50
                if (hybrid_count + sparse_count) % 50 == 0:
                    Lsetup.save_labels(new_labels, output_path)
                    print(f"💾 Progress saved ({hybrid_count} hybrid, {sparse_count} sparse)")

                if hybrid_count >= hybrid_target and sparse_count >= sparse_target:
                    print(f"\n🎯 Labeling targets reached: {hybrid_count} hybrid, {sparse_count} sparse")
                    break


            Lsetup.save_labels(new_labels, output_path)
            print(f"\n✅ Final labeled set saved to {output_path}.")

        except Exception as e:
            print(f"An unexpected error occurred while reading '{input_path}': {e}")
            return

if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Automatically label FAU sample queries using LLM.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input extracted queries JSONL file - knowledgebase/extracted_queries.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Path to the output hybrid and sparse labeled queries JSONL file - knowledgebase/additional_labeled_queries.jsonl")
    parser.add_argument("--existing", type=str, required=True, help="Path to the existing large labeled sampled queries JSONL file - knowledgebase/large_labeled_sampled_queries.jsonl")
    args = parser.parse_args()


    labeler = AutoLabeler(labeler_config)
    asyncio.run(labeler.smart_label_queries(args.input, args.output, args.existing))