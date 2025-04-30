# ------------------------------------------------------------------------------
# query_classifier/query_labelling.py - Labelling sample queries.
# ------------------------------------------------------------------------------
"""
Loads `sample_queries.jsonl` and manually Label the samples as
'dense', 'sparse', or 'hybrid'.
"""

import random

from ..constants.my_constants import SAVE_INTERVAL

from ..helpers.labelling import LabelSetup

def label_queries(input_path: str, output_path: str):
    try:
        Lsetup = LabelSetup()

        samples = Lsetup.load_samples(input_path)

        existing_labels = Lsetup.load_existing_samples(output_path)

        labels = existing_labels.copy()

        unlabeled_samples = [s for s in samples if s["id"] not in labels]

        random.shuffle(unlabeled_samples) # Shuffle unlabeled samples

        total_samples = len(samples)
        labeled_samples = len(labels)

        print(f"Loaded {total_samples} samples, {labeled_samples} already labeled. {len(unlabeled_samples)} left to label.\n")

        count_since_last_save = 0
            
        label_map = {
            "1": "dense",
            "2": "sparse",
            "3": "hybrid",
            "skip": "skip",
            "exit": "exit"
            }
        
        print("Label the queries:\n1 = Dense\n2 = Sparse\n3 = Hybrid\n(Type 'exit' to stop early.)\n")

        for i, sample in enumerate(unlabeled_samples, start = labeled_samples + 1):
            print(f"\n[{i}/{total_samples}] Query:")
            print(sample['query'])

            label = input("Label (1-Dense / 2-Sparse / 3-Hybrid / skip / exit): ").strip().lower()

            while label not in label_map:
                print("❗Invalid label. Please enter '1', '2', or '3'.")
                label = input("Label (1-Dense / 2-Sparse / 3-Hybrid / skip / exit): ").strip().lower()

            mapped_label = label_map[label]

            if mapped_label == "skip":
                continue

            if mapped_label == "exit":
                print("\n Exiting early, saving progress...")
                break
            
            labels[sample["id"]] = {
                "id": sample["id"],
                "query": sample["query"],
                "label": mapped_label
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

    parser = argparse.ArgumentParser(description = "Manually Label FAU sample queries.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input sampled queries JSONL file - knowledgebase/small_sampled_queries.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Path to the output labeled sampled queries JSONL file - knowledgebase/small_labeled_sampled_queries.jsonl")
    args = parser.parse_args()

    label_queries(args.input, args.output)