# ------------------------------------------------------------------------------
# helpers/labelling.py - Labelling saving and loading.
# ------------------------------------------------------------------------------
"""
Loads `sampled_queries.jsonl` samples and saves the labels to the samples.
"""

import json

class LabelSetup():
    def __init__(self):
        pass

    def load_samples(self, sampled_path: str):
        samples = []

        with open(sampled_path, 'r') as file:
            for line in file:
                samples.append(json.loads(line.strip()))
        return samples
    
    def load_existing_samples(self, labelled_path: str):
        try:
            labels = {}

            with open(labelled_path, 'r') as file:
                for line in file:
                    entry = json.loads(line)
                    labels[entry["id"]] = entry

            return labels
        except FileNotFoundError:
            print(f"❌ Input file '{labelled_path}' not found.")
    
    def save_labels(self, labels: dict, labelled_path: str):
        with open(labelled_path, 'w') as file:
            for  label_entry in labels.values():
                json.dump(label_entry, file)
                file.write("\n")

    
    def merge_samples(self, input_path: str, output_path: str):
        try:
            existing_samples = self.load_existing_samples(output_path)
            print(f"📁 Loaded {len(existing_samples)} existing samples from {output_path}")

            new_samples = 0

            with open(input_path, 'r', encoding = 'utf-8') as infile:
                for line in infile:
                    item = json.loads(line)
                    if item["id"] not in existing_samples:
                        existing_samples[item["id"]] = item
                        new_samples += 1

            with open(output_path, 'w', encoding = 'utf-8') as outfile:
                for item in existing_samples.values():
                    outfile.write(json.dumps(item, ensure_ascii = False) + "\n")

            print(f"✅ Merged {new_samples} new samples from {input_path} into {output_path}")

        except FileNotFoundError:
            print(f"❌ Input file '{input_path}' not found.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=" Merges samples from a single source JSONL file into the destination JSONL file")
    parser.add_argument("--input", type=str, required=True, help="Path to the new labeled queries JSONL file - knowledgebase/additional_labaled_queries.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Path to the output labeled queries JSONL file - knowledgebase/large_labeled_sampled_queries.jsonl")
    args = parser.parse_args()

    Lsetup = LabelSetup()
    Lsetup.merge_samples(args.input, args.output)