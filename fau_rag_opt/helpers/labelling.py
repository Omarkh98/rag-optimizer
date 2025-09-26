# ------------------------------------------------------------------------------
# helpers/labelling.py - Utilities for loading, saving, and merging labeled samples.
# ------------------------------------------------------------------------------
import json
import os

class LabelSetup():
    def __init__(self):
        pass

    def load_samples(self, sampled_path: str):
        if not os.path.exists(sampled_path):
            return []
        samples = []
        with open(sampled_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    samples.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"Skipping faulty JSON line in {sampled_path}")
        return samples
    
    def already_processed(self, output_path: str):
        processed_ids = set()
        if os.path.exists(output_path):
            with open(output_path, 'r') as file:
                for line in file:
                    try:
                        processed_ids.add(json.loads(line)["id"])               
                    except Exception:
                        print(f"Error")
                        continue
        return processed_ids
    
    def load_existing_samples(self, labelled_path: str):
        if not os.path.exists(labelled_path):
            return {}
        try:
            labels = {}
            with open(labelled_path, 'r', encoding='utf-8') as file:
                for line in file:
                    entry = json.loads(line)
                    labels[entry["id"]] = entry
            return labels
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Error reading existing samples from '{labelled_path}': {e}")
            return {}
        
    def save_labels(self, labels: list, labelled_path: str):
        with open(labelled_path, 'w', encoding='utf-8') as file:
            for label_entry in labels:
                json.dump(label_entry, file)
                file.write("\n")
    
    def merge_samples(self, input_path: str, output_path: str):
        try:
            existing_samples = self.load_existing_samples(output_path)
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

        except FileNotFoundError:
            print(f"Input file '{input_path}' not found.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=" Merges samples from a single source JSONL file into the destination JSONL file")
    parser.add_argument("--input", type=str, required=True, help="Path to the new labeled queries JSONL file - knowledgebase/additional_labaled_queries.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Path to the output labeled queries JSONL file - knowledgebase/large_labeled_sampled_queries.jsonl")
    args = parser.parse_args()

    Lsetup = LabelSetup()
    Lsetup.merge_samples(args.input, args.output)