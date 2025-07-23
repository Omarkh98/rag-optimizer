# ------------------------------------------------------------------------------
# merge_datasets.py - Safely merges one JSONL or JSON file into another JSONL file.
# ------------------------------------------------------------------------------

import json
import os
import argparse

class DatasetMerger:
    """
    Handles loading and merging of JSON and JSONL data files.
    """
    def __init__(self):
        pass

    def load_existing_ids(self, file_path: str) -> set:
        """
        Efficiently loads only the 'id' fields from a JSONL file into a set
        for fast lookups.
        """
        if not os.path.exists(file_path):
            return set()
        
        processed_ids = set()
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    processed_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"⚠️  Skipping malformed line in '{file_path}': {e}")
                    continue
        return processed_ids

    def load_source_data(self, file_path: str) -> list:
        """
        Loads data from a source file, intelligently handling both
        standard JSON arrays and JSONL formats.
        """
        if not os.path.exists(file_path):
            print(f"❌ Source file '{file_path}' not found.")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == '[':
                # It's a standard JSON array file
                print(f"ℹ️  Detected standard JSON array format in '{file_path}'.")
                try:
                    return json.load(f)
                except json.JSONDecodeError as e:
                    print(f"❌ Error parsing JSON array in '{file_path}': {e}")
                    return []
            else:
                print(f"ℹ️  Detected JSONL format in '{file_path}'.")
                samples = []
                for line in f:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Skipping malformed JSON line in '{file_path}': {e}")
                return samples

    def merge_files(self, source_path: str, destination_path: str):
        """
        Merges new samples from a source file into a destination file,
        avoiding duplicates based on the 'id' field.
        """
        try:
            existing_ids = self.load_existing_ids(destination_path)
            print(f"📁 Found {len(existing_ids)} existing samples in '{destination_path}'.")

            new_data = self.load_source_data(source_path)
            if not new_data:
                print("No new data to merge.")
                return

            new_samples_merged = 0
            skipped_samples = 0

            with open(destination_path, 'a', encoding='utf-8') as dest_file:
                for item in new_data:
                    try:
                        if "id" not in item:
                            print(f"⚠️  Skipping item because it has no 'id' field: {item}")
                            continue
                        
                        if item["id"] not in existing_ids:
                            dest_file.write(json.dumps(item) + '\n')
                            existing_ids.add(item["id"])
                            new_samples_merged += 1
                        else:
                            skipped_samples += 1
                    except Exception as e:
                        print(f"⚠️  Could not process item: {item}. Error: {e}")
                        continue
            
            print("\n--- Merge Complete ---")
            print(f"✅ Merged {new_samples_merged} new samples into '{destination_path}'.")
            if skipped_samples > 0:
                print(f"ℹ️  Skipped {skipped_samples} samples that were already present.")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merges new samples from a source JSON or JSONL file into a destination JSONL file, avoiding duplicates."
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the new labeled queries file (e.g., knowledgebase/no_answer_queries.jsonl)."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Path to the destination JSONL file to merge into (e.g., knowledgebase/categorized_queries.jsonl)."
    )
    args = parser.parse_args()

    handler = DatasetMerger()
    handler.merge_files(args.input, args.output)
