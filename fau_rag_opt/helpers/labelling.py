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
            return {}
    
    def save_labels(self, labels: dict, labelled_path: str):
        with open(labelled_path, 'w') as file:
            for  label_entry in labels.values():
                json.dump(label_entry, file)
                file.write("\n")
