import logging
import faiss
import yaml
import json
import sys
import os
import numpy as np


from ensure import ensure_annotations
from box import config_box
from pathlib import Path
from box.exceptions import BoxValueError
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

from fau_rag_opt.helpers.exception import CustomException


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> config_box.ConfigBox:
    """
    Read Yaml file and return Configbox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"Yaml file: {path_to_yaml} loadad successfully!")
            return config_box.ConfigBox(content)
    except BoxValueError:
        raise ValueError("Yaml file is empty")
    except Exception as e:
        raise e
    
def preprocess_knowledgebase(knowledgebase):
    """
    Convert knowledgebase into lower case letters
    """
    corpus = []
    for entry in knowledgebase:
        tokens = word_tokenize(entry["text"].lower())
        joined = ''.join(tokens)
        corpus.append(joined)
    return corpus

def load_vector_db(index_file, metadata_file):
    try:
        # Check if the FAISS index file exists
        if not os.path.exists(index_file):
            print(f"FAISS index not found at {index_file}. Creating a new one...")
            create_vector_db_from_jsonl(metadata_file, index_file)

        # Load FAISS index
        print("Loading FAISS index...")
        index = faiss.read_index(index_file)

        # Load metadata from JSONL
        metadata = []
        with open(metadata_file, "r") as f:
            for line in f:
                metadata.append(json.loads(line.strip()))
        
        logging.info("Data Loaded From Vector DB Successfully (FAISS and JSONL)!")
        return index, metadata
    except Exception as e:
        raise CustomException(e, sys)
    
def create_vector_db_from_jsonl(metadata_file, index_file):
    try:
        # Load metadata from JSONL
        metadata = []
        with open(metadata_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())  # Parse each line as JSON
                    if isinstance(entry, list):  # Handle unexpected nested lists
                        metadata.extend(entry)
                    elif isinstance(entry, dict):  # Normal case
                        metadata.append(entry)
                    else:
                        print(f"Skipping unexpected entry type: {type(entry)}")
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {line.strip()} (Error: {e})")

        # Validate metadata structure
        if not all(isinstance(entry, dict) and "text" in entry for entry in metadata):
            raise ValueError("Metadata entries must be dictionaries with a 'text' key.")

        # Initialize embedding model
        print("Initializing embedding model...")
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        # Generate embeddings for each chunk's text
        print("Generating embeddings for metadata...")
        embeddings = []
        for entry in metadata:
            try:
                embeddings.append(model.encode(entry["text"]))
            except KeyError:
                print(f"Skipping entry without 'text': {entry}")

        embeddings = np.array(embeddings, dtype="float32")  # Convert to NumPy array

        # Create FAISS index
        print("Creating FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

        # Save FAISS index
        print(f"Saving FAISS index to {index_file}...")
        faiss.write_index(index, index_file)

        print("FAISS index created and saved successfully.")

    except Exception as e:
        raise RuntimeError(f"Error creating FAISS index: {e}")