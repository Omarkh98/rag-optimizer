# ------------------------------------------------------------------------------
# encode_worker.py - An isolated process for sentence encoding.
# ------------------------------------------------------------------------------
"""
This script is designed to be called as a separate process to avoid library
conflicts. It loads the SentenceTransformer model, encodes a single query,
prints the resulting vector as a JSON string to standard output, and then exits.
"""
import sys
import json
from sentence_transformers import SentenceTransformer
import numpy as np

def main():
    """
    Takes a model name and a query from command-line arguments,
    encodes the query, and prints the vector.
    """
    if len(sys.argv) != 3:
        error_message = {"error": "Usage: python encode_worker.py <model_name> <query_text>"}
        print(json.dumps(error_message), file=sys.stderr)
        sys.exit(1)

    model_name = sys.argv[1]
    query_text = sys.argv[2]

    try:
        # Load the model, do the work, and exit.
        model = SentenceTransformer(model_name)
        embedding = model.encode([query_text], convert_to_numpy=True)
        
        # Convert numpy array to a list for JSON serialization and print.
        result = {"embedding": embedding[0].tolist()}
        print(json.dumps(result))

    except Exception as e:
        error_message = {"error": f"An error occurred during encoding: {str(e)}"}
        print(json.dumps(error_message), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()