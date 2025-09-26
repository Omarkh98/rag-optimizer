# ------------------------------------------------------------------------------
# encode_worker.py - An isolated process for sentence encoding. To solve error `segementation fault (core dumped)`.
# ------------------------------------------------------------------------------
import sys
import json
from sentence_transformers import SentenceTransformer

def main():
    if len(sys.argv) != 3:
        error_message = {"error": "Usage: python encode_worker.py <model_name> <query_text>"}
        print(json.dumps(error_message), file=sys.stderr)
        sys.exit(1)

    model_name = sys.argv[1]
    query_text = sys.argv[2]

    try:
        model = SentenceTransformer(model_name)
        embedding = model.encode([query_text], convert_to_numpy=True)
        
        result = {"embedding": embedding[0].tolist()}
        print(json.dumps(result))

    except Exception as e:
        error_message = {"error": f"An error occurred during encoding: {str(e)}"}
        print(json.dumps(error_message), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()