# ------------------------------------------------------------------------------
# query_classifier/dataset_extractor.py - Query extraction from user-assistant conversations.
# ------------------------------------------------------------------------------
"""
Loads `convos.jsonl` and extracts the initial user query asked to the assistant
as well as the conversation id.
"""

import json

def extract_user_query(input_path: str, output_path: str):
    extracted = []
    try:

        with open(input_path, 'r', encoding = 'utf-8') as infile:
            for line in infile:
                record = json.loads(line)
                conversations = record.get("conversations", [])
                if conversations and conversations[0]["role"] == "user":
                    query_content = conversations[0]["content"]
                    query_id = record.get("id", "")
                    extracted.append({
                        "id": query_id,
                        "query": query_content
                    })

        with open(output_path, 'w', encoding = 'utf-8') as outfile:
            for item in extracted:
                outfile.write(json.dumps(item, ensure_ascii = False) + "\n")
        
        print(f"✅ Extracted {len(extracted)} user queries to {output_path}")

    except Exception as e:
        print(f"An unexpected error occurred while reading '{input_path}': {e}")
        return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description = "Extract first user queries from FAU conversations dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSONL file - knowledgebase/convos.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Path to the output JSONL file  - knowledgebase/extracted_queries.jsonl")
    args = parser.parse_args()

    extract_user_query(args.input, args.output)