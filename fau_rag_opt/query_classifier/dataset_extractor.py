# ------------------------------------------------------------------------------
# query_classifier/dataset_extractor.py - Query extraction from user-assistant conversations.
# ------------------------------------------------------------------------------
import json

def extract_user_query_answer(input_path: str, output_path: str):
    extracted = []
    try:

        with open(input_path, 'r', encoding = 'utf-8') as infile:
            for line in infile:
                record = json.loads(line)
                conversations = record.get("conversations", [])
                if conversations and conversations[0]["role"] == "user" and conversations[1]["role"] == "assistant":
                    query_content = conversations[0]["content"]
                    query_answer = conversations[1]["content"]
                    query_id = record.get("id", "")
                    extracted.append({
                        "id": query_id,
                        "query": query_content,
                        "answer": query_answer
                    })

        with open(output_path, 'w', encoding = 'utf-8') as outfile:
            for item in extracted:
                outfile.write(json.dumps(item, ensure_ascii = False) + "\n")
        
    except Exception as e:
        return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description = "Extract first user queries and assistant responses from FAU conversations dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSONL file - knowledgebase/convos.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Path to the output JSONL file  - knowledgebase/extracted_queries.jsonl")
    args = parser.parse_args()

    extract_user_query_answer(args.input, args.output)