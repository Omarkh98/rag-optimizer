# ------------------------------------------------------------------------------
# query_classifier/dataset_categorization.py - Employing an LLM to categorize queries into: Factual, Comparatize, or Ambiguous.
# ------------------------------------------------------------------------------
import json
import random
import argparse
import asyncio
from collections import defaultdict

from ..constants.my_constants import SAVE_INTERVAL

from ..helpers.labelling import LabelSetup
from ..query_classifier.labeler_config import (LabelerConfig,
                                               labeler_config)

from .llm_response import llm_response

class DatasetCategorizer:
    def __init__(self, config: LabelerConfig):
        self.config = config
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.model = config.model

    def _get_prompt(self, query: str) -> str:
        return (
            "You are an expert data annotator. Your task is to categorize user queries from a university's help system "
            "into one of three categories: \"factual\", \"comparative\", or \"ambiguous\".\n\n"
            "Analyze the following user query and respond ONLY with a JSON object containing the \"category\" key.\n\n"
            "Here are the definitions of the categories:\n"
            "- \"factual\": The query asks for a direct, specific piece of information. "
            "Example: \"What are the opening hours of the library?\"\n"
            "- \"comparative\": The query asks to compare two or more distinct things. It often contains words like "
            "\"vs\", \"compare\", \"differences\", or \"similar to\". "
            "Example: \"What is the difference between the AI and Data Science master's programs?\"\n"
            "- \"ambiguous\": The query is vague, lacks context, or could be interpreted in multiple ways. "
            "A good assistant would likely need to ask for clarification. "
            "Example: \"What about the requirements?\"\n\n"
            f"User Query:\n\"{query}\""
        )

    async def categorize_queries(self, input_path: str, output_path: str, target_per_category: int):
        try:
            Lsetup = LabelSetup()

            raw_queries = Lsetup.load_samples(input_path)
            if not raw_queries:
                return
            
            all_queries = []
            for data in raw_queries:
                try:
                    all_queries.append({
                    "id": data.get("id"),
                    "query": data["query"],
                    "ground_truth": data["answer"]
                })
                except (KeyError, IndexError) as e:
                    print(f"Skipping faulty entry in {input_path}: {e}")

            random.shuffle(all_queries)
            
            existing_data = Lsetup.load_existing_samples(output_path)

            category_counts = defaultdict(int)
            for item in existing_data.values():
                category_counts[item['category']] += 1
            
            categorized_list = list(existing_data.values())
            processed_ids = set(existing_data.keys())
            queries_to_process = [q for q in all_queries if q['id'] not in processed_ids]

            print("Initial State:")
            print(f"Loaded {len(all_queries)} total queries from source.")
            print(f"Found {len(existing_data)} already categorized queries.")
            print(f"Processing up to {len(queries_to_process)} new queries.")
            print(f"Current counts: {dict(category_counts)}")
            print(f"Target per category: {target_per_category}")

            count_since_last_save = 0
            total_queries_parsed = 0

            for query_data in queries_to_process:
                total_queries_parsed += 1

                required_categories = ['factual', 'comparative', 'ambiguous']
                if all(category_counts.get(category, 0) >= target_per_category for category in required_categories):
                    print("\n All category targets have been met. Stopping early.")
                    break

                print(f"[{total_queries_parsed}/{len(queries_to_process)}] Processing Query ID: {query_data['id']}")
                print(f"Query: {query_data['query']}")

                prompt = self._get_prompt(query_data['query'])
                try:
                    response_str = await llm_response(self.api_key, self.model, self.base_url, prompt)
                    
                    raw_response = response_str.strip()
                    if raw_response.startswith("```json"):
                        raw_response = raw_response[len("```json"):].strip()
                    if raw_response.endswith("```"):
                        raw_response = raw_response[:-3].strip()

                    response_json = json.loads(raw_response)
                    category = response_json.get("category")

                except (json.JSONDecodeError, AttributeError) as e:
                    continue

                if not category or category not in ['factual', 'comparative', 'ambiguous']:
                    continue

                if category_counts[category] < target_per_category:
                    print(f" Categorized as '{category}'. Appending to dataset.")
                    category_counts[category] += 1
                    
                    new_entry = {
                        "id": query_data["id"],
                        "query": query_data["query"],
                        "ground_truth": query_data["ground_truth"],
                        "category": category
                    }
                    categorized_list.append(new_entry)
                    count_since_last_save += 1
                else:
                    print(f" Categorized as '{category}', but target of {target_per_category} is already met. Skipping query.")

                print(f"Current counts: {dict(category_counts)}\n")

                if count_since_last_save >= SAVE_INTERVAL:
                    Lsetup.save_labels(categorized_list, output_path)
                    count_since_last_save = 0
            
            Lsetup.save_labels(categorized_list, output_path)
            print("\nCategorization Completed:")
            print(f"Final dataset saved to '{output_path}'.")
            
        except Exception as e:
            print(f"An unexpected error occurred while reading '{input_path}': {e}")
            return
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Categorize queries into factual, comparative, and ambiguous categories."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to `extracted_queries.jsonl`.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--target", type=int, default=50, help="Target number of queries for each category.")
    args = parser.parse_args()

    categorizer = DatasetCategorizer(labeler_config)
    asyncio.run(categorizer.categorize_queries(
        input_path=args.input,
        output_path=args.output,
        target_per_category=args.target
    ))