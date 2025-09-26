# ------------------------------------------------------------------------------
# ahr_training_data_generator.py - Creates the training set for the AHR model.
# ------------------------------------------------------------------------------
import pandas as pd
import json
import argparse
import asyncio
import numpy as np
from typing import List, Dict, Set
from pathlib import Path

from bert_score import score as bert_score_calc

from ..helpers.labelling import LabelSetup

from ..retrievers.base import retriever_config
from ..retrievers.dense import DenseRetrieval
from ..retrievers.hybrid import HybridRetrieval

from .llm_response import llm_response

from ..query_classifier.labeler_config import (LabelerConfig,
                                               labeler_config)

from ..helpers.utils import load_vector_db

from ..constants.my_constants import (METADATA_PATH,
                                      VECTORSTORE_PATH,
                                      TOP_K,
                                      HYBRID_ALPHA)

ALPHA_STEPS = np.linspace(0, 1, 11)

class AHRDataGenerator:
    def __init__(self, config: LabelerConfig):
        self.config = config
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.model = config.model
        self.top_k = TOP_K
        self.alpha = HYBRID_ALPHA

        self.index, self.metadata = load_vector_db(VECTORSTORE_PATH, METADATA_PATH)

        self.dense = DenseRetrieval(retriever_config)
        self.hybrid = HybridRetrieval.from_config(self.alpha, retriever_config, self.metadata, self.index)

    def format_docs(self, docs: List[str]) -> str:
        return "\n\n".join(docs)
    
    def build_prompt(self, query: str, context: str) -> str:
        return (
            f"You are an expert assistant. Given the following query and a list of retrieved documents, "
            f"generate the most helpful and relevant answer based only on the provided context.\n\n"
            f"Query:\n{query}\n\nRetrieved Documents:\n{context}\n\nAnswer:"
        )
    
    async def find_optimal_alpha(self, query: str, ground_truth: str) -> Dict:
        if not query or not ground_truth:
            return {}

        best_score = -1.0
        optimal_alpha = 0.5
        
        query_embedding = self.dense._run_encode_worker(query)

        per_alpha_results = []

        for alpha in ALPHA_STEPS:
            print(f"Testing alpha = {alpha:.1f}")
            
            hybrid_retriever = HybridRetrieval.from_config(alpha, retriever_config, self.metadata, self.index)
            
            retrieved_docs = await hybrid_retriever.retrieve(query, self.metadata, self.top_k)
            context = self.format_docs(retrieved_docs)
            
            prompt = self.build_prompt(query, context)
            generated_answer = await llm_response(self.api_key, self.model, self.base_url, prompt)
            
            if not generated_answer:
                continue
            
            _, _, f1 = bert_score_calc(cands=[generated_answer], refs=[ground_truth], lang="en", model_type="distilroberta-base")
            current_score = f1.mean().item()
            per_alpha_results.append({
            "alpha": alpha,
            "generated_answer": generated_answer,
            "score": current_score,
            "retrieved_docs": retrieved_docs,
            "prompt": prompt
            })
            
            if current_score > best_score:
                best_score = current_score
                optimal_alpha = alpha
        
        print(f"Optimal alpha for this query: {optimal_alpha} (Score: {best_score:.4f})")
        return {
            "query_text": query,
            "query_embedding": json.dumps(query_embedding[0].tolist()),
            "optimal_alpha": optimal_alpha,
            "per_alpha_results": per_alpha_results
        }

    def get_processed_queries(self, output_path: str) -> Set[str]:
        output_file = Path(output_path)
        if not output_file.exists() or output_file.stat().st_size == 0:
            return set()
        try:
            df = pd.read_csv(output_path)
            return set(df['query_text'].unique())
        except pd.errors.EmptyDataError:
            return set()
    
    async def generate_training_data(self, input_path: str, output_path: str):
        label_setup = LabelSetup()
        all_samples = label_setup.load_samples(input_path)

        processed_queries = self.get_processed_queries(output_path)
        if processed_queries:
            print(f"Found {len(processed_queries)} already processed queries. Resuming session...")

        samples_to_process = [s for s in all_samples if s.get("query") not in processed_queries]

        per_alpha_output_path = output_path + ".per_alpha.jsonl"

        with open(output_path, 'a', encoding='utf-8') as f, \
            open(per_alpha_output_path, 'a', encoding='utf-8') as per_alpha_f:
            if not processed_queries:
                f.write("query_text,query_embedding,optimal_alpha\n")
                
            for i, sample in enumerate(samples_to_process, 1):
                query = sample.get("query")
                ground_truth = sample.get("ground_truth")
                print(f"\n[Processing Query {i}/{len(samples_to_process)}]")
                try:
                    result = await self.find_optimal_alpha(query, ground_truth)
                    if result:
                        escaped_query_text = result['query_text'].replace('"', '""')
                        query_text_csv = f'"{escaped_query_text}"'
                        query_embedding_csv = f"\"{result['query_embedding']}\""
                        optimal_alpha_csv = str(result['optimal_alpha'])
                        csv_line = f"{query_text_csv},{query_embedding_csv},{optimal_alpha_csv}\n"
                        f.write(csv_line)
                        f.flush()

                        per_alpha_record = {
                            "query_text": result['query_text'],
                            "optimal_alpha": result['optimal_alpha'],
                            "per_alpha_results": result['per_alpha_results'],
                            "ground_truth": ground_truth
                        }
                        per_alpha_f.write(json.dumps(per_alpha_record, ensure_ascii=False) + "\n")
                        per_alpha_f.flush()
                except Exception as e:
                    continue

        print(f"\n Training data generation completed. All results saved to '{output_path}' and '{per_alpha_output_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data for the AHR regression model.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    generator = AHRDataGenerator(labeler_config)
    asyncio.run(generator.generate_training_data(args.input, args.output))