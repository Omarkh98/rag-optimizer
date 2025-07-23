# ------------------------------------------------------------------------------
# query_classifier/retrieval_benchmark.py - Benchmarking the retrieval performance of the three retrieval methods on quries.
# ------------------------------------------------------------------------------
"""
Runs each query through the three retrieval methods and benchmarks their performance.
Performs the following Qualitative and Quantitative evaluation.
"""

# Evaluation Metrics
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

from ..query_classifier.labeler_config import (LabelerConfig,
                                               labeler_config)

from ..helpers.labelling import LabelSetup
from ..helpers.utils import (load_vector_db,
                             read_yaml)

from .llm_response import llm_response

from ..retrievers.base import retriever_config
from ..retrievers.dense import DenseRetrieval
from ..retrievers.sparse import SparseRetrieval
from ..retrievers.hybrid import HybridRetrieval

from ..constants.my_constants import (SAVE_INTERVAL,
                                      METADATA_PATH,
                                      VECTORSTORE_PATH,
                                      TOP_K,
                                      HYBRID_ALPHA,
                                      CONFIG_FILE_PATH)

from typing import List, Dict, Optional
import json
import asyncio
import argparse

class RetrievalBenchmark:
    def __init__(self, config: LabelerConfig):
        self.config = config
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.model = config.model
        self.top_k = TOP_K
        self.alpha = HYBRID_ALPHA

        self.index, self.metadata = load_vector_db(VECTORSTORE_PATH, METADATA_PATH)

        self._rouge_scorer: Optional[rouge_scorer.RougeScorer] = None
        self._similarity_model: Optional[SentenceTransformer] = None

        self.dense = DenseRetrieval(retriever_config)
        self.sparse = SparseRetrieval(retriever_config, metadata=self.metadata)
        self.hybrid = HybridRetrieval.from_config(self.alpha, retriever_config, self.metadata, self.index)

    @property
    def rouge_scorer_instance(self) -> rouge_scorer.RougeScorer:
        """Loads the ROUGE scorer model on its first access."""
        if self._rouge_scorer is None:
            print("Lazy loading ROUGE scorer...")
            self._rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            print("ROUGE scorer loaded.")
        return self._rouge_scorer

    @property
    def similarity_model_instance(self) -> SentenceTransformer:
        config_filepath = CONFIG_FILE_PATH
        config = read_yaml(config_filepath)
        """Loads the SentenceTransformer model on its first access."""
        if self._similarity_model is None:
            print("Lazy loading SentenceTransformer model (all-MiniLM-L6-v2)...")
            self._similarity_model = SentenceTransformer(config["retriever_transformer"]["transformer"])
            print("SentenceTransformer model loaded.")
        return self._similarity_model
    
    def format_docs(self, docs: List[str]) -> str:
        return "\n\n".join(docs)
    
    def build_prompt(self, query: str, context: str) -> str:
        return (
            f"You are an expert assistant. Given the following query and a list of retrieved documents, "
            f"generate the most helpful and relevant answer based only on the provided context.\n\n"
            f"Query:\n{query}\n\nRetrieved Documents:\n{context}\n\nAnswer:"
        )
    
    async def retrieve_docs(self, query: str):
        query_embeddings = self.dense._run_encode_worker(query)
        docs_dense = await self.dense.retrieve(query_embeddings, self.index, self.metadata, self.top_k)

        docs_sparse = await self.sparse.retrieve(query, self.top_k)
        
        docs_hybrid = await self.hybrid.retrieve(query, self.metadata, self.top_k)

        return docs_dense, docs_sparse, docs_hybrid
    
    def evaluate_answer(self, generated_answer: str, ground_truth: str) -> Dict[str, float]:
        if not generated_answer or not ground_truth:
            return {"bert_score_f1": 0.0, "rouge_l_f1": 0.0, "cosine_similarity": 0.0}
        
        # 1. BERTScore
        P, R, F1 = bert_score(cands=[generated_answer], refs=[ground_truth], lang="en", model_type="distilroberta-base", device="cpu")
        bert_f1 = F1.mean().item() 
        print(f"BERTScore F1: {bert_f1:.4f}")

        # 2. ROUGE-L Score
        rouge_scores = self.rouge_scorer_instance.score(ground_truth, generated_answer)
        rouge_l_f1 = rouge_scores['rougeL'].fmeasure
        print(f"ROUGE-L F1: {rouge_l_f1:.4f}")

        # 3. Sentence-BERT Cosine Similarity
        embedding_gen = self.similarity_model_instance.encode(generated_answer, convert_to_tensor=True)
        embedding_gt = self.similarity_model_instance.encode(ground_truth, convert_to_tensor=True)
        cosine_sim = util.pytorch_cos_sim(embedding_gen, embedding_gt).item()
        print(f"Cosine Similarity: {cosine_sim:.4f}")

        return {
            "bert_score_f1": round(bert_f1, 4),
            "rouge_l_f1": round(rouge_l_f1, 4),
            "cosine_similarity": round(cosine_sim, 4)
        }
    
    async def run_benchmark(self, input_path: str, output_path: str):
        Lsetup = LabelSetup()
        all_samples = Lsetup.load_samples(input_path)
        print(f"✅ Loaded {len(all_samples)} samples from '{input_path}'.")

        existing_results = Lsetup.load_existing_samples(output_path)
        processed_ids = set(existing_results.keys())
        print(f"✅ Found {len(processed_ids)} already processed samples. Skipping them.")

        samples_to_process = [s for s in all_samples if s.get("id") not in processed_ids]

        results_batch = []

        for i, sample in enumerate(samples_to_process, 1):
            query_id = sample.get("id")
            query = sample.get("query")
            ground_truth = sample.get("ground_truth")

            if not all([query_id, query]):
                print(f"⚠️ Skipping sample at index {i-1} due to missing 'id', 'query', or 'ground_truth'.")
                continue

            print(f"\n--- [Processing Query {i}/{len(samples_to_process)}] ID: {query_id} ---")
            print(f"Query: {query}")

            try:
                # 1. Retrieve documents from all pipelines
                print("📄 Retrieving documents for all methods...")
                docs_dense, docs_sparse, docs_hybrid = await self.retrieve_docs(query)

                contexts = {
                "dense": self.format_docs(docs_dense),
                "sparse": self.format_docs(docs_sparse),
                "hybrid": self.format_docs(docs_hybrid)
                }

                # 2. Generate an answer for each retrieval method
                print("🤖 Generating answers with LLM for each context...")
                prompts = {method: self.build_prompt(query, context) for method, context in contexts.items()}

                gen_dense, gen_sparse, gen_hybrid = await asyncio.gather(
                    llm_response(self.api_key, self.model, self.base_url, prompts["dense"]),
                    llm_response(self.api_key, self.model, self.base_url, prompts["sparse"]),
                    llm_response(self.api_key, self.model, self.base_url, prompts["hybrid"])
                )

                generated_answers = {
                "dense": gen_dense.strip(),
                "sparse": gen_sparse.strip(),
                "hybrid": gen_hybrid.strip()
                }

                # 3. Evaluate each generated answer
                print("📈 Evaluating generated answers...")
                eval_results = {}

                for method, answer in generated_answers.items():
                    eval_results[method] = {
                        "generated_answer": answer,
                        "retrieved_context": contexts[method],
                        "scores": self.evaluate_answer(answer, ground_truth)
                    }
                
                # 4. Assemble the final result object for this query
                final_result = {
                    "id": query_id,
                    "query": query,
                    "query_category": sample.get("category", "unknown"),
                    "ground_truth": ground_truth,
                    "dense_result": eval_results["dense"],
                    "sparse_result": eval_results["sparse"],
                    "hybrid_result": eval_results["hybrid"]
                }
                results_batch.append(final_result)

            except Exception as e:
                print(f"❌ Error processing query ID {query_id}: {e}")
                results_batch.append({"id": query_id, "query": query, "status": "error", "reason": str(e)})
                continue

            # 5. Save progress periodically
            if len(results_batch) >= SAVE_INTERVAL:
                print(f"\n💾 Saving batch of {len(results_batch)} results to disk...")
                # Append to file
                with open(output_path, 'a', encoding='utf-8') as f:
                    for item in results_batch:
                        f.write(json.dumps(item) + '\n')
                results_batch = []

        if results_batch:
            print(f"\n💾 Saving final batch of {len(results_batch)} results to disk...")
            with open(output_path, 'a', encoding='utf-8') as f:
                for item in results_batch:
                    f.write(json.dumps(item) + '\n')

        print("\n✅ Benchmark complete! All queries processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a full RAG benchmark with multiple retrieval methods and evaluations.")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the input benchmark dataset (e.g., categorized_queries.jsonl)."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Path to save the detailed evaluation results JSONL file."
    )
    args = parser.parse_args()

    benchmarker = RetrievalBenchmark(labeler_config)
    asyncio.run(benchmarker.run_benchmark(args.input, args.output))