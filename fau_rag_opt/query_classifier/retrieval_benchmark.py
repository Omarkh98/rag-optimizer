# ------------------------------------------------------------------------------
# query_classifier/retrieval_benchmark.py - Benchmarking the retrieval performance of the three retrieval methods on quries.
# ------------------------------------------------------------------------------
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
        if self._rouge_scorer is None:
            self._rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            print("ROUGE scorer loaded.")
        return self._rouge_scorer

    @property
    def similarity_model_instance(self) -> SentenceTransformer:
        config_filepath = CONFIG_FILE_PATH
        config = read_yaml(config_filepath)
        if self._similarity_model is None:
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
        
        P, R, F1 = bert_score(cands=[generated_answer], refs=[ground_truth], lang="en", model_type="distilroberta-base", device="cpu")
        bert_f1 = F1.mean().item() 
        print(f"BERTScore F1: {bert_f1:.4f}")

        rouge_scores = self.rouge_scorer_instance.score(ground_truth, generated_answer)
        rouge_l_f1 = rouge_scores['rougeL'].fmeasure
        print(f"ROUGE-L F1: {rouge_l_f1:.4f}")

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

        existing_results = Lsetup.load_existing_samples(output_path)
        processed_ids = set(existing_results.keys())

        samples_to_process = [s for s in all_samples if s.get("id") not in processed_ids]

        results_batch = []

        for i, sample in enumerate(samples_to_process, 1):
            query_id = sample.get("id")
            query = sample.get("query")
            ground_truth = sample.get("ground_truth")

            if not all([query_id, query]):
                continue

            print(f"\n Processing Query {i}/{len(samples_to_process)}] ID: {query_id}")
            print(f"Query: {query}")

            try:
                docs_dense, docs_sparse, docs_hybrid = await self.retrieve_docs(query)

                contexts = {
                "dense": self.format_docs(docs_dense),
                "sparse": self.format_docs(docs_sparse),
                "hybrid": self.format_docs(docs_hybrid)
                }

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

                eval_results = {}

                for method, answer in generated_answers.items():
                    eval_results[method] = {
                        "generated_answer": answer,
                        "retrieved_context": contexts[method],
                        "scores": self.evaluate_answer(answer, ground_truth)
                    }
                
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
                results_batch.append({"id": query_id, "query": query, "status": "error", "reason": str(e)})
                continue

            if len(results_batch) >= SAVE_INTERVAL:
                with open(output_path, 'a', encoding='utf-8') as f:
                    for item in results_batch:
                        f.write(json.dumps(item) + '\n')
                results_batch = []

        if results_batch:
            with open(output_path, 'a', encoding='utf-8') as f:
                for item in results_batch:
                    f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a full RAG benchmark with multiple retrieval methods and evaluations.")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True
    )
    args = parser.parse_args()

    benchmarker = RetrievalBenchmark(labeler_config)
    asyncio.run(benchmarker.run_benchmark(args.input, args.output))