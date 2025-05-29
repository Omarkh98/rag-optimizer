# ------------------------------------------------------------------------------
# query_classifier/eval_asnwer_quality.py - Measure how close a generated answer is to a known correct (ground truth) answer
# ------------------------------------------------------------------------------
"""
RAG evaluation via answer quality, using the similarity between generated answers and ground truth answers
to infer which retrieval method (dense, sparse, hybrid) worked best for each query.
"""

from bert_score import score as bert_score

from ..query_classifier.labeler_config import LabelerConfig
from ..helpers.labelling import LabelSetup

from .llm_response import query_university_endpoint
from ..helpers.utils import load_vector_db

from ..constants.my_constants import SAVE_INTERVAL

from ..retrievers.base import retriever_config
from ..retrievers.dense import DenseRetrieval
from ..retrievers.sparse import SparseRetrieval
from ..retrievers.hybrid import HybridRetrieval

from ..constants.my_constants import (METADATA_PATH,
                                                VECTORSTORE_PATH,
                                                TOP_K, HYBRID_ALPHA)
from typing import List, Dict
import aiofiles
import json
import aiohttp

class RetrieverEvaluator:
    def __init__(self, config: LabelerConfig):
        self.config = config
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.model = config.model

        self.index, self.metadata = load_vector_db(VECTORSTORE_PATH, METADATA_PATH)

        self.dense = DenseRetrieval(retriever_config)
        self.sparse = SparseRetrieval(retriever_config, metadata=self.metadata)
        self.hybrid = HybridRetrieval.from_config(HYBRID_ALPHA, retriever_config, self.metadata, self.index)

        self.top_k = TOP_K

    def format_docs(self, docs: List[str]) -> str:
        return "\n\n".join(docs)

    def get_best_method(self, scores: Dict[str, float]) -> str:
        return max(scores, key=scores.get)

    def compare_retrieval_methods(self, gen_answer: str, gt_answer: str) -> float:
        P, R, F1 = bert_score(
            cands = [gen_answer],
            refs = [gt_answer],
            lang = "en",
            model_type = "distilroberta-base",
            device = "cpu"
        )
        return F1[0].item()
    
    def build_prompt(self, query: str, context: str) -> str:
        return (
            f"You are an expert assistant. Given the following query and a list of retrieved documents, "
            f"generate the most helpful and relevant answer.\n\n"
            f"Query:\n{query}\n\nRetrieved Documents:\n{context}\n\nAnswer:"
        )

    async def retrieve_docs(self, query: str):
        query_embeddings = self.dense.encode_query(query)
        docs_dense = await self.dense.retrieve(query_embeddings, self.index, self.metadata, self.top_k)

        docs_sparse = await self.sparse.retrieve(query, self.top_k)
        
        docs_hybrid = await self.hybrid.retrieve(query, self.metadata, self.top_k)

        return docs_dense, docs_sparse, docs_hybrid

    async def generate_answers(self, input_path: str, output_path: str):
        Lsetup = LabelSetup()
        samples = Lsetup.load_samples(input_path)
        print(f"✅ Loaded {len(samples)} samples.\n")

        already_processed = Lsetup.already_processed(output_path)

        samples_to_process = [s for s in samples if s["id"] not in already_processed]

        labeled_batch = []

        async with aiohttp.ClientSession() as session:
            for i, sample in enumerate(samples_to_process):
                query = sample["query"]
                query_id = sample["id"]
                gt_answer = sample["answer"]
                print(f"🔍 [{i+1}/{len(samples_to_process)}] Processing query: {query_id}")

                try:
                    print("📄 Retrieving documents...")
                    docs_dense, docs_sparse, docs_hybrid = await self.retrieve_docs(query)

                    dense_context = self.format_docs(docs_dense)
                    sparse_context = self.format_docs(docs_sparse)
                    hybrid_context = self.format_docs(docs_hybrid)

                    print("🔁 Generating answers with LLM for each retrieval method...")
                    dense_prompt = self.build_prompt(query, dense_context)
                    sparse_prompt = self.build_prompt(query, sparse_context)
                    hybrid_prompt = self.build_prompt(query, hybrid_context)

                    A_dense, A_sparse, A_hybrid = await asyncio.gather(
                        query_university_endpoint(session, self.api_key, self.model, self.base_url, dense_prompt),
                        query_university_endpoint(session, self.api_key, self.model, self.base_url, sparse_prompt),
                        query_university_endpoint(session, self.api_key, self.model, self.base_url, hybrid_prompt)
                    )

                    A_dense = A_dense.strip().lower()
                    A_sparse = A_sparse.strip().lower()
                    A_hybrid = A_hybrid.strip().lower()

                    score_dense = self.compare_retrieval_methods(A_dense, gt_answer)
                    score_sparse = self.compare_retrieval_methods(A_sparse, gt_answer)
                    score_hybrid = self.compare_retrieval_methods(A_hybrid, gt_answer)

                    scores = {"dense": score_dense, "sparse": score_sparse, "hybrid": score_hybrid}
                    best_method = self.get_best_method(scores)

                    answer_map = {
                        "dense": A_dense,
                        "sparse": A_sparse,
                        "hybrid": A_hybrid
                    }

                    best_answer = answer_map[best_method]

                    labeled_entry = {
                        "id": query_id,
                        "query": query,
                        "gt_answer": gt_answer,
                        "scores": scores,
                        "label": best_method,
                        "best_answer": best_answer
                    }
                    
                    labeled_batch.append(labeled_entry)
                except Exception as e:
                    print(f"❌ Error processing ID {query_id}: {e}")
                    continue

                if len(labeled_batch) >= SAVE_INTERVAL:
                    print(f"\n💾 Saving batch of {len(labeled_batch)} disk...")
                    async with aiofiles.open(output_path, 'a') as out_file:
                        for item in labeled_batch:
                            await out_file.write(json.dumps(item) + "\n")
                    labeled_batch = []

            if labeled_batch:
                print(f"💾 Saving final batch of {len(labeled_batch)} to disk...")
                async with aiofiles.open(output_path, "a") as out_file:
                    for item in labeled_batch:
                        await out_file.write(json.dumps(item) + "\n")

            print("✅ Done! All remaining queries processed.\n")


if __name__ == "__main__":
    import argparse
    import asyncio
    from ..query_classifier.labeler_config import labeler_config

    parser = argparse.ArgumentParser(description="Evaluate best retrieval method per query using GT answers")
    parser.add_argument("--input", type=str, required=True, help="Path to the input sampled queries and answers JSONL File - knowledgebase/sampled_queries.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output evaluated samples JSONL File - knowledgebase/evaluated_samples.jsonl")
    args = parser.parse_args()

    
    evaluator = RetrieverEvaluator(labeler_config)
    asyncio.run(evaluator.generate_answers(args.input, args.output))