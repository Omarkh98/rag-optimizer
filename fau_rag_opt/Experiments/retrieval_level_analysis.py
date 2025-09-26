# ------------------------------------------------------------------------------
# retrieval_level_analysis.py - Evaluates the retrieval stage of RAG pipelines.
# ------------------------------------------------------------------------------
import pandas as pd
import argparse
import asyncio
from typing import List, Dict, Optional

from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

from ..query_classifier.labeler_config import LabelerConfig

from ..helpers.labelling import LabelSetup

from ..retrievers.base import retriever_config
from ..retrievers.dense import DenseRetrieval
from ..retrievers.sparse import SparseRetrieval
from ..retrievers.hybrid import HybridRetrieval

from ..helpers.utils import (load_vector_db,
                             read_yaml)

from ..constants.my_constants import (METADATA_PATH,
                                      VECTORSTORE_PATH,
                                      TOP_K,
                                      HYBRID_ALPHA,
                                      CONFIG_FILE_PATH)

class RetrievalLevelAnalyzer:
    def __init__(self, config: LabelerConfig):
        self.config = config
        self.top_k = TOP_K
        self.alpha = HYBRID_ALPHA
        self.hit_threshold = 0.3

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
    
    async def retrieve_docs(self, query: str):
        query_embeddings = self.dense._run_encode_worker(query)
        docs_dense = await self.dense.retrieve(query_embeddings, self.index, self.metadata, self.top_k)

        docs_sparse = await self.sparse.retrieve(query, self.top_k)
        
        docs_hybrid = await self.hybrid.retrieve(query, self.metadata, self.top_k)

        return docs_dense, docs_sparse, docs_hybrid
    
    def _calculate_retrieval_metrics(self, retrieved_docs: List[str], ground_truth: str) -> Dict[str, float]:
        if not retrieved_docs or not ground_truth:
            return {"hit_rate": 0.0, "mrr": 0.0, "context_recall": 0.0, "context_relevance": 0.0}

        hit = 0.0
        reciprocal_rank = 0.0
        first_hit_found = False
        for i, doc in enumerate(retrieved_docs):
            score = self.rouge_scorer_instance.score(ground_truth, doc)['rougeL'].recall
            
            if score >= self.hit_threshold:
                hit = 1.0
                if not first_hit_found:
                    reciprocal_rank = 1 / (i + 1)
                    first_hit_found = True
        
        concatenated_context = " ".join(retrieved_docs)        
        context_recall_score = self.rouge_scorer_instance.score(ground_truth, concatenated_context)['rougeL'].recall

        embedding_gt = self.similarity_model_instance.encode(ground_truth, convert_to_tensor=True)
        embedding_context = self.similarity_model_instance.encode(concatenated_context, convert_to_tensor=True)
        context_relevance_score = util.pytorch_cos_sim(embedding_gt, embedding_context).item()

        return {
            "hit_rate": hit,
            "mrr": reciprocal_rank,
            "context_recall": round(context_recall_score, 4),
            "context_relevance": round(context_relevance_score, 4)
        }
    
    async def run_evaluation(self, input_path: str, output_path: str):
        label_setup = LabelSetup()
        all_samples = label_setup.load_samples(input_path)
        all_results = []

        for i, sample in enumerate(all_samples, 1):
            query = sample.get("query")
            ground_truth = sample.get("ground_truth")

            if not query or not ground_truth:
                continue

            print(f"\n [Processing Query {i}/{len(all_samples)}] ---")
            print(f"Query: {query[:80]}...")

            try:
                docs_dense, docs_sparse, docs_hybrid = await self.retrieve_docs(query)

                metrics_dense = self._calculate_retrieval_metrics(docs_dense, ground_truth)
                metrics_sparse = self._calculate_retrieval_metrics(docs_sparse, ground_truth)
                metrics_hybrid = self._calculate_retrieval_metrics(docs_hybrid, ground_truth)

                for method, metrics in [("dense", metrics_dense), ("sparse", metrics_sparse), ("hybrid", metrics_hybrid)]:
                    result = {"method": method, **metrics}
                    all_results.append(result)

            except Exception as e:
                print(f"Error processing query: {query}. Error: {e}")
                continue
        
        self.analyze_and_save(all_results, output_path)

    def analyze_and_save(self, results: List[Dict], output_path: str):
        if not results:
            return

        df = pd.DataFrame(results)

        summary_df = df.groupby('method')[['hit_rate', 'mrr', 'context_recall', 'context_relevance']].mean()
        
        print("\n\n Retrieval-Level Evaluation Summary:")
        print(summary_df.round(4).to_markdown(index=True))

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Retrieval-Level Evaluation Summary:\n")
            f.write(summary_df.round(4).to_markdown(index=True))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a retrieval-level evaluation on a benchmark dataset.")
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

    analyzer = RetrievalLevelAnalyzer(args.input)
    asyncio.run(analyzer.run_evaluation(args.input, args.output))