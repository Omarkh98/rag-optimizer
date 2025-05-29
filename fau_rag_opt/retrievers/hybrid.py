from fau_rag_opt.helpers.exception import CustomException

from ..retrievers.dense import DenseRetrieval
from ..retrievers.sparse import SparseRetrieval
from ..retrievers.base import RetrieverConfig

from fau_rag_opt.constants.my_constants import (HYBRID_ALPHA,
                                                TOP_K)

import sys
import time
import logging

class HybridRetrieval:
    def __init__(self, alpha: float, dense_retriever: DenseRetrieval, sparse_retriever: SparseRetrieval, index):
        self.alpha = alpha
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.index = index

    async def retrieve(self, query: str, metadata:list, top_k: int = TOP_K):
        try:
            start_time = time.time()

            # Computing individual scores
            query_embeddings = self.dense.retriever.encode([query])
            dense_scores_array, dense_indices_array = await self.dense.get_dense_scores(query_embeddings, self.index, top_k)
            dense_scores = {int(i): float(s) for i, s in zip(dense_indices_array, dense_scores_array)}

            sparse_scores = await self.sparse.compute_sparse_scores(query)
            
            # Combining Scores
            final_scores = []
            for i in range(len(metadata)):
                dense_score = dense_scores.get(i, 0.0)
                sparse_score = sparse_scores[i] if i < len(sparse_scores) else 0.0
                hybrid_score = self.alpha * dense_score + (1 - self.alpha) * sparse_score
                final_scores.append((i, hybrid_score))
            
            # Top Results
            top_indices = sorted(final_scores, key=lambda x: x[1], reverse=True)[:top_k]
            
            retrieved_docs = [metadata[i]["text"] for i, _ in top_indices]

            elapsed_time = time.time() - start_time

            logging.info(f"Hybrid retrieval completed in {elapsed_time:.4f} seconds.")
            return retrieved_docs

        except Exception as e:
            logging.error(f"Error during hybrid retrieval: {e}")
            raise CustomException(e, sys)
        
    @classmethod
    def from_config(cls, alpha, config: RetrieverConfig, metadata: list, index):
        dense = DenseRetrieval(config)
        sparse = SparseRetrieval(config, metadata)
        return cls(alpha, dense, sparse, index)