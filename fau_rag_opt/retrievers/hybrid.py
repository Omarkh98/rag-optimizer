from fau_rag_opt.helpers.exception import CustomException

from fau_rag_opt.retrievers.dense import dense_retrieval_instance
from fau_rag_opt.retrievers.sparse import sparse_retrieval_instance

from fau_rag_opt.constants.my_constants import (HYBRID_ALPHA,
                                   TOP_K)

import sys
import time
import logging

class HybridRetrieval:
    def __init__(self, alpha: float):
        self.alpha = alpha

    async def retrieve(self, query: str, metadata:list, top_k: int = TOP_K):
        try:
            start_time = time.time()

            # Computing individual scores
            dense_scores = dense_retrieval_instance.compute_dense_scores(query, metadata)
            print("Dense scores computed")
            sparse_scores = sparse_retrieval_instance.compute_sparse_scores(query)
            print("Sparse scores computed")
            
            # Combining Scores
            final_scores = [
                self.alpha * d + (1 - self.alpha) * s
                for d, s in zip(dense_scores, sparse_scores)
            ]
            
            # Top Results
            top_indices = sorted(range(len(final_scores)), key = lambda i: final_scores[i], reverse = True)[:top_k]
            
            retrieved_docs = [metadata[i]["text"] for i in top_indices]

            elapsed_time = time.time() - start_time

            logging.info(f"Hybrid retrieval completed in {elapsed_time:.4f} seconds.")
            return retrieved_docs

        except Exception as e:
            logging.error(f"Error during hybrid retrieval: {e}")
            raise CustomException(e, sys)

hybrid_retrieval_instance = HybridRetrieval(alpha = HYBRID_ALPHA)