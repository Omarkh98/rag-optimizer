from fau_rag_opt.helpers.exception import CustomException

from ..retrievers.dense import DenseRetrieval
from ..retrievers.sparse import SparseRetrieval
from ..retrievers.base import RetrieverConfig

from fau_rag_opt.helpers.utils import load_vector_db

from fau_rag_opt.constants.my_constants import (
    METADATA_PATH,
    VECTORSTORE_PATH,
    TOP_K)

from fau_rag_opt.constants.my_constants import TOP_K
from typing import List, Dict, Any

import sys
import time
import logging

class HybridRetrieval:
    def __init__(self, alpha: float, dense_retriever: DenseRetrieval, sparse_retriever: SparseRetrieval, index):
        self.alpha = alpha
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.index = index
        _, self.metadata = load_vector_db(VECTORSTORE_PATH, METADATA_PATH)

    def _apply_filters(self, indices: List[int], metadata_filters: Dict[str, Any]) -> List[int]:
        """Filters a list of document indices based on metadata criteria."""
        if not metadata_filters:
            return indices

        filtered_indices = []
        for i in indices:
            doc_metadata = self.metadata[i].get('metadata', {})
            match = True
            for key, value in metadata_filters.items():
                if doc_metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered_indices.append(i)
        
        print(f"     - Filtering: {len(indices)} docs -> {len(filtered_indices)} docs after applying {metadata_filters}")
        return filtered_indices

    async def retrieve_metadata_filters(self, query: str, top_k: int = TOP_K, metadata_filters: Dict[str, Any] = None) -> List[str]:
        """
        Performs hybrid retrieval and applies metadata filters if provided.
        """
        # 1. Get query embedding
        query_embedding = self.dense._run_encode_worker(query)

        # 2. Get dense scores and indices
        dense_scores, dense_indices = await self.dense.get_dense_scores(query_embedding, self.index, top_k * 5)

        # 3. Get sparse scores and indices (for the same docs)
        sparse_scores_all = await self.sparse.compute_sparse_scores(query)

        # 4. Reciprocal Rank Fusion (RRF) to combine scores
        # We need a mapping from document index to its rank in each list
        dense_rank = {doc_id: i + 1 for i, doc_id in enumerate(dense_indices)}

        # For sparse, we need to rank all documents to find the ranks of the dense candidates
        sparse_ranked_indices = sorted(range(len(sparse_scores_all)), key=lambda i: sparse_scores_all[i], reverse=True)
        sparse_rank = {doc_id: i + 1 for i, doc_id in enumerate(sparse_ranked_indices)}

        # Combine scores for documents found by the dense retriever
        fused_scores = {}
        for doc_id in dense_indices:
            # RRF score = 1/(k + rank_dense) + 1/(k + rank_sparse)
            # We use a small constant k=60 to reduce the impact of high ranks
            rrf_score = 0
            if doc_id in dense_rank:
                rrf_score += self.alpha * (1 / (60 + dense_rank[doc_id]))
            if doc_id in sparse_rank:
                rrf_score += (1 - self.alpha) * (1 / (60 + sparse_rank[doc_id]))
            fused_scores[doc_id] = rrf_score

        # 5. Sort by fused score
        sorted_indices = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

        # 6. Apply metadata filters (THIS IS THE NEW STEP)
        filtered_indices = self._apply_filters(sorted_indices, metadata_filters)

        # 7. Get the final top-k documents from the filtered list
        final_indices = filtered_indices[:top_k]

        return [self.metadata[i]["text"] for i in final_indices]

    async def retrieve(self, query: str, metadata:list, top_k: int = TOP_K):
        try:
            start_time = time.time()

            # Computing individual scores
            query_embeddings = self.dense._run_encode_worker(query)
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