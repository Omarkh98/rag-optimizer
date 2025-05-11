from sentence_transformers import SentenceTransformer
from fau_rag_opt.helpers.exception import CustomException

from fau_rag_opt.retrievers.base import (RetrieverConfig,
                                         retriever_config)

from fau_rag_opt.constants.my_constants import TOP_K

import sys
import time
import logging

class DenseRetrieval:
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.retriever = SentenceTransformer(self.config.transformer)

    def encode_query(self, query: str):
        return self.retriever.encode([query], convert_to_numpy=True)
    
    async def get_dense_scores(self, query_embedding, index, top_k = TOP_K):
        try:
            scores, indices = index.search(query_embedding, top_k)
            return scores[0], indices[0]
        except Exception as e:
            raise CustomException(e, sys)
        
    async def retrieve(self, query_embedding, index, metadata, top_k: int = TOP_K):
        try:
            start_time = time.time()
            _, indices = index.search(query_embedding, top_k)
            retrieved_docs = [metadata[i]["text"] for i in indices[0]]
            elapsed_time = time.time() - start_time
            logging.info(f"Dense retrieval completed in {elapsed_time:.4f} seconds.")
            return retrieved_docs
        except Exception as e:
            raise CustomException(e, sys)