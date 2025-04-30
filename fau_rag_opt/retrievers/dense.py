from sentence_transformers import SentenceTransformer
from fau_rag_opt.helpers.exception import CustomException
from typing import List

from fau_rag_opt.retrievers.base import (RetrieverConfig,
                             retriever_config)

from sklearn.metrics.pairwise import cosine_similarity

from fau_rag_opt.constants.my_constants import TOP_K

import sys
import time
import logging

class DenseRetrieval:
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.retriever = SentenceTransformer(self.config.transformer)

    def compute_dense_scores(self, query: str, metadata) -> List[float]:
        try:
            corpus = [entry["text"] for entry in metadata]
            corpus_embeddings = self.retriever.encode(corpus, show_progress_bar=True)

            query_embedding = self.retriever.encode([query])
            dense_scores = cosine_similarity(query_embedding, corpus_embeddings).flatten().tolist()

            logging.info("Dense Scores Computed Successfully!")
            return dense_scores
        
        except Exception as e:
            logging.error(f"Error Computing Dense Scores: {e}")
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
            print(f"Error generating embedding: {e}")
            raise CustomException(e, sys)
        
# Instance Creation
dense_retrieval_instance = DenseRetrieval(retriever_config)