from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from typing import List

from fau_rag_opt.helpers.exception import CustomException
from fau_rag_opt.helpers.utils import preprocess_knowledgebase
from fau_rag_opt.helpers.loader import MetadataLoader

from fau_rag_opt.retrievers.base import (RetrieverConfig,
                             retriever_config)

from sklearn.metrics.pairwise import cosine_similarity

from fau_rag_opt.constants.my_constants import (METADATA_PATH,
                       TOP_K)

import sys
import time
import logging

class SparseRetrieval:
    def __init__(self, config: RetrieverConfig, metadata: list):
        self.config = config
        self.retriever = SentenceTransformer(self.config.transformer)
        self.vectorizer = TfidfVectorizer()
        self.metadata = metadata

        # Corpus Prep
        self.corpus = preprocess_knowledgebase(metadata)
        self.document_vectors = self.vectorizer.fit_transform(self.corpus)

    def compute_sparse_scores(self, query: str) -> List[float]:
        try:
            query_vector = self.vectorizer.transform([query.lower()])
            sparse_scores = cosine_similarity(query_vector, self.document_vectors).flatten().tolist()

            logging.info("Sparse Scores Computed Successfully!")
            return sparse_scores
        
        except Exception as e:
            logging.error(f"Error Computing Sparse Scores: {e}")
            raise CustomException(e,sys)
        
    async def retrieve(self, query: str, top_k: int = TOP_K):
        try:
            start_time = time.time()

            query_tokens = word_tokenize(query.lower())
            query_processed = ' '.join(query_tokens)

            query_vector = self.vectorizer.transform([query_processed])
            similarity_scores = cosine_similarity(query_vector, self.document_vectors).flatten()

            top_indices = similarity_scores.argsort()[::-1][:top_k]
            retrieved_docs = [self.metadata[i]["text"] for i in top_indices]

            elapsed_time = time.time() - start_time

            logging.info(f"Sparse retrieval completed in {elapsed_time:.4f} seconds.")
            return retrieved_docs

        except Exception as e:
            logging.error(f"Error during sparse retrieval: {e}")
            raise CustomException(e, sys)

loader = MetadataLoader(METADATA_PATH)
metadata_list = loader.load()

sparse_retrieval_instance = SparseRetrieval(retriever_config, metadata = metadata_list)