from sentence_transformers import SentenceTransformer
from fau_rag_opt.helpers.exception import CustomException

from fau_rag_opt.retrievers.base import RetrieverConfig

from fau_rag_opt.constants.my_constants import TOP_K

from typing import Optional

import sys
import time
import logging
import subprocess
import json
import numpy as np

class DenseRetrieval:
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self._model: Optional[SentenceTransformer] = None
        self.worker_script_path = "fau_rag_opt/helpers/encode_worker.py"

    def _run_encode_worker(self, query: str) -> np.ndarray:
        """
        Calls the external worker script to encode a query in an isolated process.
        """
        model_name = self.config.transformer
        python_executable = sys.executable

        command = [
            python_executable,
            self.worker_script_path,
            model_name,
            query
        ]

        try:
            # Execute the command and capture the output
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8'
            )
            
            result = json.loads(process.stdout)
            if "embedding" in result:
                return np.array([result["embedding"]], dtype='float32')
            else:
                raise CustomException(f"Worker failed: {result.get('error', 'Unknown error')}", sys)

        except subprocess.CalledProcessError as e:
            error_output = e.stderr
            raise CustomException(f"Encoding worker process failed with error: {error_output}", sys)
        except Exception as e:
            raise CustomException(e, sys)
    
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