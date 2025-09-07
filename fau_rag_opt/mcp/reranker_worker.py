# ------------------------------------------------------------------------------
# rerank_worker.py - A dedicated worker for the reranking process.
# ------------------------------------------------------------------------------
"""
Designed to be called as a separate process to avoid memory
conflicts between FAISS and PyTorch-based sentence-transformers models.
"""

import sys
import json
from sentence_transformers.cross_encoder import CrossEncoder

def rerank(data: dict):
    """Loads the model, performs reranking, and returns the sorted documents."""
    query = data.get("query")
    documents = data.get("documents")
    model_name = data.get("model_name", 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    top_n = data.get("top_n", 3)

    if not query or not documents:
        return []

    # 1. Load the heavy model in this isolated process
    model = CrossEncoder(model_name)
    
    # 2. Prepare pairs and predict scores
    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs, show_progress_bar=False)
    
    # 3. Sort by score and return the top N
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    
    return [doc for score, doc in scored_docs[:top_n]]

if __name__ == "__main__":
    # Read the JSON data from standard input
    input_data = json.load(sys.stdin)
    
    # Perform the reranking
    reranked_documents = rerank(input_data)
    
    # Print the final result as a JSON list to standard output
    print(json.dumps(reranked_documents))
