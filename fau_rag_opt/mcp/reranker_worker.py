# ------------------------------------------------------------------------------
# rerank_worker.py - An isolated process for the memory-intensive reranker.
# ------------------------------------------------------------------------------
import sys
import json

from sentence_transformers.cross_encoder import CrossEncoder

def rerank(data):
    query = data['query']
    documents = data['documents']
    model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    
    model = CrossEncoder(model_name)
    
    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs, show_progress_bar=False)
    
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs]

if __name__ == "__main__":
    try:
        input_data = sys.stdin.read()
        data = json.loads(input_data)
        reranked_documents = rerank(data)
        print(json.dumps(reranked_documents))
    except Exception as e:
        sys.exit(1)