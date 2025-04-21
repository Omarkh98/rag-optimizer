import pytest
from fau_rag_opt.retrievers.dense import DenseRetrieval
from fau_rag_opt.retrievers.sparse import SparseRetrieval
from fau_rag_opt.retrievers.hybrid import HybridRetrieval
from fau_rag_opt.retrievers.base import retriever_config

from fau_rag_opt.helpers.loader import MetadataLoader

from fau_rag_opt.constants import METADATA_PATH

@pytest.mark.asyncio
async def test_hybrid_retrieve():
    # query = "What support does FAU offer students?"
    query = "Tell me about the Data Science program?"

    loader = MetadataLoader(METADATA_PATH)
    metadata = loader.load()[:10]

    dense_retrieval_instance = DenseRetrieval(retriever_config)  # new instance per test
    sparse_retrieval_instance = SparseRetrieval(retriever_config, metadata)
    hybrid_retrieval_instance = HybridRetrieval(alpha = 0.5)

    dense_scores = dense_retrieval_instance.compute_dense_scores(query, metadata)
    sparse_scores = sparse_retrieval_instance.compute_sparse_scores(query)

    final_scores = [
        hybrid_retrieval_instance.alpha * d + (1 - hybrid_retrieval_instance.alpha) * s
        for d, s in zip(dense_scores, sparse_scores)
    ]

    top_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)[:5]
    results = [metadata[i]["text"] for i in top_indices]

    assert isinstance(results, list), "Result should be a list."
    assert len(results) <= 5, "Should not return more than top_k results."

    for doc in results:
        assert isinstance(doc, str), "Each retrieved document should be a string."

    print("\nHybrid Retrieval Results:")
    for i, doc in enumerate(results):
        print(f"{i + 1}: {doc}")