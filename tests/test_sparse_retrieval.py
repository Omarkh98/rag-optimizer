import pytest
from fau_rag_opt.retrievers.sparse import sparse_retrieval_instance

@pytest.mark.asyncio
async def test_sparse_retrieve():
    # query = "What support does FAU offer students?"
    query = "Tell me about the Data Science program?"

    results = await sparse_retrieval_instance.retrieve(query, top_k = 5)

    print(f"results: {results}")

    assert isinstance(results, list), "Result should be a list."
    assert len(results) > 0, "Result list should not be empty."
    assert all(isinstance(doc, str) for doc in results), "Each retrieved document should be a string."

@pytest.mark.asyncio
async def test_sparse_compute_scores():
    # query = "What support does FAU offer students?"
    query = "Tell me about the Data Science program?"

    scores = sparse_retrieval_instance.compute_sparse_scores(query)

    # print(f"scores: {scores}")

    assert isinstance(scores, list), "Scores should be a list."
    assert len(scores) == len(sparse_retrieval_instance.metadata), "Score count should match metadata size."
    assert all(isinstance(score, float) for score in scores), "All scores should be floats."

