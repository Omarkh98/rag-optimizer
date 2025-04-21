import pytest
from fau_rag_opt.retrievers.dense import dense_retrieval_instance
from fau_rag_opt.helpers.utils import load_vector_db
from fau_rag_opt.helpers.loader import MetadataLoader

from fau_rag_opt.constants import (METADATA_PATH, 
                                   VECTORSTORE_PATH)

@pytest.mark.asyncio
async def test_dense_retrieve():
    # query = "What support does FAU offer students?"
    query = "Tell me about the Data Science program?"

    index, metadata = load_vector_db(VECTORSTORE_PATH, METADATA_PATH)

    query_embeddings = dense_retrieval_instance.retriever.encode([query])

    results = await dense_retrieval_instance.retrieve(query_embeddings, index, metadata, top_k = 5)

    print(f"results: {results}")

    assert isinstance(results, list)
    assert len(results) <= 5
    assert all(isinstance(doc, str) for doc in results)

@pytest.mark.asyncio
async def test_dense_compute_scores():
    # query = "What support does FAU offer students?"
    query = "Tell me about the Data Science program?"

    loader = MetadataLoader(METADATA_PATH)
    metadata = loader.load()[:10]

    scores = dense_retrieval_instance.compute_dense_scores(query, metadata)
    print(f"scores: {scores}")

    assert isinstance(scores, list)
    assert len(scores) == len(metadata)
    assert all(isinstance(score, float) for score in scores)