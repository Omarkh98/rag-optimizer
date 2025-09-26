# ------------------------------------------------------------------------------
# final_benchmark_demonstration.py - A comprehensive comparison of all retrieval pipelines.
# ------------------------------------------------------------------------------
import asyncio
from typing import List

from fau_rag_opt.retrievers.dense import DenseRetrieval
from fau_rag_opt.retrievers.sparse import SparseRetrieval
from fau_rag_opt.retrievers.hybrid import HybridRetrieval
from fau_rag_opt.retrievers.base import retriever_config

from fau_rag_opt.mcp.mcp_multi_tool_ahr_pipeline import MultiToolAHRPipeline

from ..helpers.utils import load_vector_db

from ..constants.my_constants import (METADATA_PATH,
                                      VECTORSTORE_PATH,
                                      TOP_K,
                                      HYBRID_ALPHA)

index, metadata = load_vector_db(VECTORSTORE_PATH, METADATA_PATH)

def print_document_summary(documents: List[str], pipeline_name: str, notes: str = ""):
    print(f"RESULTS for: {pipeline_name}")
    if notes:
        print(f"   Note: {notes}")
    
    if not documents:
        print("   -> No documents found.")
    else:
        for i, doc in enumerate(documents[:5], 1):
            snippet = doc.strip().replace('\n', ' ').replace('\r', '')
            print(f"   {i}. {snippet[:120]}...")
    print("\n")

async def run_dense(query: str):
    retriever = DenseRetrieval(retriever_config)
    query_embeddings = retriever._run_encode_worker(query)
    documents = await retriever.retrieve(query_embeddings, index, metadata, TOP_K)
    print_document_summary(documents, "1. DENSE RETRIEVAL")

async def run_sparse(query: str):
    retriever = SparseRetrieval(retriever_config, metadata)
    documents = await retriever.retrieve(query, TOP_K)
    print_document_summary(documents, "2. SPARSE RETRIEVAL")

async def run_static_hybrid(query: str, alpha: float = 0.5):
    retriever = HybridRetrieval.from_config(HYBRID_ALPHA, retriever_config, metadata, index)
    documents = await retriever.retrieve(query, metadata, TOP_K)
    print_document_summary(documents, "3. STATIC HYBRID RETRIEVAL", f"Fixed alpha = {alpha}")

async def run_heuristic_ahr(pipeline: MultiToolAHRPipeline, query: str):
    plan = ["query_classification_retrieval"]
    documents = await pipeline.run(query, plan=plan)
    print_document_summary(documents, "4. HEURISTIC AHR", "Uses query classification only")

async def run_mcp_ahr(pipeline: MultiToolAHRPipeline, query: str):
    documents = await pipeline.run(query)
    print_document_summary(documents, "5. MCP MULTI-TOOL AHR", "Uses dynamic MCP planner")


async def main():
    challenge_query = "ML applications 2022"
    
    print(f"COMPREHENSIVE PIPELINE BENCHMARK for Query: '{challenge_query}'")
    
    await run_dense(challenge_query)
    await run_sparse(challenge_query)
    await run_static_hybrid(challenge_query)

    advanced_pipeline = MultiToolAHRPipeline()
    
    await run_heuristic_ahr(advanced_pipeline, challenge_query)
    await run_mcp_ahr(advanced_pipeline, challenge_query)

if __name__ == "__main__":
    asyncio.run(main())