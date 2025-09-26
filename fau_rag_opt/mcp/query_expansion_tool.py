# ------------------------------------------------------------------------------
# query_expansion_tool.py - Generates and searches for multiple query variations.
# ------------------------------------------------------------------------------
import asyncio
from typing import Dict, Any, List

from fau_rag_opt.retrievers.base import retriever_config
from fau_rag_opt.retrievers.hybrid import HybridRetrieval
from fau_rag_opt.helpers.utils import load_vector_db
from fau_rag_opt.constants.my_constants import (METADATA_PATH, VECTORSTORE_PATH, TOP_K)
from fau_rag_opt.query_classifier.llm_response import llm_response

class QueryExpansionTool:
    def __init__(self):
        self.top_k = TOP_K
        self.default_alpha = 0.6
        self._index = None
        self._metadata = None

    @property
    def index(self):
        if self._index is None:
            self._index, self._metadata = load_vector_db(VECTORSTORE_PATH, METADATA_PATH)
        return self._index

    @property
    def metadata(self):
        if self._metadata is None:
            self._index, self._metadata = load_vector_db(VECTORSTORE_PATH, METADATA_PATH)
        return self._metadata

    async def _retrieve_for_query(self, query: str, filters: Dict) -> List[str]:
        retriever = HybridRetrieval.from_config(self.default_alpha, retriever_config, self.metadata, self.index)
        return await retriever.retrieve(query, self.metadata, self.top_k, ids=filters)

    async def run(self, state: Dict[str, Any], llm_params: Dict) -> Dict[str, Any]:
        query = state.get("search_query", state["original_query"])
        filters = state.get("filters", {})
        
        prompt = (
            "Generate 3 diverse, related search queries based on the user query. "
            "Return ONLY a JSON list of strings.\n\n"
            f"Original Query: \"{query}\""
        )
        response = await llm_response(**llm_params, prompt=prompt)
        
        parser = llm_params.get('parser')
        expanded_queries = parser(response, response_type=list) or []
        
        all_queries = [query] + expanded_queries
        print(f" Expanding into: {len(all_queries)} queries")

        tasks = [self._retrieve_for_query(q, filters) for q in all_queries]
        results = await asyncio.gather(*tasks)
        
        unique_docs = {doc for doc_list in results for doc in doc_list}
        state["retrieved_docs"] = list(unique_docs)
        return state