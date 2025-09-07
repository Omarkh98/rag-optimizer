import json
import asyncio
from typing import List, Dict, Any
from fau_rag_opt.query_classifier.llm_response import llm_response
from fau_rag_opt.helpers.utils import load_vector_db
from fau_rag_opt.retrievers.base import retriever_config
from fau_rag_opt.retrievers.hybrid import HybridRetrieval
from fau_rag_opt.constants.my_constants import (
    METADATA_PATH,
    VECTORSTORE_PATH,
    TOP_K)

class QueryExpansionTool:
    def __init__(self):
        self.top_k = TOP_K
        self.default_alpha = 0.6
        self.index, self.metadata = load_vector_db(VECTORSTORE_PATH, METADATA_PATH)

        print("  ⚙️ Executing Tool: Query Expansion")

    async def _get_expanded_queries(self, query: str, llm_params: Dict) -> List[str]:
        """Generates diverse, related search queries using an LLM."""
        prompt = (
            "You are a query expansion expert. Generate 3 diverse, related search queries based on the user's query. "
            "These queries should explore different facets or phrasings of the original topic. "
            "Return ONLY a JSON list of strings.\n\n"
            f"Original Query: \"{query}\""
        )
        response = await llm_response(**llm_params, prompt=prompt)
        try:
            expanded = json.loads(response)
            if isinstance(expanded, list):
                return expanded
        except json.JSONDecodeError:
            pass
        return []
    
    async def _retrieve_for_query(self, query: str, filters: Dict) -> List[str]:
        """Performs a single hybrid retrieval for one query."""
        retriever = HybridRetrieval.from_config(self.default_alpha, retriever_config, self.metadata, self.index)
        return await retriever.retrieve_metadata_filters(query, self.top_k, metadata_filters=filters)

    async def run(self, state: Dict[str, Any], llm_params: Dict) -> Dict[str, Any]:
        """
        Expands the search query, retrieves documents for all versions in parallel,
        and updates the state with a combined, unique list of documents.
        """
        # 1. READ from the pipeline state
        query = state.get("search_query", state["original_query"])
        filters = state.get("filters", {})

        # 2. PERFORM the tool's core logic
        # Step A: Expand the query
        expanded_queries = await self._get_expanded_queries(query, llm_params)
        all_queries = [query] + expanded_queries
        print(f"     - Expanded into: {all_queries}")

        # Step B: Retrieve for all queries in parallel
        retrieval_tasks = [self._retrieve_for_query(q, filters) for q in all_queries]
        results_list = await asyncio.gather(*retrieval_tasks)
        
        # Step C: Combine and deduplicate the results
        unique_docs = {}
        for doc_list in results_list:
            for doc in doc_list:
                if doc not in unique_docs:
                    unique_docs[doc] = True
        
        combined_docs = list(unique_docs.keys())

        # 3. WRITE the result back to the pipeline state
        state["retrieved_docs"] = combined_docs
        
        # 4. RETURN the modified state
        return state