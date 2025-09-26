# ------------------------------------------------------------------------------
# query_classification_tool.py - Tool for classifying queries.
# ------------------------------------------------------------------------------
from typing import Dict, Any

from fau_rag_opt.retrievers.base import retriever_config
from fau_rag_opt.retrievers.hybrid import HybridRetrieval
from fau_rag_opt.helpers.utils import load_vector_db
from fau_rag_opt.constants.my_constants import (METADATA_PATH, VECTORSTORE_PATH, HYBRID_ALPHA_MAP, TOP_K)
from fau_rag_opt.query_classifier.llm_response import llm_response

class QueryClassificationTool:
    def __init__(self):
        self.alpha_map = HYBRID_ALPHA_MAP
        self.top_k = TOP_K
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

    async def run(self, state: Dict[str, Any], llm_params: Dict) -> Dict[str, Any]:
        query = state.get("search_query", state["original_query"])
        filters = state.get("filters", {})
        
        prompt = (
            "Classify the following query as 'factual', 'comparative', or 'ambiguous'. "
            f"Respond with only the category name in lowercase.\n\nQuery: \"{query}\""
        )
        
        params_for_llm = llm_params.copy()
        parser = params_for_llm.pop('parser', None)
        
        response = await llm_response(**params_for_llm, prompt=prompt)
        
        category = parser(response, response_type=str) if parser else response.strip().lower()
        
        if category not in self.alpha_map:
            category = 'default'

        alpha = self.alpha_map[category]
        print(f" Classified as '{category}', applying alpha={alpha}")
        
        retriever = HybridRetrieval.from_config(alpha, retriever_config, self.metadata, self.index)
        
        if filters:
            print(f" Retrieving with filters: {filters}")
            retrieved_docs = await retriever.retrieve_metadata_filters(query, self.top_k, metadata_filters=filters)
        else:
            print(" Retrieving without filters.")
            retrieved_docs = await retriever.retrieve(query, self.metadata, self.top_k)
        
        state["retrieved_docs"] = retrieved_docs
        return state