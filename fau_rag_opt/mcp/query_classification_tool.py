from fau_rag_opt.retrievers.base import retriever_config
from fau_rag_opt.retrievers.hybrid import HybridRetrieval
from fau_rag_opt.query_classifier.llm_response import llm_response
from fau_rag_opt.helpers.utils import load_vector_db
from fau_rag_opt.constants.my_constants import (
    METADATA_PATH,
    VECTORSTORE_PATH,
    HYBRID_ALPHA_MAP,
    TOP_K)
from typing import List, Dict, Any

class QueryClassificationTool:
    def __init__(self):
        self.alpha_map = HYBRID_ALPHA_MAP
        self.top_k = TOP_K
        self.index, self.metadata = load_vector_db(VECTORSTORE_PATH, METADATA_PATH)
        print("  ⚙️ Executing Tool: Query Classification")

    async def _private_classify(self, query: str, llm_params: Dict) -> str:
        prompt = (
            "Classify the following query as 'factual', 'comparative', or 'ambiguous'. "
            f"Respond with only the category name in lowercase.\n\nQuery: \"{query}\""
        )
        category = (await llm_response(**llm_params, prompt=prompt)).strip().lower()
        return category
    
    async def __private_retrieve(self, query: str, alpha: float, filters: Dict) -> List[str]:
        retriever = HybridRetrieval.from_config(alpha, retriever_config, self.metadata, self.index)
        return await retriever.retrieve_metadata_filters(query, self.top_k, metadata_filters=filters)
    
    async def run(self, state: Dict[str, Any], llm_params: Dict) -> Dict[str, Any]:
        """
        The main entrypoint for the tool.
        Receives the pipeline state, performs its action, and returns the updated state.
        """
        # 1. READ from the pipeline state
        query = state.get("search_query", state["original_query"])
        filters = state.get("filters", {})

        # 2. PERFORM the tool's core logic
        category = await self._private_classify(query, llm_params)
        alpha = self.alpha_map.get(category, self.alpha_map['default'])
        print(f"     - Classified as '{category}', applying alpha={alpha}")
        
        retrieved_docs = await self.__private_retrieve(query, alpha, filters)

        # 3. WRITE the result back to the pipeline state
        state["retrieved_docs"] = retrieved_docs
        
        # 4. RETURN the modified state for the next tool
        return state