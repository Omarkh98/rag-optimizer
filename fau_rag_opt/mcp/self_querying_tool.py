# ------------------------------------------------------------------------------
# self_querying_tool.py - Parses queries to extract search strings and filters.
# ------------------------------------------------------------------------------
from typing import Dict, Any
from fau_rag_opt.query_classifier.llm_response import llm_response

class SelfQueryingTool:
    def __init__(self):
        pass

    async def run(self, state: Dict[str, Any], llm_params: Dict) -> Dict[str, Any]:
        query = state["original_query"]
        prompt = (
            "You are an expert at parsing natural language queries into structured search requests. "
            "Extract a search query and any metadata filters from the user's text. "
            "Your available metadata fields are `year` (integer) and `category` (string).\n"
            "Respond ONLY with a JSON object containing `search_query` and a `filters` dictionary.\n\n"
            "--- Examples ---\n"
            "User Text: 'Tell me about AI courses from 2023' -> {\"search_query\": \"AI courses\", \"filters\": {\"year\": 2023}}\n"
            "User Text: 'What is the difference between AI and Data Science?' -> {\"search_query\": \"difference between AI and Data Science\", \"filters\": {}}\n"
            "----------------\n\n"
            f"User Text: \"{query}\""
        )
        
        params_for_llm = llm_params.copy()
        parser = params_for_llm.pop('parser', None)
        
        response = await llm_response(**params_for_llm, prompt=prompt)
        
        if parser:
            parsed_data = parser(response, response_type=dict)
        else:
            try:
                import json
                parsed_data = json.loads(response.strip())
            except:
                parsed_data = None

        if parsed_data and isinstance(parsed_data.get('filters'), dict):
            new_query = parsed_data.get('search_query', query)
            filters = parsed_data.get('filters', {})
            print(f" Self-querying successful. New query: '{new_query}', Filters: {filters}")
            state["search_query"] = new_query
            state["filters"] = filters
        else:
            print(" Self-querying parsing failed. Using original query.")
        
        return state