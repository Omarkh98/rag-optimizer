import json
import re
from typing import Dict, Any
from fau_rag_opt.query_classifier.llm_response import llm_response

class SelfQueryingTool:
    def __init__(self):
        print("  ⚙️ Executing Tool: Self Querying")

    async def run(self, state: Dict[str, Any], llm_params: Dict) -> Dict[str, Any]:
        """
        Receives the current pipeline state, parses the query for filters,
        and returns the updated state.
        """
        query = state["original_query"]

        prompt = (
            "You are an expert at parsing natural language queries into structured search requests. "
            "Extract a search query and any metadata filters from the user's text. "
            "Your available metadata fields are `year` (integer) and `category` (string).\n"
            "Respond ONLY with a JSON object containing `search_query` (string) and a `filters` (dict).\n\n"
            "--- Examples ---\n"
            "User Text: 'Tell me about AI courses from 2023' -> {\"search_query\": \"AI courses\", \"filters\": {\"year\": 2023}}\n"
            "User Text: 'What is the difference between AI and Data Science?' -> {\"search_query\": \"difference between AI and Data Science\", \"filters\": {}}\n"
            "----------------\n\n"
            f"User Text: \"{query}\""
        )
        response = await llm_response(**llm_params, prompt=prompt)

        new_search_query = query
        filters = {}

        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                json_string = match.group(0)
                parsed = json.loads(json_string)
                new_search_query = parsed.get("search_query", query)
                filters = parsed.get("filters", {})
                print(f"     - Self-querying successful. New query: '{new_search_query}', Filters: {filters}")
            else:
                 print("     - Self-querying parsing failed: No JSON object found in response. Using original query.")
        except json.JSONDecodeError:
            print("     - Self-querying parsing failed: Invalid JSON. Using original query.")
            
        state["search_query"] = new_search_query
        state["filters"] = filters

        return state