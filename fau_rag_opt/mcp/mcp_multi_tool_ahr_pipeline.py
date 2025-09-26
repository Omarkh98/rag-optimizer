# ------------------------------------------------------------------------------
# multi_tool_ahr_pipeline.py - The final, advanced RAG system.
# ------------------------------------------------------------------------------
import re
import json
from typing import List

from fau_rag_opt.query_classifier.llm_response import llm_response
from fau_rag_opt.query_classifier.labeler_config import labeler_config

from fau_rag_opt.mcp.query_classification_tool import QueryClassificationTool
from fau_rag_opt.mcp.self_querying_tool import SelfQueryingTool
from fau_rag_opt.mcp.query_expansion_tool import QueryExpansionTool
from fau_rag_opt.mcp.reranker_tool import RerankerTool

class MultiToolAHRPipeline:
    def __init__(self):
        self.config = labeler_config
        self.llm_params = {
            "api_key": self.config.api_key,
            "model": self.config.model,
            "base_url": self.config.base_url
        }

        self.query_classification_tool = QueryClassificationTool()
        self.self_querying_tool = SelfQueryingTool()
        self.query_expansion_tool = QueryExpansionTool()
        self.reranker_tool = RerankerTool()

        self.tools = {
            "query_classification_retrieval": self.query_classification_tool.run,
            "query_expansion": self.query_expansion_tool.run,
            "self_querying": self.self_querying_tool.run,
            "rerank": self.reranker_tool.run
        }

    def _parse_llm_response(self, response: str, response_type: type = list):
        try:
            if "<|channel|>final<|message|>" in response:
                response = response.split("<|channel|>final<|message|>")[-1]

            match = re.search(r'```(json)?\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                clean_response = match.group(2)
            else:
                clean_response = response

            parsed_json = json.loads(clean_response.strip())
            
            if isinstance(parsed_json, response_type):
                return parsed_json
            
        except (json.JSONDecodeError, TypeError):
            pass
        
        if response_type == str:
            return response.strip()
        
        return None

    async def get_execution_plan(self, query: str) -> List[str]:
        print(f"\n--- AHR Planning for Query: '{query}' ---")
        prompt = (
            "You are a master orchestrator for a RAG pipeline. Your goal is to create an optimal execution plan. "
            "Based on the user's query, return a JSON list of tool names in the correct execution order.\n\n"
            "--- Available Tools ---\n"
            "- `query_classification_retrieval`: The main retrieval tool. Classifies the query and uses the best alpha for hybrid search. Should almost always be used.\n"
            "- `self_querying`: Use ONLY if the query contains explicit metadata filters (like a year '2023' or a category 'library'). Must come before retrieval.\n"
            "- `query_expansion`: Use for short, broad, or ambiguous queries to get more comprehensive results. Must come before retrieval.\n"
            "- `rerank`: Use to improve the quality of the final document list. Must come after retrieval.\n\n"
            "--- Examples ---\n"
            "Query: 'What is the difference between AI and Data Science?' -> [\"query_classification_retrieval\", \"rerank\"]\n"
            "Query: 'student life' -> [\"query_expansion\", \"query_classification_retrieval\", \"rerank\"]\n"
            "Query: 'Tell me about courses from 2023' -> [\"self_querying\", \"query_classification_retrieval\", \"rerank\"]\n"
            "-------------------\n\n"
            f"Query: \"{query}\""
        )
        response = await llm_response(**self.llm_params, prompt=prompt)
        
        plan = self._parse_llm_response(response, response_type=list)

        if plan and all(tool in self.tools for tool in plan):
            print(f" Planner decided on plan: {plan}")
            return plan
        
        return ["query_classification_retrieval"]

    async def run(self, query: str, plan: List[str] = None) -> List[str]:
        if plan is None:
            plan = await self.get_execution_plan(query)
        else:
            print(f" Using manually provided plan: {plan}")

        pipeline_state = {
            "original_query": query, "search_query": query,
            "retrieved_docs": [], "filters": {}
        }

        tool_params = self.llm_params.copy()
        tool_params['parser'] = self._parse_llm_response
        
        for tool_name in plan:
            if tool_name in self.tools:
                print(f"  -> Executing Tool: {tool_name}")
                tool_instance = self.tools[tool_name]
                pipeline_state = await tool_instance(pipeline_state, tool_params)
        
        final_docs = pipeline_state.get("retrieved_docs", [])
        return final_docs