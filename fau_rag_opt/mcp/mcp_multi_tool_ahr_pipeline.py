# ------------------------------------------------------------------------------
# multi_tool_ahr_pipeline.py - The final, advanced RAG system.
# ------------------------------------------------------------------------------
"""
The final capstone AHR system, which functions as a
multi-tool orchestrator inspired by MCP principle.
"""
import re
import json
import asyncio
from typing import List

from fau_rag_opt.query_classifier.llm_response import llm_response
from fau_rag_opt.query_classifier.labeler_config import labeler_config

from fau_rag_opt.mcp import (
    SelfQueryingTool,
    QueryClassificationTool,
    QueryExpansionTool,
    RerankerTool)

class MultiToolAHRPipeline:
    def __init__(self):
        """Initializes the multi-tool AHR system and all its components."""

        self.config = labeler_config
        self.llm_params = {
            "api_key": self.config.api_key,
            "model": self.config.model,
            "base_url": self.config.base_url
        }

        self.self_querying_tool = SelfQueryingTool()
        self.query_classification_tool = QueryClassificationTool()
        self.query_expansion_tool = QueryExpansionTool()
        self.reranker_tool = RerankerTool()

        # --- Store tool methods in a dictionary for easy access ---
        self.tools = {
            "query_classification_retrieval": self.query_classification_tool.run,
            "query_expansion": self.query_expansion_tool.run,
            "self_querying": self.self_querying_tool.run,
            "rerank": self.reranker_tool.run
        }

        print("✅ Multi-Tool AHR Pipeline Initialized.")

    # --- ORCHESTRATOR (MCP Implementation) ---
    async def get_execution_plan(self, query: str) -> List[str]:
        """The MCP-inspired planning step. The LLM decides which tools to use."""
        print(f"\n--- AHR Planning for Query: '{query}' ---")

        prompt = (
            "You are a master RAG pipeline orchestrator. Based on the user's query, "
            "create an optimal execution plan from the available tools. "
            "Return ONLY a JSON list of tool names in the correct execution order.\n\n"
            "--- Available Tools ---\n"
            f"- `self_querying`: Use for queries with explicit filters (e.g., dates, categories).\n"
            f"- `query_classification_retrieval`: The main retrieval step. Should almost always be included.\n"
            f"- `query_expansion`: Use for broad or ambiguous queries to improve recall.\n"
            f"- `rerank`: Use to refine and prioritize the best documents from a larger initial set.\n\n"
            "--- Examples ---\n"
            "Query: 'What are the main differences between AI and Data Science?' -> [\"query_classification_retrieval\", \"rerank\"]\n"
            "Query: 'Tell me about research on compilers from 2023' -> [\"self_querying\", \"query_classification_retrieval\", \"rerank\"]\n"
            "Query: 'What is artificial intelligence?' -> [\"query_expansion\", \"rerank\"]\n\n"
            f"--- User Query ---\n"
            f"\"{query}\""
        )

        response = await llm_response(**self.llm_params, prompt=prompt)
        try:
            # 1. Strip whitespace
            clean_response = response.strip()
            
            # 2. Find the JSON list within the response using a regular expression
            # This will find a string that starts with '[' and ends with ']'
            match = re.search(r'\[.*\]', clean_response, re.DOTALL)
            if match:
                json_string = match.group(0)
                plan = json.loads(json_string)
                if isinstance(plan, list) and all(isinstance(i, str) for i in plan):
                    print(f"  -> Planner decided on plan: {plan}")
                    return plan
        except (json.JSONDecodeError, TypeError):
            pass
        
        print("  - Could not generate a valid plan, using default.")
        return ["query_classification_retrieval"]
    
    async def run(self, query: str) -> List[str]:
        """The main entrypoint. It gets a plan and executes it."""
        plan = await self.get_execution_plan(query)
        
        # Initialize state that can be modified by tools
        pipeline_state = {
            "original_query": query,
            "search_query": query,
            "retrieved_docs": [],
            "filters": {}
        }

        for tool_name in plan:
            print(f"  -> Executing Tool: {tool_name}")
            tool_function = self.tools[tool_name]
            # Each tool receives the current state and can modify it
            pipeline_state = await tool_function(pipeline_state, self.llm_params)

        final_docs = pipeline_state.get("retrieved_docs", [])
        print(f"--- Pipeline Finished. Final context has {len(final_docs)} documents. ---")
        return final_docs
    
if __name__ == '__main__':
    async def run_examples():
        pipeline = MultiToolAHRPipeline()

        query1 = "What are the main differences between the Bachelor's in Computer Science and Business Information Systems?"
        await pipeline.run(query1)
        
        print("\n" + "="*80 + "\n")
        
        query2 = "Tell me about research on compilers from 2023"
        await pipeline.run(query2)

    asyncio.run(run_examples())