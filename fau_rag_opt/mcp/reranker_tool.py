# ------------------------------------------------------------------------------
# reranker_tool.py - Re-ranks documents for relevance using a Cross-Encoder.
# ------------------------------------------------------------------------------
import json
import subprocess
import sys
import asyncio
from typing import List, Dict, Any
from fau_rag_opt.constants.my_constants import TOP_N

class RerankerTool:
    def __init__(self):
        self.top_n = TOP_N
        self.worker_script_path = __file__.replace("reranker_tool.py", "reranker_worker.py")

    def _print_ranking_comparison(self, original_docs: List[str], reranked_docs: List[str]):
        original_map = {doc: i + 1 for i, doc in enumerate(original_docs)}

        for i, doc in enumerate(original_docs, 1):
            snippet = doc.strip().replace('\n', ' ').replace('\r', '')
            print(f"  {i:2d}. {snippet[:100]}...")

        for i, doc in enumerate(reranked_docs, 1):
            original_rank = original_map.get(doc)
            snippet = doc.strip().replace('\n', ' ').replace('\r', '')
            movement = ""
            if original_rank is not None:
                if original_rank > i: movement = f"↑ (was #{original_rank})"
                elif original_rank < i: movement = f"↓ (was #{original_rank})"
                else: movement = f"  (was #{original_rank})"
            print(f"  {i:2d}. {snippet[:80]}... {movement}")

    async def run(self, state: Dict[str, Any], llm_params: Dict) -> Dict[str, Any]:
        query = state.get("search_query", state["original_query"])
        documents = state.get("retrieved_docs", [])

        if not documents:
            print(" No documents to rerank. Skipping.")
            return state

        print(f" Reranking {len(documents)} documents via isolated worker...")
        
        input_data = json.dumps({"query": query, "documents": documents})
        python_executable = sys.executable
        
        process = await asyncio.create_subprocess_exec(
            python_executable, self.worker_script_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=input_data.encode())

        if process.returncode != 0:
            print(f" Reranker worker failed: {stderr.decode()}")
            return state
        
        try:
            reranked_docs = json.loads(stdout.decode())
            self._print_ranking_comparison(documents, reranked_docs[:self.top_n])
            state["retrieved_docs"] = reranked_docs[:self.top_n]
        except json.JSONDecodeError:
            print(" Failed to parse reranker worker output.")
        
        return state