from typing import List, Dict, Any
from fau_rag_opt.constants.my_constants import TOP_N
import subprocess
import sys
import json

class RerankerTool:
    def __init__(self):
        self.model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        self.top_n = TOP_N
        self.worker_script_path = 'fau_rag_opt/mcp/reranker_worker.py'

        print("  ⚙️ Executing Tool: Reranker")

    def _run_rerank_worker(self, query: str, documents: List[str]) -> List[str]:
        """
        Executes the rerank worker in a separate process to avoid memory conflicts.
        """
        # Prepare the data to be sent to the worker
        input_data = json.dumps({
            "query": query,
            "documents": documents,
            "model_name": self.model_name,
            "top_n": self.top_n
        })

        try:
            # Get the path to the current Python executable
            python_executable = sys.executable
            
            # Run the worker script as a subprocess
            process = subprocess.run(
                [python_executable, self.worker_script_path],
                input=input_data,
                capture_output=True,
                text=True,
                check=True
            )
            # Parse the JSON output from the worker
            return json.loads(process.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
            print(f"     - Rerank worker failed: {e}")
            # Fallback: return the original documents, truncated
            return documents[:self.top_n]
        
    def _print_ranking_comparison(self, original_docs: List[str], reranked_docs: List[str]):
        """
        Prints a comparison of original vs reranked document ordering.
        """
        print("\n" + "="*80)
        print("📊 RERANKING COMPARISON")
        print("="*80)
        
        # Show original documents with their ranks
        print(f"\n🔍 ORIGINAL RETRIEVAL ORDER ({len(original_docs)} documents):")
        print("-" * 60)
        for i, doc in enumerate(original_docs, 1):
            # Truncate long documents for display
            doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
            print(f"  {i:2d}. {doc_preview}")
        
        # Show reranked documents with their new ranks and original positions
        print(f"\n🎯 RERANKED ORDER (Top {len(reranked_docs)} documents):")
        print("-" * 60)
        for new_rank, doc in enumerate(reranked_docs, 1):
            # Find original position of this document
            try:
                original_rank = original_docs.index(doc) + 1
            except ValueError:
                original_rank = "?"
            
            # Truncate long documents for display
            doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
            
            # Show movement with arrows
            if original_rank != "?" and original_rank != new_rank:
                if original_rank > new_rank:
                    movement = f"↑ (was #{original_rank})"
                else:
                    movement = f"↓ (was #{original_rank})"
            else:
                movement = f"(was #{original_rank})"
            
            print(f"  {new_rank:2d}. {doc_preview}")
            print(f"      {movement}")
        
        # Summary statistics
        print(f"\n📈 RERANKING SUMMARY:")
        print(f"     - Retrieved: {len(original_docs)} documents")
        print(f"     - Reranked top: {len(reranked_docs)} documents")
        print(f"     - Model used: {self.model_name}")
        print("="*80 + "\n")

    async def run(self, state: Dict[str, Any], llm_params: Dict) -> Dict[str, Any]:
        """
        Re-ranks the retrieved documents via an external worker and updates the state.
        """
        query = state.get("search_query", state["original_query"])
        documents = state.get("retrieved_docs", [])

        if not documents:
            print("     - No documents to rerank. Skipping.")
            return state

        print(f"     - Reranking {len(documents)} documents via isolated worker...")
        
        # Outsource the heavy lifting to the worker process
        reranked_docs = self._run_rerank_worker(query, documents)


        original_docs = documents.copy()
        self._print_ranking_comparison(original_docs, reranked_docs)

        # Update the state with the new, sorted list
        state["retrieved_docs"] = reranked_docs
        
        return state