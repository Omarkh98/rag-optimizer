# ------------------------------------------------------------------------------
# benchmark_analysis.py - Analyzes the output of the retrieval benchmark.
# ------------------------------------------------------------------------------
"""
This script loads the detailed results from `retrieval_benchmark.jsonl`,
performs a quantitative analysis, and prints summary tables that can be
used directly in a research paper or thesis.

It calculates:
1. Overall average scores for each retrieval method.
2. Average scores for each retrieval method, broken down by query category.
3. A "win/loss" record for each method based on the highest score per query.
"""

import pandas as pd
import json
import argparse

class BenchmarkAnalyzer:
    def __init__(self, results_path: str):
        self.df = self._load_results_to_dataframe(results_path)
        print(f"✅ Successfully loaded {len(self.df)} results into DataFrame.")

    def _load_results_to_dataframe(self, path: str) -> pd.DataFrame:
        """Loads a JSONL file into a pandas DataFrame for easy analysis."""
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"⚠️ Skipping malformed JSON line in {path}")
        
        # Normalize the nested JSON structure into a flat table
        flat_records = []
        for r in records:
            if r.get("status") == "error":
                continue
            
            base_info = {
                "id": r.get("id"),
                "query_category": r.get("query_category", "unknown")
            }
            
            for method in ["dense", "sparse", "hybrid"]:
                method_key = f"{method}_result"
                if method_key in r:
                    record = base_info.copy()
                    record["method"] = method
                    record.update(r[method_key]["scores"])
                    flat_records.append(record)
                    
        return pd.DataFrame(flat_records)
    
    def analyze_overall_performance(self):
        """Calculates and prints the average scores for each method across all queries."""
        print("\n--- 1. Overall Performance (Average Scores) ---")
        
        # Group by retrieval method and calculate the mean for each score column
        overall_scores = self.df.groupby('method')[['bert_score_f1', 'rouge_l_f1', 'cosine_similarity']].mean()
        
        print(overall_scores.round(4).to_markdown(index=True))

    def analyze_performance_by_category(self):
        """
        Calculates and prints average scores for each method, grouped by query category.
        """
        print("\n--- 2. Performance by Query Category (Average Scores) ---")
        
        if 'query_category' not in self.df.columns:
            print("⚠️ 'query_category' column not found. Skipping this analysis.")
            return
            
        # Group by both category and method, then calculate the mean
        category_scores = self.df.groupby(['query_category', 'method'])[['bert_score_f1', 'rouge_l_f1', 'cosine_similarity']].mean()
        
        # Format for better readability
        print(category_scores.round(4).to_markdown(index=True))

    def analyze_win_loss_record(self, primary_metric: str = 'bert_score_f1'):
        """
        Determines which method "won" for each query based on the highest score
        for the specified primary metric.
        """
        print(f"\n--- 3. 'Best Method' Analysis (based on highest {primary_metric}) ---")

        # Pivot the table to have methods as columns
        metric_df = self.df.pivot(index='id', columns='method', values=primary_metric)
        
        # Find the column name (method) with the maximum value for each row (query)
        metric_df['best_method'] = metric_df.idxmax(axis=1)
        
        # Count the occurrences of each "best_method"
        win_counts = metric_df['best_method'].value_counts().reset_index()
        win_counts.columns = ['method', 'win_count']
        
        # Calculate percentage
        win_counts['win_percentage'] = (win_counts['win_count'] / win_counts['win_count'].sum() * 100).round(2)

        print(win_counts.to_markdown(index=False))

    def run_full_analysis(self):
        if self.df.empty:
            print("❌ DataFrame is empty. Cannot run analysis.")
            return
            
        self.analyze_overall_performance()
        self.analyze_performance_by_category()
        self.analyze_win_loss_record(primary_metric='bert_score_f1')
        self.analyze_win_loss_record(primary_metric='cosine_similarity')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the results of the RAG benchmark."
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the benchmark results file (e.g., retrieval_benchmark.jsonl)."
    )
    args = parser.parse_args()

    analyzer = BenchmarkAnalyzer(args.input)
    analyzer.run_full_analysis()