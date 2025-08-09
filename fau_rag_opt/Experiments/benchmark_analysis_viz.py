# ------------------------------------------------------------------------------
# benchmark_analysis.py - Analyzes the output of the retrieval benchmark.
# ------------------------------------------------------------------------------
"""
Loads the detailed results from `retrieval_benchmark.jsonl`,
performs a quantitative analysis, prints summary tables, and generates visualizations.
"""

import pandas as pd
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import PercentFormatter
import numpy as np

# Set global plot style
sns.set(style="whitegrid", palette="pastel", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 300

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
        return overall_scores

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
        return category_scores.reset_index()

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
        return win_counts

    def _save_plot(self, plot: plt.Figure, filename: str, output_dir: str):
        """Helper function to save plots"""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        plot.savefig(path, bbox_inches='tight')
        plt.close(plot)
        print(f"📊 Saved plot: {path}")

    def plot_overall_performance(self, overall_scores: pd.DataFrame, output_dir: str):
        """Generate bar plot for overall performance metrics"""
        metrics = ['bert_score_f1', 'rouge_l_f1', 'cosine_similarity']
        metric_names = ['BERT F1', 'ROUGE-L F1', 'Cosine Similarity']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]
            overall_scores.reset_index().plot.bar(
                x='method', 
                y=metric, 
                ax=ax,
                legend=False,
                color=sns.color_palette()
            )
            ax.set_title(f'Average {name}')
            ax.set_xlabel('Retrieval Method')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=0)
            
            # Add data labels
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height():.3f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points'
                )
        
        plt.suptitle('Overall Retrieval Performance Comparison', fontsize=16)
        plt.tight_layout()
        self._save_plot(plt.gcf(), "overall_performance.png", output_dir)

    def plot_category_performance(self, category_scores: pd.DataFrame, output_dir: str):
        """Generate grouped bar plots for each metric by query category"""
        metrics = ['bert_score_f1', 'rouge_l_f1', 'cosine_similarity']
        metric_names = ['BERT F1', 'ROUGE-L F1', 'Cosine Similarity']
        
        # Exclude 'no_answer' category for better visualization
        category_scores = category_scores[category_scores['query_category'] != 'no_answer']
        
        for metric, name in zip(metrics, metric_names):
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(
                data=category_scores,
                x='query_category',
                y=metric,
                hue='method',
                hue_order=['dense', 'sparse', 'hybrid'],
                palette='Set2'
            )
            
            plt.title(f'{name} by Query Category', fontsize=16)
            plt.xlabel('Query Category', fontsize=12)
            plt.ylabel(name, fontsize=12)
            plt.ylim(0, category_scores[metric].max() * 1.15)
            plt.legend(title='Method', loc='upper right')
            
            # Add data labels
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height():.3f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points'
                )
            
            plt.tight_layout()
            self._save_plot(plt.gcf(), f"category_{metric}.png", output_dir)

    def plot_win_loss_comparison(self, bert_wins: pd.DataFrame, cosine_wins: pd.DataFrame, output_dir: str):
        """Generate side-by-side bar plots for win comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for i, (df, title) in enumerate(zip(
            [bert_wins, cosine_wins],
            ['BERT F1 Wins', 'Cosine Similarity Wins']
        )):
            ax = axes[i]
            sns.barplot(
                data=df,
                x='method',
                y='win_percentage',
                ax=ax,
                palette='viridis',
                order=['dense', 'sparse', 'hybrid']
            )
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Method')
            ax.set_ylabel('Win Percentage (%)')
            ax.set_ylim(0, 50)
            
            # Add data labels
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height():.1f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points'
                )
        
        plt.suptitle('Retrieval Method Win Comparison', fontsize=16)
        plt.tight_layout()
        self._save_plot(plt.gcf(), "win_comparison.png", output_dir)

    def run_full_analysis(self, output_dir: str = None):
        if self.df.empty:
            print("❌ DataFrame is empty. Cannot run analysis.")
            return
            
        # Run analyses
        overall_scores = self.analyze_overall_performance()
        category_scores = self.analyze_performance_by_category()
        bert_wins = self.analyze_win_loss_record(primary_metric='bert_score_f1')
        cosine_wins = self.analyze_win_loss_record(primary_metric='cosine_similarity')
        
        # Generate visualizations if output directory is provided
        if output_dir:
            print(f"\n📈 Generating visualizations in directory: {output_dir}")
            self.plot_overall_performance(overall_scores, output_dir)
            self.plot_category_performance(category_scores, output_dir)
            self.plot_win_loss_comparison(bert_wins, cosine_wins, output_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze the results of the RAG benchmark."
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the benchmark results file (e.g., retrieval_benchmark.jsonl)."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="benchmark_plots",
        help="Directory to save visualization plots (default: benchmark_plots)"
    )
    args = parser.parse_args()

    analyzer = BenchmarkAnalyzer(args.input)
    analyzer.run_full_analysis(output_dir=args.output_dir)

if __name__ == "__main__":
    main()