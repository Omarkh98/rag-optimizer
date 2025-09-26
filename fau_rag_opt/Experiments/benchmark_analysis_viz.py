# ------------------------------------------------------------------------------
# benchmark_analysis.py - Analyzes the output of the retrieval benchmark.
# ------------------------------------------------------------------------------
import pandas as pd
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid", palette="pastel", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 300

class BenchmarkAnalyzer:
    def __init__(self, results_path: str):
        self.df = self._load_results_to_dataframe(results_path)

    def _load_results_to_dataframe(self, path: str) -> pd.DataFrame:
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping malformed JSON line in {path}")
        
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
        print("\n1. Overall Performance (Average Scores)")
        
        overall_scores = self.df.groupby('method')[['bert_score_f1', 'rouge_l_f1', 'cosine_similarity']].mean()
        
        print(overall_scores.round(4).to_markdown(index=True))
        return overall_scores

    def analyze_performance_by_category(self):
        print("\n2. Performance by Query Category (Average Scores)")
        
        if 'query_category' not in self.df.columns:
            print("'query_category' column not found. Skipping this analysis.")
            return
            
        category_scores = self.df.groupby(['query_category', 'method'])[['bert_score_f1', 'rouge_l_f1', 'cosine_similarity']].mean()
        
        print(category_scores.round(4).to_markdown(index=True))
        return category_scores.reset_index()

    def analyze_win_loss_record(self, primary_metric: str = 'bert_score_f1'):
        print(f"\n3. 'Best Method' Analysis (based on highest {primary_metric})")

        metric_df = self.df.pivot(index='id', columns='method', values=primary_metric)
        
        metric_df['best_method'] = metric_df.idxmax(axis=1)
        
        win_counts = metric_df['best_method'].value_counts().reset_index()
        win_counts.columns = ['method', 'win_count']
        
        win_counts['win_percentage'] = (win_counts['win_count'] / win_counts['win_count'].sum() * 100).round(2)

        print(win_counts.to_markdown(index=False))
        return win_counts

    def _save_plot(self, plot: plt.Figure, filename: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        plot.savefig(path, bbox_inches='tight')
        plt.close(plot)
        print(f"Saved plot: {path}")

    def plot_overall_performance(self, overall_scores: pd.DataFrame, output_dir: str):
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
        metrics = ['bert_score_f1', 'rouge_l_f1', 'cosine_similarity']
        metric_names = ['BERT F1', 'ROUGE-L F1', 'Cosine Similarity']
        
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
            return
            
        overall_scores = self.analyze_overall_performance()
        category_scores = self.analyze_performance_by_category()
        bert_wins = self.analyze_win_loss_record(primary_metric='bert_score_f1')
        cosine_wins = self.analyze_win_loss_record(primary_metric='cosine_similarity')
        
        if output_dir:
            print(f"\n Generating visualizations in directory: {output_dir}")
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
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="benchmark_plots"
    )
    args = parser.parse_args()

    analyzer = BenchmarkAnalyzer(args.input)
    analyzer.run_full_analysis(output_dir=args.output_dir)

if __name__ == "__main__":
    main()