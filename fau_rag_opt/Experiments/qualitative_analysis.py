# ------------------------------------------------------------------------------
# qualitative_analysis.py - Analyzes the human-scored qualitative results.
# ------------------------------------------------------------------------------
import pandas as pd
import argparse

class QualitativeAnalyzer:
    def __init__(self, results_path: str):
        try:
            self.df = pd.read_csv(results_path)
            score_columns = ['correctness', 'completeness', 'conciseness', 'fluency_coherence']
            
            missing_cols = [col for col in score_columns if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing expected score columns in CSV: {missing_cols}")

            self.df['average_score'] = self.df[score_columns].mean(axis=1)
        except FileNotFoundError:
            print(f"Error: Results file not found at '{results_path}'")
            self.df = pd.DataFrame()
        except ValueError as e:
            print(f"Error: {e}")
            self.df = pd.DataFrame()


    def analyze_overall_performance(self):
        print("\n--- 1. Overall Qualitative Performance (Average Scores) ---")
        
        score_columns = ['correctness', 'completeness', 'conciseness', 'fluency_coherence', 'average_score']
        
        overall_scores = self.df.groupby('method')[score_columns].mean()
        overall_scores = overall_scores.sort_values(by='average_score', ascending=False)
        
        print(overall_scores.round(3).to_markdown(index=True))

    def analyze_performance_by_category(self):
        print("\n--- 2. Qualitative Performance by Query Category (Average Score) ---")
        
        if 'category' not in self.df.columns:
            return
        
        category_scores = self.df.groupby(['category', 'method'])['average_score'].mean().unstack()
        
        print(category_scores.round(3).to_markdown(index=True))

    def run_full_analysis(self):
        if self.df.empty:
            return
            
        self.analyze_overall_performance()
        self.analyze_performance_by_category()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the results of the qualitative evaluation."
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True
        )
    args = parser.parse_args()

    analyzer = QualitativeAnalyzer(args.input)
    analyzer.run_full_analysis()