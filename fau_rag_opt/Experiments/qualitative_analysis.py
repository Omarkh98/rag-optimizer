# ------------------------------------------------------------------------------
# qualitative_analysis.py - Analyzes the human-scored qualitative results.
# ------------------------------------------------------------------------------
import pandas as pd
import argparse

class QualitativeAnalyzer:
    def __init__(self, results_path: str):
        """
        Initializes the analyzer by loading the qualitative results CSV.
        """
        try:
            self.df = pd.read_csv(results_path)
            # --- UPDATED: Correctly identify all score columns from your CSV ---
            score_columns = ['correctness', 'completeness', 'conciseness', 'fluency_coherence']
            
            # Ensure all score columns exist before proceeding
            missing_cols = [col for col in score_columns if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing expected score columns in CSV: {missing_cols}")

            # Calculate the average score for each rated answer
            self.df['average_score'] = self.df[score_columns].mean(axis=1)
            print(f"✅ Successfully loaded and processed {len(self.df)} evaluations for {self.df['query_id'].nunique()} unique queries.")
        except FileNotFoundError:
            print(f"❌ Error: Results file not found at '{results_path}'")
            self.df = pd.DataFrame()
        except ValueError as e:
            print(f"❌ Error: {e}")
            self.df = pd.DataFrame()


    def analyze_overall_performance(self):
        """
        Calculates and prints the average scores for each method across all queries.
        """
        print("\n--- 1. Overall Qualitative Performance (Average Scores) ---")
        
        # Define the columns to aggregate, including the new average_score
        score_columns = ['correctness', 'completeness', 'conciseness', 'fluency_coherence', 'average_score']
        
        # Group by retrieval method and calculate the mean for each score column
        overall_scores = self.df.groupby('method')[score_columns].mean()
        
        # Sort by the average_score to easily see the winner
        overall_scores = overall_scores.sort_values(by='average_score', ascending=False)
        
        print(overall_scores.round(3).to_markdown(index=True))

    def analyze_performance_by_category(self):
        """
        Calculates and prints average scores for each method, grouped by query category.
        """
        print("\n--- 2. Qualitative Performance by Query Category (Average Score) ---")
        
        if 'category' not in self.df.columns:
            print("⚠️ 'category' column not found. Skipping this analysis.")
            return
            
        # Group by both category and method, then calculate the mean of the 'average_score'
        # .unstack() pivots the 'method' to be the columns for easier comparison
        category_scores = self.df.groupby(['category', 'method'])['average_score'].mean().unstack()
        
        print(category_scores.round(3).to_markdown(index=True))

    def run_full_analysis(self):
        """Runs all analysis methods sequentially."""
        if self.df.empty:
            print("❌ DataFrame is empty. Cannot run analysis.")
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
        required=True, 
        help="Path to the qualitative results CSV file (e.g., qualitative_eval_results.csv)."
    )
    args = parser.parse_args()

    analyzer = QualitativeAnalyzer(args.input)
    analyzer.run_full_analysis()