# ------------------------------------------------------------------------------
# query_classifier/hybrid_weight_regressor.py - 
# ------------------------------------------------------------------------------
"""
Input:

Output:

Description:
"""

import json
import numpy as np
import pandas as pd
import argparse
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import joblib

class HybridWeightRegressor:
    def __init__(self, embedder_model_name: str, model_path: str = None):
        print(f"Initializing embedder with model: {embedder_model_name}")
        self.embedder = SentenceTransformer(embedder_model_name)

        if model_path:
            print(f"Loading pre-trained regressor from {model_path}...")
            self.regressor = joblib.load(model_path)
        else:
            print("Initializing a new MLP Regressor...")
            self.regressor = MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                alpha=1e-4,
                max_iter=500,
                random_state=42,
                verbose=True,
                early_stopping=True
            )
    
    def load_training_data(self, input_path: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Loads the training data and returns embeddings (X), alphas (y), and the raw DataFrame.
        """
        df = pd.read_csv(input_path)
        
        # The embeddings are stored as JSON strings; parse them.
        X = np.array([json.loads(embedding) for embedding in df['query_embedding']])
        y = df['optimal_alpha'].values
        
        return X, y, df
    
    def train(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Trains the MLP regressor and returns the test sets for evaluation.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"🚀 Training hybrid weight regressor on {len(X_train)} samples...")
        self.regressor.fit(X_train, y_train)
        
        return X_test, y_test
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Evaluates the trained model on the test set with MSE and R-squared.
        """
        preds = self.regressor.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        print(f"\n--- Model Evaluation ---")
        print(f"📊 Mean Squared Error (MSE) on test set: {mse:.6f}")
        print(f"🎯 R-squared (R²) on test set: {r2:.4f}")
        print("------------------------")
        
        print("\n🔍 Sample Predictions vs. True Values:")
        for i in range(min(5, len(preds))):
            print(f"  - Predicted Alpha: {preds[i]:.3f} | True Alpha: {y_test[i]:.3f}")
        
        return preds
    
    def predict_alpha(self, query: str, clip_range: tuple[float, float] = (0.0, 1.0)) -> float:
        """
        Predicts the optimal alpha for a single query string.
        """
        embedding = self.embedder.encode([query], convert_to_numpy=True)
        alpha = self.regressor.predict(embedding)[0]
        return float(np.clip(alpha, *clip_range))
    
    def save_model(self, path: str):
        """Saves the trained regressor model to a file."""
        joblib.dump(self.regressor, path)
        print(f"\n💾 Saved trained model to {path}")

    def visualize_performance(self, y_test: np.ndarray, y_pred: np.ndarray, output_path: str):
        """
        Generates and saves a scatter plot of predicted vs. true alpha values.
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', label='Predictions')
        
        # Add a perfect prediction line (y=x)
        ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel("True Optimal Alpha", fontsize=12)
        ax.set_ylabel("Predicted Alpha", fontsize=12)
        ax.set_title("AHR Model: Predicted vs. True Alpha", fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"📈 Performance visualization saved to {output_path}")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a regressor to predict the optimal hybrid retrieval alpha.")
    parser.add_argument("--input", required=True, help="Path to the training data CSV file (ahr_training_data.csv).")
    parser.add_argument("--model_out", default="ahr_alpha_regressor.joblib", help="Path to save the trained model file.")
    parser.add_argument("--plot_out", default="ahr_performance_plot.png", help="Path to save the performance plot.")
    # Add argument for embedder model
    parser.add_argument("--embedder", default="all-MiniLM-L6-v2", help="Name of the SentenceTransformer model to use.")
    args = parser.parse_args()

    # Initialize a new model for training
    ahr_model = HybridWeightRegressor(embedder_model_name=args.embedder)

    # Load the data
    print("📂 Loading and preparing training data...")
    X, y, _ = ahr_model.load_training_data(args.input)
    print(f"📊 Loaded {len(X)} samples.")

    # Train the model and get the test set for evaluation
    X_test, y_test = ahr_model.train(X, y)

    # Evaluate the model and get predictions
    y_pred = ahr_model.evaluate(X_test, y_test)

    # Save the trained model
    ahr_model.save_model(args.model_out)
    
    # Create and save the performance visualization
    ahr_model.visualize_performance(y_test, y_pred, args.plot_out)

    print("\n✅ AHR model training complete.")

    # def load_data(self, input_path: str) -> Tuple[List[str], List[str]]:
    #     queries, weights = [], []
    #     with open(input_path, 'r') as file:
    #         for line in file:
    #             try:
    #                 data = json.loads(line)
    #                 queries.append(data["query"])
    #                 d, s, h = data["dense_score"], data["sparse_score"], data["hybrid_score"]
    #                 total = d + s + h
    #                 weights.append([d / total, s / total, h / total])
    #             except Exception as e:
    #                 print(f"❌ Skipping malformed line: {e}")
    #     return queries, weights
    

    # def encode_queries(self, queries: List[str]) -> np.ndarray:
    #     embeddings = self.embedder.encode(queries, show_progress_bar = True, convert_to_numpy = True)
    #     return embeddings.astype(np.float32)
    
    # def train(self, X: np.ndarray, y: np.ndarray):
    #     print("🚀 Training hybrid weight regressor...")
    #     self.regressor.fit(X, y)

    # def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
    #     preds = self.regressor.predict(X_test)
    #     mse = mean_squared_error(y_test, preds)
    #     print(f"\n📊 MSE on test set: {mse:.6f}")
    #     print("🔍 Sample predicted vs true weights:")
    #     for i in range(5):
    #         print(f"Pred: {np.round(preds[i], 3)} | True: {np.round(y_test[i], 3)}")

    # def predict(self, query: str) -> List[float]:
    #     embedding = self.embedder.encode([query], convert_to_numpy = True).astype(np.float32)
    #     weights = self.regressor.predict(embedding)[0]
    #     return weights / weights.sum()
    
    # def get_dynamic_alpha(self, query: str, clip_range: Tuple[float, float] = (0.2, 0.8)) -> float:
    #     """Compute dynamic alpha (dense weight) for hybrid retrieval based on query."""
    #     weights = self.predict(query)
    #     dense_w, sparse_w, _ = weights
    #     alpha = dense_w / (dense_w + sparse_w)
    #     return float(np.clip(alpha, *clip_range))

    # def save_model(self, path: str):
    #     joblib.dump(self.regressor, path)
    #     print(f"💾 Saved model to {path}")
    
    # def visualize_predictions(self, queries: List[str], true_weights: List[List[float]], num: int = 10):
    #     X = self.encode_queries(queries[:num])
    #     preds = self.regressor.predict(X)

    #     for i in range(num):
    #         plt.figure(figsize=(6, 3))
    #         x = np.arange(3)
    #         plt.bar(x - 0.15, true_weights[i], width=0.3, label='True', color='skyblue')
    #         plt.bar(x + 0.15, preds[i], width=0.3, label='Predicted', color='salmon')
    #         plt.xticks(x, ['Dense', 'Sparse', 'Hybrid'])
    #         plt.ylim(0, 1)
    #         plt.title(f'Query {i+1}: {queries[i][:50]}...')
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show()

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Train a regressor to predict dense/sparse/hybrid weights.")
#     parser.add_argument("--input", required=True, help="Path to evaluated_samples.jsonl")
#     parser.add_argument("--model_out", default="hybrid_weight_regressor.joblib", help="Path to save model")
#     args = parser.parse_args()

#     config_filepath = CONFIG_FILE_PATH
#     config = read_yaml(config_filepath)
#     model = config["retriever_transformer"]["transformer"]

#     reg = HybridWeightRegressor(model)

#     print("📂 Loading data...")
#     queries, weights = reg.load_data(args.input)

#     print(f"📊 Loaded {len(queries)} samples")
#     X = reg.encode_queries(queries)
#     y = np.array(weights, dtype=np.float32)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
#     reg.train(X_train, y_train)
#     reg.evaluate(X_test, y_test)
#     reg.save_model(args.model_out)

#     print("✅ Done.")

# python -m  fau_rag_opt.query_classifier.hybrid_weight_regressor --input fau_rag_opt/knowledgebase/evaluated_encoded_samples.jsonl --model_out hybrid_weight_regressor.joblib