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
from typing import List, Tuple
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import joblib

from ..constants.my_constants import CONFIG_FILE_PATH

from ..helpers.utils import read_yaml

class HybridWeightRegressor:
    def __init__(self, model_name: str):
        self.embedder = SentenceTransformer(model_name)
        self.regressor = MLPRegressor(
            hidden_layer_sizes=(768, 384, 128),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            alpha=1e-5,
            max_iter=300,
            random_state=42,
            verbose=True
        )

    def load_data(self, input_path: str) -> Tuple[List[str], List[str]]:
        queries, weights = [], []
        with open(input_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    queries.append(data["query"])
                    d, s, h = data["dense_score"], data["sparse_score"], data["hybrid_score"]
                    total = d + s + h
                    weights.append([d / total, s / total, h / total])
                except Exception as e:
                    print(f"❌ Skipping malformed line: {e}")
        return queries, weights
    

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        embeddings = self.embedder.encode(queries, show_progress_bar = True, convert_to_numpy = True)
        return embeddings.astype(np.float32)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        print("🚀 Training hybrid weight regressor...")
        self.regressor.fit(X, y)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        preds = self.regressor.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        print(f"\n📊 MSE on test set: {mse:.6f}")
        print("🔍 Sample predicted vs true weights:")
        for i in range(5):
            print(f"Pred: {np.round(preds[i], 3)} | True: {np.round(y_test[i], 3)}")

    def predict(self, query: str) -> List[float]:
        embedding = self.embedder.encode([query], convert_to_numpy = True).astype(np.float32)
        weights = self.regressor.predict(embedding)[0]
        return weights / weights.sum()
    
    def get_dynamic_alpha(self, query: str, clip_range: Tuple[float, float] = (0.2, 0.8)) -> float:
        """Compute dynamic alpha (dense weight) for hybrid retrieval based on query."""
        weights = self.predict(query)
        dense_w, sparse_w, _ = weights
        alpha = dense_w / (dense_w + sparse_w)
        return float(np.clip(alpha, *clip_range))

    def save_model(self, path: str):
        joblib.dump(self.regressor, path)
        print(f"💾 Saved model to {path}")
    
    def visualize_predictions(self, queries: List[str], true_weights: List[List[float]], num: int = 10):
        X = self.encode_queries(queries[:num])
        preds = self.regressor.predict(X)

        for i in range(num):
            plt.figure(figsize=(6, 3))
            x = np.arange(3)
            plt.bar(x - 0.15, true_weights[i], width=0.3, label='True', color='skyblue')
            plt.bar(x + 0.15, preds[i], width=0.3, label='Predicted', color='salmon')
            plt.xticks(x, ['Dense', 'Sparse', 'Hybrid'])
            plt.ylim(0, 1)
            plt.title(f'Query {i+1}: {queries[i][:50]}...')
            plt.legend()
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a regressor to predict dense/sparse/hybrid weights.")
    parser.add_argument("--input", required=True, help="Path to evaluated_samples.jsonl")
    parser.add_argument("--model_out", default="hybrid_weight_regressor.joblib", help="Path to save model")
    args = parser.parse_args()

    config_filepath = CONFIG_FILE_PATH
    config = read_yaml(config_filepath)
    model = config["retriever_transformer"]["transformer"]

    reg = HybridWeightRegressor(model)

    print("📂 Loading data...")
    queries, weights = reg.load_data(args.input)

    print(f"📊 Loaded {len(queries)} samples")
    X = reg.encode_queries(queries)
    y = np.array(weights, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    reg.train(X_train, y_train)
    reg.evaluate(X_test, y_test)
    reg.save_model(args.model_out)

    print("✅ Done.")

    # python -m  fau_rag_opt.query_classifier.hybrid_weight_regressor --input fau_rag_opt/knowledgebase/evaluated_encoded_samples.jsonl --model_out hybrid_weight_regressor.joblib