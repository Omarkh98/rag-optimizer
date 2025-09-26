# ------------------------------------------------------------------------------
# query_classifier/hybrid_weight_regressor.py - Train a regressor to predict the optimal hybrid retrieval alpha.
# ------------------------------------------------------------------------------
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
        self.embedder = SentenceTransformer(embedder_model_name)

        if model_path:
            self.regressor = joblib.load(model_path)
        else:
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
        df = pd.read_csv(input_path)
        
        X = np.array([json.loads(embedding) for embedding in df['query_embedding']])
        y = df['optimal_alpha'].values
        
        return X, y, df
    
    def train(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.regressor.fit(X_train, y_train)
        
        return X_test, y_test
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        preds = self.regressor.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        print(f"Mean Squared Error (MSE) on test set: {mse:.6f}")
        print(f"R-squared (R²) on test set: {r2:.4f}")
        
        for i in range(min(5, len(preds))):
            print(f"  - Predicted Alpha: {preds[i]:.3f} | True Alpha: {y_test[i]:.3f}")
        
        return preds
    
    def predict_alpha(self, query: str, clip_range: tuple[float, float] = (0.0, 1.0)) -> float:
        embedding = self.embedder.encode([query], convert_to_numpy=True)
        alpha = self.regressor.predict(embedding)[0]
        return float(np.clip(alpha, *clip_range))
    
    def save_model(self, path: str):
        joblib.dump(self.regressor, path)

    def visualize_performance(self, y_test: np.ndarray, y_pred: np.ndarray, output_path: str):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', label='Predictions')
        
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
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a regressor to predict the optimal hybrid retrieval alpha.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model_out", default="ahr_alpha_regressor.joblib")
    parser.add_argument("--plot_out", default="ahr_performance_plot.png")

    parser.add_argument("--embedder", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    ahr_model = HybridWeightRegressor(embedder_model_name=args.embedder)

    X, y, _ = ahr_model.load_training_data(args.input)

    X_test, y_test = ahr_model.train(X, y)

    y_pred = ahr_model.evaluate(X_test, y_test)

    ahr_model.save_model(args.model_out)
    
    ahr_model.visualize_performance(y_test, y_pred, args.plot_out)

    print("\n AHR model training complete with MLPRegressor")