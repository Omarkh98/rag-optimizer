# ------------------------------------------------------------------------------
# hybrid_weight_regressor.py - Trains an XGBoost model for AHR.
# ------------------------------------------------------------------------------
import pandas as pd
import json
import numpy as np
import argparse
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor  # Using XGBoost instead of MLP

class HybridWeightRegressor:
    def __init__(self):
        self.regressor = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1  # Use all available CPU cores
        )
        print("Initializing a new XGBoost Regressor...")

    def load_training_data(self, input_path: str):
        print("📂 Loading and preparing training data...")
        df = pd.read_csv(input_path)
        
        # Safely parse the string representation of the embedding list
        X = np.array(df['query_embedding'].apply(json.loads).tolist(), dtype=np.float32)
        y = df['optimal_alpha'].values.astype(np.float32)
        
        print(f"📊 Loaded {len(df)} samples.")
        return X, y

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        print(f"🚀 Training hybrid weight regressor on {len(X_train)} samples...")
        self.regressor.fit(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        print("\n--- Model Evaluation ---")
        preds = self.regressor.predict(X_test)
        
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        print(f"📊 Mean Squared Error (MSE) on test set: {mse:.6f}")
        print(f"🎯 R-squared (R²) on test set: {r2:.4f}")
        print("------------------------\n")
        
        print("🔍 Sample Predictions vs. True Values:")
        for i in range(min(5, len(X_test))):
            print(f"  - Predicted Alpha: {preds[i]:.3f} | True Alpha: {y_test[i]:.3f}")

        return preds, y_test

    def save_model(self, path: str):
        joblib.dump(self.regressor, path)
        print(f"💾 Saved trained model to {path}")

    def visualize_performance(self, y_true, y_pred, output_path: str):
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', label='Predictions')
        plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel("True Optimal Alpha")
        plt.ylabel("Predicted Alpha")
        plt.title("AHR Model Performance: True vs. Predicted Alpha")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.savefig(output_path)
        print(f"📈 Performance visualization saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the AHR regression model.")
    parser.add_argument("--input", required=True, help="Path to the ahr_training_data.csv file.")
    parser.add_argument("--model_out", default="ahr_alpha_regressor.joblib", help="Path to save the trained model.")
    parser.add_argument("--plot_out", default="ahr_performance_plot.png", help="Path to save the performance plot.")
    args = parser.parse_args()

    reg = HybridWeightRegressor()
    X, y = reg.load_training_data(args.input)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    reg.train(X_train, y_train)
    preds, true_vals = reg.evaluate(X_test, y_test)
    
    reg.save_model(args.model_out)
    reg.visualize_performance(true_vals, preds, args.plot_out)

    print("✅ Done.")