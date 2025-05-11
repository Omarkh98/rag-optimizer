# ------------------------------------------------------------------------------
# query_classifier/baseline_classifier.py - Initial Baseline classifier.
# ------------------------------------------------------------------------------
"""
Loads `labeled_encoded_queries.jsonl`, splits the data into train, test, and
validation and trains the model using TF-ID vectorizer and Logistic Regression.
Finally, classifies the input query as "Dense-0", "Sparse-1", or "Hybrid-2"
"""

import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


class BaselineClassifier:
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.df = None
        self.vectorizer = TfidfVectorizer(max_features = 5000, ngram_range = (1,2))
        self.model = LogisticRegression(max_iter = 1000, class_weight = 'balanced')

    def load_data(self):
        print("Loading data...")
        with open(self.input_path, 'r') as file:
            lines = [json.loads(line) for line in file]

        self.df = pd.DataFrame(lines)
        print(f"✅ Loaded {len(self.df)} queries.")

    def preprocess(self):
        print("Preprocessing data...")
        self.X = self.vectorizer.fit_transform(self.df["query"])
        self.y = self.df["label_encoded"]

    def train(self, test_size: float = 0.2, random_state: int = 42):
        print("Training baseline classifier...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size = test_size, random_state = random_state, stratify = self.y)
        self.model.fit(X_train, y_train)

        print("Evaluating...")
        y_pred = self.model.predict(X_test)
        print("✅ Classification Report:\n")
        print(classification_report(y_test, y_pred, target_names=["dense", "sparse", "hybrid"]))
        print("🧩 Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    def save(self, model_path: str, vectorizer_path: str):
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"💾 Model and vectorizer saved to '{model_path}' and '{vectorizer_path}'")

if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser(description = "Train Query Type Classifier (dense/sparse/hybrid)")
        parser.add_argument("--input", type=str, required=True, help="Path to the input labeled, encoded query data - knowledgebase/labeled_encoded_queries.jsonl")
        parser.add_argument("--output", type=str, default="query_classifier.joblib", help="Output path for trained classifier - query_classifier.joblib")
        parser.add_argument("--vectorizer_out", type=str, default="vectorizer.joblib", help="Output path for vectorizer - vectorizer.joblib")
        args = parser.parse_args()

        base_classifier = BaselineClassifier(args.input)
        base_classifier.load_data()
        base_classifier.preprocess()
        base_classifier.train()
        base_classifier.save(model_path = args.output, vectorizer_path = args.vectorizer_out)