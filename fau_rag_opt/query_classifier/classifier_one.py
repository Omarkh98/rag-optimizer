# ------------------------------------------------------------------------------
# query_classifier/classifier_one.py - Train classifier to predict best RAG retrieval method.
# ------------------------------------------------------------------------------
"""
Input: Query

Output: "Dense-0", "Sparse-1", or "Hybrid-2"

Description:
"""

import argparse
import joblib
import json
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from typing import List, Tuple

from ..constants.my_constants import CONFIG_FILE_PATH

from ..helpers.utils import read_yaml


class ClassifierOne:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def load_data(self, input_path: str) -> Tuple[List[str], List[str]]:
        queries, labels = [], []
        with open(input_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    queries.append(data["query"])
                    labels.append(data["label_encoded"])
                except Exception:
                    continue
        return queries, labels
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        embeddings = self.model.encode(queries, show_progress_bar = True, convert_to_numpy = True)
        return embeddings.astype(np.float32)

    def train_classifier(self, X: np.ndarray, y: List[str]) -> MLPClassifier:
        X = np.array(X).astype(np.float32)
        y = np.array(y).astype(int)

        clf = MLPClassifier(
            hidden_layer_sizes=(768, 384, 128),
            activation='relu',
            solver='adam',
            alpha=1e-5,                      # ↓ regularization
            batch_size=32,                  # ↑ finer updates
            learning_rate_init=0.001,
            learning_rate='adaptive',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42,
            verbose=True
        )
        print("Training MLP Classifier...")
        clf.fit(X, y)
        return clf

    def evaluate_classifier(self, classifier, X_test, y_test):
        y_pred = classifier.predict(X_test)

        print("\n🔍 Classification Report:\n")
        print(classification_report(y_test, y_pred, digits = 4))

        print("\n🔍 Confusion Matrix:\n")
        print(confusion_matrix(y_test, y_pred))

    def main(self, args):
        print(f"📂 Loading data from {args.input} ...")
        queries, labels = self.load_data(args.input)

        print(f"📊 Loaded {len(queries)} queries. Training-test split...")
        queries_train, queries_test, labels_train, labels_test = train_test_split(
            queries, labels, test_size=0.15, stratify=labels, random_state=42
        )

        print("🔗 Encoding queries using SentenceTransformer...")
        X_train = self.encode_queries(queries_train)
        X_test = self.encode_queries(queries_test)

        print("⚖️ Applying RandomOverSampler to balance classes...")
        ros = RandomOverSampler(random_state=42)
        X_train, labels_train = ros.fit_resample(X_train, labels_train)

        print("🎯 Training classifier...")
        clf = self.train_classifier(X_train, labels_train)

        print("📈 Evaluating classifier...")
        self.evaluate_classifier(clf, X_test, labels_test)

        print("💾 Saving classifier...")
        joblib.dump(clf, args.classifier_out)

        print("✅ Done.")

if __name__ == "__main__":
    config_filepath = CONFIG_FILE_PATH
    config = read_yaml(config_filepath)
    model = config["retriever_transformer"]["transformer"]

    clfone = ClassifierOne(model)

    parser = argparse.ArgumentParser(description = "Train classifier to predict best RAG retrieval method.")
    parser.add_argument("--input", type=str, required=True, help="Path to - knowledgebase/evaluated_encoded_samples.jsonl")
    parser.add_argument("--classifier_out", type=str, default="classifier_one.joblib", help="Path to save classifier - classifier_one.joblib")
    args = parser.parse_args()

    clfone.main(args)

# python -m  fau_rag_opt.query_classifier.classifier_one --input fau_rag_opt/knowledgebase/evaluated_encoded_samples.jsonl --classifier_out retrieval_classifier.joblib