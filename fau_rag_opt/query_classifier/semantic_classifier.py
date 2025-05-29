# ------------------------------------------------------------------------------
# query_classifier/semantic_classifier.py - Enhancing the Baseline Classifier with Semantic Understanding.
# ------------------------------------------------------------------------------
"""
Classifer uses a Sentence Transformer instead of basic TF-IDF vectorizer which lacks 
semantic understanding for query representation.

Initial Run - model: Logistic Regression.
Second Run - model: XGBClassifier.
Third Run - model: MLPClassifier.

Finally, classifies the input query as "Dense-0", "Sparse-1", or "Hybrid-2"
"""

import json
import joblib
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

from ..constants.my_constants import CONFIG_FILE_PATH

from ..helpers.utils import read_yaml

class SemanticClassifier:
    def __init__(self, model_name: str):
        self.embedder = SentenceTransformer(model_name)
        # self.logistic_model = LogisticRegression(max_iter = 1000, class_weight = 'balanced')
        self.xgb_model = XGBClassifier(use_label_encoder = False, eval_metric = 'mlogloss')
        # self.mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), 
        #                   max_iter=500, 
        #                   activation='relu', 
        #                   solver='adam', 
        #                   early_stopping=True,
        #                   random_state=42)

    def load_data(self, input_path: str):
        queries, labels = [], []
        with open(input_path, 'r') as file:
            for line in file:
                obj = json.loads(line)
                queries.append(obj["query"])
                labels.append(obj["label_encoded"])
        return queries, labels
    
    def encode_queries(self, queries):
        print("Encoding queries with Sentence Transformer...")
        return self.embedder.encode(queries, show_progress_bar = True)
    
    def train(self, X_train, y_train):
        print("Training model...")
        self.xgb_model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test, labels_names = None):
        print("Evaluating model...")
        predictions = self.xgb_model.predict(X_test)
        report = classification_report(y_test, predictions, target_names = labels_names)
        matrix = confusion_matrix(y_test, predictions)
        
        print("Classification Report: \n")
        print(report)

        print("Confusion Matrix: \n")
        print(matrix)

        return report, matrix
    
    def save(self, model_path: str):
        joblib.dump(self.xgb_model, model_path)
        print(f"💾 Saved model to '{model_path}'")

    def load(self, model_path: str):
        self.xgb_model = joblib.load(model_path)
        print(f"✅ Loaded model and embedder.")

if __name__ == "__main__":
        import argparse
        from sklearn.model_selection import train_test_split

        parser = argparse.ArgumentParser(description = "Semantic Query Classifier (dense/sparse/hybrid)")
        parser.add_argument("--input", type=str, required=True, help="Path to the input labeled, encoded query data - knowledgebase/evaluated_encoded_samples.jsonl")
        parser.add_argument("--classifier_out", type=str, default="classifier_mlp.joblib", help="Output path for trained model - classifier_mlp.joblib")
        args = parser.parse_args()

        config_filepath = CONFIG_FILE_PATH
        config = read_yaml(config_filepath)
        model_path = config["retriever_transformer"]["transformer"]

        # 1. Initialize classifier
        semantic_classifier = SemanticClassifier(model_path)

        # 2. Load and Encode data
        queries, labels = semantic_classifier.load_data(args.input)
        X = semantic_classifier.encode_queries(queries)

        # 3. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.2, stratify = labels, random_state = 42)

        # 4. Train
        semantic_classifier.train(X_train, y_train)

        # 5. Evaluate
        semantic_classifier.evaluate(X_test, y_test, labels_names = ["dense", "sparse", "hybrid"])

        # 6. Save
        semantic_classifier.save(model_path = args.classifier_out)

        # python -m  fau_rag_opt.query_classifier.semantic_classifier --input fau_rag_opt/knowledgebase/evaluated_encoded_samples.jsonl --classifier_out classifier_mlp.joblib