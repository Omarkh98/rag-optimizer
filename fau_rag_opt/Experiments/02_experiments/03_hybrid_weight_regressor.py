from fau_rag_opt.query_classifier.hybrid_weight_regressor import HybridWeightRegressor
from fau_rag_opt.constants.my_constants import CONFIG_FILE_PATH
from fau_rag_opt.helpers.utils import read_yaml

import json
import joblib
import warnings

warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.",
    category=FutureWarning
)

config_filepath = CONFIG_FILE_PATH
config = read_yaml(config_filepath)
model = config["retriever_transformer"]["transformer"]

query_file_path = "fau_rag_opt/knowledgebase/evaluated_encoded_samples.jsonl"
num_queries_to_process = 50

retrieved_results = []

with open(query_file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= num_queries_to_process:
            break
        try:
                query_data = json.loads(line)
                query = query_data.get("query")
                if query is None:
                    print(f"Skipping line {i+1} due to missing 'query' key: {line.strip()}")
                    continue

                reg = HybridWeightRegressor(model)
                reg.regressor = reg.regressor = joblib.load("fau_rag_opt/classifiers/hybrid_weight_regressor.joblib")

                alpha = reg.get_dynamic_alpha(query)
                print(f"🔁 Dynamic alpha (dense weight): {alpha:.3f}, (sparse weight): {1-alpha:.3f}")

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {i+1}: {e} - Line: {line.strip()}")
        except Exception as e:
            print(f"An error occurred while processing query on line {i+1}: {e}")