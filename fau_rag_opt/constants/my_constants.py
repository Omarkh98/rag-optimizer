from pathlib import Path

CONFIG_FILE_PATH = Path("fau_rag_opt/config/config.yaml")

VECTORSTORE_PATH = "fau_rag_opt/knowledgebase/vector_index_fau.faiss"

METADATA_PATH = "fau_rag_opt/knowledgebase/quality_html-pdf.jsonl"

SAMPLED_PATH = "fau_rag_opt/knowledgebase/sampled_queries.jsonl"

LABELLED_PATH = "fau_rag_opt/knowledgebase/labeled_sampled_queries.jsonl"

HYBRID_ALPHA = 0.5

TOP_K = 10

TOP_N = 5

SAVE_INTERVAL = 10

HYBRID_ALPHA_MAP = {'comparative': 0.8, 
                    'ambiguous': 0.5,
                    'factual': 0.6,
                    'default': 0.6 }