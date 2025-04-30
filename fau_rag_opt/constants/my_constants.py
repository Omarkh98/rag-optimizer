from pathlib import Path

CONFIG_FILE_PATH = Path("fau_rag_opt/config/config.yaml")

VECTORSTORE_PATH = "fau_rag_opt/knowledgebase/vector_index_fau.faiss"

METADATA_PATH = "fau_rag_opt/knowledgebase/quality_html-pdf.jsonl"

SAMPLED_PATH = "fau_rag_opt/knowledgebase/sampled_queries.jsonl"

LABELLED_PATH = "fau_rag_opt/knowledgebase/labeled_sampled_queries.jsonl"

HYBRID_ALPHA = 0.5

TOP_K = 5

SAVE_INTERVAL = 10