from pathlib import Path

CONFIG_FILE_PATH = Path("fau_rag_opt/config/config.yaml")

VECTORSTORE_PATH = "fau_rag_opt/knowledgebase/vector_index_fau.faiss"

METADATA_PATH = "fau_rag_opt/knowledgebase/quality_html-pdf.jsonl"

HYBRID_ALPHA = 0.5

TOP_K = 5