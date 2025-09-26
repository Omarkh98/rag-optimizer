# RAG Optimizer: Optimization Strategies for Retrieval-Augmented Generation

## Overview

This repository contains the core implementation for the thesis project titled:

> **“Optimization Strategies for Balancing Accuracy and Responsiveness in Retrieval-Augmented Generation”**  
> conducted at Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).

The objective of this work is to research, implement, and evaluate advanced retrieval optimization techniques for Retrieval-Augmented Generation (RAG) systems — focusing on adaptive hybrid retrieval, query-type classification, and retrieval strategy selection to improve both accuracy and responsiveness.

---

## Thesis Project Information

**Title:** Optimization Strategies for Balancing Accuracy and Responsiveness in Retrieval-Augmented Generation  
**Author:** Omar Lotfy  
**Institution:** Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)  
**Department:** Data Science.  
**Supervisor:** Sebastian Wind  
**Submission Date:** 01.10.2025

## Key Features

- **Adaptive Hybrid Retrieval**: Dynamic routing of queries to optimal retrieval methods (dense, sparse, hybrid, filtered vector search)
- **Query Type Classification**: Automatic classification of queries to determine the most suitable retrieval strategy
- **Query Optimization**: Advanced techniques including query expansion, self-query rewriting, and hybrid filtering
- **Modular Architecture**: Extensible design for easy integration with existing RAG pipelines
- **Performance Monitoring**: Built-in evaluation metrics for accuracy and latency tracking
- **FAU-Specific Optimizations**: Tailored for university administrative scenarios with adaptability for broader domains

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | Python 3.10+ | Core implementation language |
| **API Framework** | FastAPI | High-performance REST API |
| **Vector Database** | FAISS | Dense vector storage and retrieval |
| **LLM Integration** | OpenAI API / Local Models | Generation component |
| **Containerization** | Docker | Deployment and reproducibility |

## Project Structure

```
rag-optimizer/
│
├── fau_rag_opt/                       # Main package
│   ├── __init__.py
│   ├── constants/
│   │   └── my_constants.py
│   ├── experiments/
│   │   ├── retrieval_level_analysis.py
│   │   └── ahr_results_demo.py
│   ├── helpers/
│   │   ├── __init__.py
│   │   ├── labelling.py
│   │   └── utils.py
│   ├── knowledgebase/
│   │   ├── __init__.py
│   │   └── categorized_queries.jsonl
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── mcp_multi_tool_ahr_pipeline.py
│   │   ├── query_classification_tool.py
│   │   ├── self_querying_tool.py
│   │   ├── query_expansion_tool.py
│   │   └── reranker_tool.py
│   ├── query_classifier/
│   │   ├── __init__.py
│   │   ├── ahr_training_data_generator.py
│   │   ├── labeler_config.py
│   │   └── llm_response.py
│   ├── retrievers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── dense.py
│   │   ├── sparse.py
│   │   └── hybrid.py
│   └── optimization/
│       ├── __init__.py
│       └── query_optimizer.py
│
├── config/
│   └── default_config.yaml
│
├── scripts/                           # Utility scripts for running experiments, data prep, etc.
│   └── run_pipeline.py
│
├── tests/                             # Unit and integration tests
│   ├── __init__.py
│   ├── test_retrievers.py
│   ├── test_query_classifier.py
│   └── test_mcp_tools.py
│
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.