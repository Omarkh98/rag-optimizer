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
├── benchmark_plots
│   ├── ahr_performance_plot.png
│   ├── category_bert_score_f1.png
│   ├── category_cosine_similarity.png
│   ├── category_rouge_l_f1.png
│   ├── overall_performance.png
│   ├── retireval_level_evaluation_by_method.png
│   └── win_comparison.png
├── fau_rag_opt
│   ├── config
│   │   ├── config.yaml
│   │   └── __init__.py
│   ├── constants
│   │   ├── __init__.py
│   │   └── my_constants.py
│   ├── experiments
│   │   ├── 01_experiments
│   │   │   ├── 01_label_distribution.ipynb
│   │   │   ├── 03_baseline_classifier.ipynb
│   │   │   ├── 05_classifiers_comp.ipynb
│   │   │   ├── 06_classifier_testing.ipynb
│   │   │   └── images
│   │   │       ├── Classifier_Performance_Comp.png
│   │   │       ├── Initial_Label_Dist.png
│   │   │       └── New_Label_Dist.png
│   │   ├── 02_experiments
│   │   │   ├── 01_label_distribution_encoding.ipynb
│   │   │   ├── 02_label_quality_diagnosis.ipynb
│   │   │   ├── 03_hybrid_weight_regressor.py
│   │   │   └── images
│   │   │       └── Initial_Label_Distribution.png
│   │   ├── ahr_alpha_regressor.joblib
│   │   ├── benchmark_analysis_viz.py
│   │   ├── final_benchmark_demo.py
│   │   ├── __init__.py
│   │   ├── qualitative_analysis.py
│   │   ├── qualitative_eval_app.py
│   │   └── retrieval_level_analysis.py
│   ├── helpers
│   │   ├── encode_worker.py
│   │   ├── exception.py
│   │   ├── __init__.py
│   │   ├── labelling.py
│   │   ├── loader.py
│   │   ├── rate_limiter.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── knowledgebase
│   │   ├── ahr_training_data.csv
│   │   ├── ahr_training_data.csv.per_alpha.jsonl
│   │   ├── categorized_queries.jsonl
│   │   ├── convos.jsonl
│   │   ├── extracted_queries.jsonl
│   │   ├── __init__.py
│   │   ├── qualitative_eval_results.csv
│   │   ├── quality_html-pdf.jsonl
│   │   ├── retrieval_benchmark.jsonl
│   │   └── vector_index_fau.faiss
│   ├── mcp
│   │   ├── __init__.py
│   │   ├── mcp_multi_tool_ahr_pipeline.py
│   │   ├── query_classification_tool.py
│   │   ├── query_expansion_tool.py
│   │   ├── reranker_tool.py
│   │   ├── reranker_worker.py
│   │   └── self_querying_tool.py
│   ├── query_classifier
│   │   ├── ahr_training_data_generator.py
│   │   ├── dataset_categorization.py
│   │   ├── dataset_extractor.py
│   │   ├── eval_answer_quality.py
│   │   ├── hybrid_weight_regressor_xg.py
│   │   ├── hybrid_weight_regressor.py
│   │   ├── __init__.py
│   │   ├── labeler_config.py
│   │   ├── llm_response.py
│   │   └── retrieval_benchmark.py
│   └── retrievers
│       ├── base.py
│       ├── dense.py
│       ├── hybrid.py
│       ├── __init__.py
│       └── sparse.py
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
└── tests
    ├── __init__.py
    ├── pytest.ini
    ├── test_dense_retrieval.py
    ├── test_hybrid_retrieval.py
    └── test_sparse_retrieval.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.