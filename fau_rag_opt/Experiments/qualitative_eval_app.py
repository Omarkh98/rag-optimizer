import json
import random
import pandas as pd
import streamlit as st
from pathlib import Path

BENCHMARK_FILE = "fau_rag_opt/knowledgebase/retrieval_benchmark.jsonl"
OUTPUT_CSV = "fau_rag_opt/experiments/qualitative_eval_results.csv"


def initialize_session():
    """
    This function runs only once per session. It loads, filters, shuffles,
    and stores the dataset in the session_state to ensure stability.
    """
    # 1. Check if the results file exists and load evaluated IDs
    evaluated_ids = set()
    results_path_obj = Path(OUTPUT_CSV)
    if results_path_obj.exists() and results_path_obj.stat().st_size > 0:
        try:
            results_df = pd.read_csv(OUTPUT_CSV)
            if "query_id" in results_df.columns:
                evaluated_ids = set(results_df["query_id"].unique())
        except pd.errors.EmptyDataError:
            pass

    # 2. Load the full benchmark dataset
    full_dataset = []
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                full_dataset.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    st.session_state.total_queries = len(full_dataset)

    # 3. Filter out already evaluated queries
    unevaluated_queries = [
        item for item in full_dataset if item.get("id") not in evaluated_ids
    ]

    # 4. Process and shuffle the remaining queries ONCE
    processed_dataset = []
    for item in unevaluated_queries:
        if not all(k in item for k in ["dense_result", "sparse_result", "hybrid_result"]):
            continue

        answers = {
            "Dense": item["dense_result"]["generated_answer"],
            "Sparse": item["sparse_result"]["generated_answer"],
            "Hybrid": item["hybrid_result"]["generated_answer"],
        }

        shuffled = list(answers.items())
        random.shuffle(shuffled) # Shuffle happens here and only here
        anonymized = [{"label": chr(65 + i), "answer": ans, "method": method}
                      for i, (method, ans) in enumerate(shuffled)]

        processed_dataset.append({
            "id": item.get("id", ""),
            "query": item.get("query", ""),
            "category": item.get("query_category", ""),
            "ground_truth": item.get("ground_truth", ""),
            "answers": anonymized # Store the shuffled order
        })
    
    # 5. Store the prepared, stable dataset in the session state
    st.session_state.dataset = processed_dataset
    st.session_state.current_index = 0
    st.session_state.queries_done_count = len(evaluated_ids)


def save_progress(results):
    """Append results to CSV."""
    df = pd.DataFrame(results)
    output_path = Path(OUTPUT_CSV)
    file_exists_and_has_content = output_path.exists() and output_path.stat().st_size > 0
    df.to_csv(OUTPUT_CSV, mode="a", header=not file_exists_and_has_content, index=False)


def main():
    st.set_page_config(page_title="Qualitative Evaluation Tool", layout="centered")
    st.title("🎯 Qualitative Evaluation Tool")
    st.write("Rate each generated answer on a 1–5 scale for multiple criteria.")

    if "dataset" not in st.session_state:
        initialize_session()

    dataset = st.session_state.dataset
    idx = st.session_state.current_index

    if not dataset or idx >= len(dataset):
        st.success("✅ All queries have been evaluated!")
        st.write("You can find the complete results in:", OUTPUT_CSV)
        return

    sample = dataset[idx]
    
    total_queries = st.session_state.total_queries
    queries_done = st.session_state.queries_done_count
    
    st.subheader(f"Evaluating Query {queries_done + 1} of {total_queries}")
    st.progress((queries_done + 1) / total_queries)

    st.markdown(f"**Category:** `{sample['category']}`")
    st.markdown(f"**User Query:** {sample['query']}")

    st.markdown(
        f"<div style='background-color:#e8f5e9; padding:10px; border-radius:5px;'><b>Ground Truth Answer:</b><br>{sample['ground_truth']}</div>",
        unsafe_allow_html=True
    )

    ratings = []
    for ans in sample["answers"]:
        st.markdown(f"### Answer {ans['label']}")
        st.write(ans["answer"])

        c1, c2, c3, c4 = st.columns(4)
        correctness = c1.slider(f"Correctness ({ans['label']})", 1, 5, 3, key=f"correctness_{idx}_{ans['label']}")
        completeness = c2.slider(f"Completeness ({ans['label']})", 1, 5, 3, key=f"completeness_{idx}_{ans['label']}")
        conciseness = c3.slider(f"Conciseness ({ans['label']})", 1, 5, 3, key=f"conciseness_{idx}_{ans['label']}")
        coherence_fluency = c4.slider(f"Fluency/Coherence ({ans['label']})", 1, 5, 3, key=f"coherence_fluency_{idx}_{ans['label']}")

        ratings.append({
            "query_id": sample["id"],
            "query": sample["query"],
            "category": sample["category"],
            "method": ans["method"],
            "correctness": correctness,
            "completeness": completeness,
            "conciseness": conciseness,
            "fluency_coherence": coherence_fluency
        })

    if st.button("Save & Next"):
        save_progress(ratings)
        st.session_state.current_index += 1
        st.session_state.queries_done_count += 1
        st.rerun()

if __name__ == "__main__":
    main()