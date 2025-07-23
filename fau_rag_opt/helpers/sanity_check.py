# ------------------------------------------------------------------------------
# sanity_check.py - A minimal script to test for library conflicts.
# ------------------------------------------------------------------------------
from bert_score import score
import numpy as np
import faiss
from transformers import AutoTokenizer
from bert_score import score

def run_check():
    """
    This function loads the libraries most likely to conflict.
    If this script runs without a segmentation fault, the environment is stable.
    """
    print("--- Sanity Check Initializing ---")
    
    try:
        P, R, F1 = score(cands=["hello world"], refs=["hello there"], lang="en", model_type="distilroberta-base", device="cpu")
        print("F1", {F1})

        # 1. Test FAISS
        print("✅ [1/2] Initializing FAISS...")
        d = 64  # dimension
        nb = 100  # database size
        xb = np.random.random((nb, d)).astype('float32')
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        print("✅ FAISS initialized and used successfully.")

        # 2. Test Transformers
        print("\n✅ [2/2] Initializing Hugging Face Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        text = "Hello, world!"
        encoded_input = tokenizer(text, return_tensors='pt')
        print("✅ Hugging Face Tokenizer initialized and used successfully.")

        print("\n--- ✅✅✅ Sanity Check PASSED! ---")
        print("Your environment appears to be stable. You can now run the main benchmark script.")

    except Exception as e:
        print(f"\n--- ❌❌❌ Sanity Check FAILED! ---")
        print(f"An error occurred: {e}")
        print("There is still an issue with your environment.")

if __name__ == "__main__":
    run_check()