# evaluation/evaluate_corpus.py
"""
Corpus retrieval evaluation for LegalAId.
Measures: Hit Rate @K, MRR, Context Coverage, Avg Retrieved Length
Uses qa_test.jsonl as probe queries against the FAISS index.
"""

import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent.parent
FAISS_DIR = BASE_DIR / "data" / "faiss_index"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "results"

INDEX_PATH = FAISS_DIR / "legal_index.faiss"
METADATA_PATH = FAISS_DIR / "metadata.pkl"
QA_TEST_PATH = PROCESSED_DIR / "qa_test.jsonl"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 8


def load_index():
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def load_qa_test(max_samples=200):
    samples = []
    with open(QA_TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples[:max_samples]


def token_overlap(text_a: str, text_b: str) -> float:
    """Simple token overlap ratio (answer in chunk)."""
    a_tokens = set(text_a.lower().split())
    b_tokens = set(text_b.lower().split())
    if not a_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens)


def evaluate_corpus():
    print("Loading FAISS index and metadata...")
    index, metadata = load_index()

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Loading QA test set...")
    qa_samples = load_qa_test()
    print(f"  Evaluating on {len(qa_samples)} QA samples")

    hit_at_k = []       # 1 if any retrieved chunk overlaps with answer
    mrr_scores = []     # 1/rank of first relevant chunk
    coverage_scores = []  # fraction of answer tokens covered by top-K chunks
    avg_chunk_lens = []

    for sample in qa_samples:
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        if not question or not answer:
            continue

        # Embed question
        q_vec = model.encode([question], normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype=np.float32)

        # Search FAISS
        distances, indices = index.search(q_vec, TOP_K)
        retrieved_indices = indices[0]

        # Get retrieved chunks
        retrieved_chunks = []
        for idx in retrieved_indices:
            if idx < 0 or idx >= len(metadata):
                continue
            chunk_text = metadata[idx].get("text", "")
            retrieved_chunks.append(chunk_text)

        if not retrieved_chunks:
            hit_at_k.append(0)
            mrr_scores.append(0.0)
            coverage_scores.append(0.0)
            continue

        # Compute overlap of answer with each chunk
        overlaps = [token_overlap(answer, chunk) for chunk in retrieved_chunks]

        # Hit@K: at least one chunk has overlap > threshold
        OVERLAP_THRESHOLD = 0.25
        hit = int(any(o >= OVERLAP_THRESHOLD for o in overlaps))
        hit_at_k.append(hit)

        # MRR: 1 / rank of first relevant chunk
        mrr = 0.0
        for rank, o in enumerate(overlaps, start=1):
            if o >= OVERLAP_THRESHOLD:
                mrr = 1.0 / rank
                break
        mrr_scores.append(mrr)

        # Coverage: how much of the answer is covered by combined top-K text
        combined_text = " ".join(retrieved_chunks)
        coverage = token_overlap(answer, combined_text)
        coverage_scores.append(coverage)

        # Avg chunk length
        avg_chunk_lens.append(np.mean([len(c.split()) for c in retrieved_chunks]))

    results = {
        "num_samples": len(hit_at_k),
        "hit_rate_at_k": round(float(np.mean(hit_at_k)), 4),
        "mrr": round(float(np.mean(mrr_scores)), 4),
        "avg_coverage": round(float(np.mean(coverage_scores)), 4),
        "avg_retrieved_chunk_words": round(float(np.mean(avg_chunk_lens)), 1),
        "top_k_used": TOP_K,
        "overlap_threshold": 0.25,
    }

    print("\n========== CORPUS RETRIEVAL EVALUATION ==========")
    print(f"  Samples evaluated:         {results['num_samples']}")
    print(f"  Hit Rate @{TOP_K}:              {results['hit_rate_at_k']:.4f}")
    print(f"  MRR:                       {results['mrr']:.4f}")
    print(f"  Avg Coverage (answer in top-K): {results['avg_coverage']:.4f}")
    print(f"  Avg Chunk Length (words):  {results['avg_retrieved_chunk_words']}")
    print("=================================================\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "corpus_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

    return results


if __name__ == "__main__":
    evaluate_corpus()