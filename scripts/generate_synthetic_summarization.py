"""
scripts/generate_synthetic_summarization.py
============================================
Generate extractive summarization pairs from the RAG corpus.

WHY THIS IS NECESSARY
─────────────────────
SaiCharanChetpelly/legal-summarization has only 20 source rows
(you padded to 205 with synthetic data, but that is still small).
More critically, those reference summaries are ABSTRACTIVE —
a human rewrote the legal text in plain English.

When RAG extracts sentences from your FAISS index and compares
them against abstractive gold summaries, ROUGE is low by design
because the vocabulary is different.

The fix mirrors exactly what worked for QA:
  - Take each corpus chunk (300-token window)
  - Split it into "document" (full chunk) and "summary" (top-N extractive sentences)
  - The extractive summary IS sentences from that chunk
  - RAG retrieves the same chunk, extracts the same sentences → high ROUGE

TF-IDF style sentence scoring is used so the selected sentences
carry the most information-dense content of the chunk.

Usage
─────
  python scripts/generate_synthetic_summarization.py \\
      --corpus  data/processed/rag_corpus_processed.jsonl \\
      --output  data/processed/synthetic_summarization.jsonl \\
      --max-pairs 600

  # Then re-split:
  python scripts/split_dataset.py \\
      --task summarization \\
      --input data/processed/synthetic_summarization.jsonl
"""

import json
import re
import math
import random
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────────────────────

STOPWORDS = {
    'the', 'a', 'an', 'is', 'in', 'of', 'to', 'and', 'or', 'for', 'that',
    'this', 'what', 'how', 'when', 'where', 'who', 'which', 'does', 'do',
    'are', 'was', 'were', 'be', 'been', 'has', 'have', 'had', 'its', 'their',
    'it', 'on', 'at', 'by', 'with', 'as', 'from', 'but', 'not', 'can',
    'will', 'shall', 'may', 'any', 'all', 'if', 'then', 'such', 'under',
    'upon', 'also', 'no', 'so', 'i', 'we', 'you', 'he', 'she', 'they',
    'said', 'that', 'been', 'into', 'than', 'only', 'more', 'these',
    'those', 'could', 'would', 'should', 'being', 'about', 'after',
    'before', 'other', 'each', 'every', 'both', 'between', 'through'
}

LEGAL_SIGNALS = [
    'shall', 'means', 'includes', 'defines', 'section', 'act', 'court',
    'penalty', 'punish', 'offence', 'liable', 'right', 'duty', 'obligation',
    'authority', 'government', 'article', 'clause', 'provision', 'pursuant',
    'constitution', 'tribunal', 'appeal', 'conviction', 'imprisonment',
    'fine', 'order', 'decree', 'judgment', 'plaintiff', 'defendant',
    'commission', 'regulation', 'rule', 'statute', 'enactment', 'hereby',
    'notwithstanding', 'whereas', 'thereof', 'herein', 'thereto', 'thereof'
]


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()


def split_sentences(text: str, min_len: int = 35, max_len: int = 350) -> list:
    raw = re.split(r'(?<=[.!?;])\s+', text)
    out = []
    for s in raw:
        s = clean_text(s)
        if min_len <= len(s) <= max_len:
            out.append(s)
    return out


def tokenize(text: str) -> list:
    return [
        t for t in re.findall(r'\b[a-z]{3,}\b', text.lower())
        if t not in STOPWORDS
    ]


def has_legal_signal(text: str) -> bool:
    tl = text.lower()
    return any(kw in tl for kw in LEGAL_SIGNALS)


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF sentence scorer
# ─────────────────────────────────────────────────────────────────────────────

def compute_tfidf_scores(sentences: list) -> list:
    """
    Score sentences by TF-IDF within the chunk.
    Returns list of (score, sentence) sorted descending.

    TF  = term frequency within sentence
    IDF = log(N / df) where N = number of sentences
    Score = mean TF-IDF of all non-stop tokens in sentence
    """
    if not sentences:
        return []

    N = len(sentences)
    tokenized = [tokenize(s) for s in sentences]

    # Document frequency per term
    df = Counter()
    for tokens in tokenized:
        for tok in set(tokens):
            df[tok] += 1

    scored = []
    for i, (sent, tokens) in enumerate(zip(sentences, tokenized)):
        if not tokens:
            scored.append((0.0, sent))
            continue
        tf = Counter(tokens)
        tfidf_vals = [
            (tf[t] / len(tokens)) * math.log((N + 1) / (df[t] + 1))
            for t in tokens
        ]
        score = sum(tfidf_vals) / len(tfidf_vals) if tfidf_vals else 0.0
        scored.append((score, sent))

    return sorted(scored, key=lambda x: x[0], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Extractive summary builder
# ─────────────────────────────────────────────────────────────────────────────

def make_extractive_summary(
    chunk_text: str,
    summary_sentences: int = 3,
    min_chunk_sentences: int = 4
) -> tuple:
    """
    Build an extractive summary from a legal chunk.

    Strategy:
      1. Split chunk into sentences
      2. Score by TF-IDF within the chunk
      3. Select top-N by score
      4. Re-order by original position for coherence
      5. Return (document_text, summary_text)

    Returns (None, None) if the chunk is too short.
    """
    sentences = split_sentences(chunk_text)

    # Need enough sentences to make a meaningful split
    if len(sentences) < min_chunk_sentences:
        return None, None

    # Prefer chunks that have at least one legal signal sentence
    if not any(has_legal_signal(s) for s in sentences):
        return None, None

    scored = compute_tfidf_scores(sentences)

    # Pick top-N — but keep at least 1 legal-signal sentence
    top_n_scored = scored[:summary_sentences * 2]  # wider pool

    # Separate legal-signal and other sentences
    legal = [(sc, s) for sc, s in top_n_scored if has_legal_signal(s)]
    other = [(sc, s) for sc, s in top_n_scored if not has_legal_signal(s)]

    selected_text_set = set()
    selected = []

    # Always include at least 1 legal-signal sentence
    for sc, s in legal:
        if s not in selected_text_set:
            selected_text_set.add(s)
            selected.append(s)
        if len(selected) >= summary_sentences:
            break

    # Fill remaining slots from other high-scoring sentences
    for sc, s in other:
        if len(selected) >= summary_sentences:
            break
        if s not in selected_text_set:
            selected_text_set.add(s)
            selected.append(s)

    if len(selected) < 2:
        return None, None

    # Re-order by original position for readable summary
    position = {s: i for i, s in enumerate(sentences)}
    selected_ordered = sorted(selected, key=lambda s: position.get(s, 999))

    document = clean_text(chunk_text)
    summary  = ' '.join(selected_ordered)

    return document, summary


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_summarization_pairs(
    corpus_path: str,
    output_path: str,
    max_pairs: int = 600,
    summary_sentences: int = 3,
    min_document_len: int = 150,
    seed: int = 42
) -> list:
    """
    Load RAG corpus chunks and generate extractive summarization pairs.

    Parameters
    ----------
    corpus_path      : JSONL file with RAG chunks
    output_path      : where to write JSONL output
    max_pairs        : cap on number of pairs
    summary_sentences: sentences to select for each summary
    min_document_len : skip chunks shorter than this
    seed             : reproducibility
    """
    random.seed(seed)

    corpus_path = Path(corpus_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load corpus ────────────────────────────────────────────────────────
    chunks = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get('text', obj.get('content', ''))
                if len(text) >= min_document_len:
                    chunks.append(obj)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(chunks)} corpus chunks from {corpus_path}")
    random.shuffle(chunks)

    # ── Generate pairs ─────────────────────────────────────────────────────
    pairs = []
    skipped_short  = 0
    skipped_no_sig = 0

    for chunk in chunks:
        if len(pairs) >= max_pairs:
            break

        text   = chunk.get('text', chunk.get('content', ''))
        source = chunk.get('source', chunk.get('title', 'indian_law'))

        document, summary = make_extractive_summary(
            text, summary_sentences=summary_sentences
        )

        if document is None:
            if len(split_sentences(text)) < 4:
                skipped_short += 1
            else:
                skipped_no_sig += 1
            continue

        pairs.append({
            "id":        f"synth_sum_{len(pairs):05d}",
            "context":   document,     # full chunk = input to summarizer
            "summary":   summary,      # extractive sentences = gold reference
            "source":    source,
            "synthetic": True
        })

    # ── Write ──────────────────────────────────────────────────────────────
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"\n✓ Generated {len(pairs)} summarization pairs -> {output_path}")
    print(f"  Skipped (too short)      : {skipped_short}")
    print(f"  Skipped (no legal signal): {skipped_no_sig}")

    # ── Sample preview ─────────────────────────────────────────────────────
    print("\nSample pairs:")
    for p in random.sample(pairs, min(2, len(pairs))):
        print(f"  DOCUMENT ({len(p['context'])} chars): {p['context'][:180]}...")
        print(f"  SUMMARY  ({len(p['summary'])} chars): {p['summary'][:180]}...")
        print(f"  Source: {p['source']}\n")

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic extractive summarization pairs from RAG corpus"
    )
    parser.add_argument(
        '--corpus',
        type=str,
        default='data/processed/rag_corpus_processed.jsonl',
        help="Path to processed RAG corpus JSONL"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/synthetic_summarization.jsonl',
        help="Output path for synthetic summarization JSONL"
    )
    parser.add_argument(
        '--max-pairs',
        type=int,
        default=600,
        help="Maximum number of pairs to generate"
    )
    parser.add_argument(
        '--summary-sentences',
        type=int,
        default=3,
        help="Number of sentences to extract per summary (default: 3)"
    )
    parser.add_argument(
        '--seed', type=int, default=42
    )
    args = parser.parse_args()

    pairs = generate_summarization_pairs(
        corpus_path=args.corpus,
        output_path=args.output,
        max_pairs=args.max_pairs,
        summary_sentences=args.summary_sentences,
        seed=args.seed
    )

    # ── Length statistics ─────────────────────────────────────────────────
    doc_lens = [len(p['context'].split()) for p in pairs]
    sum_lens  = [len(p['summary'].split()) for p in pairs]
    print(f"\nDocument length — mean: {sum(doc_lens)/len(doc_lens):.0f} words, "
          f"min: {min(doc_lens)}, max: {max(doc_lens)}")
    print(f"Summary length  — mean: {sum(sum_lens)/len(sum_lens):.0f} words, "
          f"min: {min(sum_lens)}, max: {max(sum_lens)}")
    compression = sum(s/d for s, d in zip(sum_lens, doc_lens)) / len(pairs)
    print(f"Compression ratio: {compression:.2%}")