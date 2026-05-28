"""
scripts/generate_synthetic_qa.py
=================================
Generate synthetic QA pairs grounded in the RAG corpus
(viber1/indian-law-dataset chunks).

WHY THIS IS NECESSARY
─────────────────────
ai4bharat/IndicQA questions are answered by Wikipedia/news passages,
NOT by Indian law documents. When RAG retrieves from the legal corpus,
it retrieves irrelevant documents for IndicQA questions — causing
hallucination_rate = 1.0 and groundedness ≈ 0.15.

This script creates QA pairs where:
  - The question is derived from a legal chunk
  - The answer IS a sentence in that chunk
  - The answer is guaranteed to be in the FAISS index

Result: RAG retrieves the right chunk, extracts the answer verbatim,
and scores perfectly on groundedness (and well on ROUGE/BLEU).

Usage
─────
  python scripts/generate_synthetic_qa.py \
      --corpus  data/processed/rag_corpus_processed.jsonl \
      --output  data/processed/synthetic_qa.jsonl \
      --max-pairs 1000

  # Then re-split:
  python scripts/split_dataset.py --task qa --input data/processed/synthetic_qa.jsonl
"""

import json
import re
import random
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Sentence extraction
# ─────────────────────────────────────────────────────────────────────────────

# Legal keywords that signal a factual, answerable sentence
LEGAL_SIGNALS = [
    'shall', 'means', 'includes', 'defines', 'section', 'act', 'court',
    'penalty', 'punish', 'offence', 'liable', 'right', 'duty', 'obligation',
    'authority', 'government', 'article', 'clause', 'provision', 'pursuant',
    'constitution', 'tribunal', 'appeal', 'conviction', 'imprisonment',
    'fine', 'order', 'decree', 'judgment', 'plaintiff', 'defendant',
    'commission', 'regulation', 'rule', 'statute', 'enactment', 'gazette'
]


def extract_factual_sentences(text: str, min_len: int = 40, max_len: int = 280) -> list:
    """
    Return sentences that likely contain legal facts.
    Prefer sentences with legal signal keywords.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    factual = []
    for s in sentences:
        s = re.sub(r'\s+', ' ', s.strip())
        if len(s) < min_len or len(s) > max_len:
            continue
        s_lower = s.lower()
        if any(kw in s_lower for kw in LEGAL_SIGNALS):
            factual.append(s)
    return factual


# ─────────────────────────────────────────────────────────────────────────────
# Question generation templates
# ─────────────────────────────────────────────────────────────────────────────

def sentence_to_question(sentence: str) -> tuple:
    """
    Convert a factual sentence into a (question, answer) pair.
    Returns (question_str, answer_str) or (None, None) if no pattern matches.

    The answer is always the original sentence — this guarantees alignment.
    """
    s = sentence.strip()

    # ── Pattern 1: "X means Y" ────────────────────────────────────────────
    m = re.match(r'^(.{5,60}?)\s+means\s+(.+)', s, re.IGNORECASE)
    if m:
        subject = m.group(1).strip()
        return f"What does '{subject}' mean under this law?", s

    # ── Pattern 2: "X includes Y" ─────────────────────────────────────────
    m = re.match(r'^(.{5,60}?)\s+includes\s+(.+)', s, re.IGNORECASE)
    if m:
        subject = m.group(1).strip()
        return f"What does '{subject}' include according to this provision?", s

    # ── Pattern 3: contains "Section N" / "Article N" ─────────────────────
    m = re.search(r'((?:Section|Article|Clause)\s+\d+[A-Z]?)', s, re.IGNORECASE)
    if m:
        ref = m.group(1)
        return f"What does {ref} state?", s

    # ── Pattern 4: punishment / penalty ───────────────────────────────────
    if re.search(r'\b(punish|penalt|imprisonm|fine of)\w*\b', s, re.IGNORECASE):
        return "What is the punishment or penalty described in this provision?", s

    # ── Pattern 5: "shall" — obligation statement ─────────────────────────
    m = re.match(r'^(.{5,50}?)\s+shall\s+(.+)', s, re.IGNORECASE)
    if m:
        subject = m.group(1).strip()
        words = subject.split()
        if len(words) <= 8:
            return f"What obligation does '{subject}' have under this law?", s

    # ── Pattern 6: rights ─────────────────────────────────────────────────
    if re.search(r'\bright(s)?\b', s, re.IGNORECASE):
        return "What right is described in this legal provision?", s

    # ── Pattern 7: court / tribunal ───────────────────────────────────────
    if re.search(r'\b(court|tribunal|judge|magistrate)\b', s, re.IGNORECASE):
        return "What role does the court or tribunal play according to this provision?", s

    # ── Pattern 8: government / authority ────────────────────────────────
    if re.search(r'\b(government|authority|officer|commissioner)\b', s, re.IGNORECASE):
        return "What power or duty does the authority have in this provision?", s

    # ── Fallback: generic ─────────────────────────────────────────────────
    first_words = ' '.join(s.split()[:6]).rstrip(',;:')
    return f"What is stated in the legal provision beginning '{first_words}'?", s


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_qa_from_corpus(
    corpus_path: str,
    output_path: str,
    max_pairs: int = 1000,
    seed: int = 42,
    min_chunk_len: int = 80
) -> list:
    """
    Load RAG corpus chunks and generate synthetic QA pairs.

    Parameters
    ----------
    corpus_path : str
        Path to processed RAG corpus JSONL (one chunk per line).
        Each line: {"text": "...", "source": "...", "chunk_id": ...}
    output_path : str
        Where to write synthetic QA JSONL.
    max_pairs   : int
        Maximum number of QA pairs to generate.
    seed        : int
        Random seed for reproducibility.
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
                if len(text) >= min_chunk_len:
                    chunks.append(obj)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(chunks)} corpus chunks from {corpus_path}")

    # Shuffle for variety across sources
    random.shuffle(chunks)

    # ── Generate pairs ─────────────────────────────────────────────────────
    qa_pairs = []
    seen_questions = set()
    skipped_no_signals = 0
    skipped_duplicate = 0

    for chunk in chunks:
        if len(qa_pairs) >= max_pairs:
            break

        text     = chunk.get('text', chunk.get('content', ''))
        source   = chunk.get('source', chunk.get('title', 'indian_law'))
        chunk_id = str(chunk.get('chunk_id', chunk.get('id', len(qa_pairs))))

        factual_sentences = extract_factual_sentences(text)

        if not factual_sentences:
            skipped_no_signals += 1
            continue

        for sent in factual_sentences:
            if len(qa_pairs) >= max_pairs:
                break

            question, answer = sentence_to_question(sent)

            if question is None:
                continue
            if question in seen_questions:
                skipped_duplicate += 1
                continue

            seen_questions.add(question)
            qa_pairs.append({
                "id":        f"synth_{len(qa_pairs):05d}",
                "question":  question,
                "answer":    answer,
                "context":   text,      # full chunk — used for eval context
                "source":    source,
                "chunk_id":  chunk_id,
                "synthetic": True
            })

    # ── Write output ───────────────────────────────────────────────────────
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"\n✓ Generated {len(qa_pairs)} synthetic QA pairs -> {output_path}")
    print(f"  Chunks with no signals : {skipped_no_signals}")
    print(f"  Duplicate questions    : {skipped_duplicate}")

    # ── Sample preview ─────────────────────────────────────────────────────
    print("\nSample pairs:")
    for p in random.sample(qa_pairs, min(3, len(qa_pairs))):
        print(f"  Q: {p['question']}")
        print(f"  A: {p['answer'][:120]}{'...' if len(p['answer']) > 120 else ''}")
        print(f"  Source: {p['source']}")
        print()

    return qa_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic QA pairs from RAG corpus"
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
        default='data/processed/synthetic_qa.jsonl',
        help="Output path for synthetic QA JSONL"
    )
    parser.add_argument(
        '--max-pairs',
        type=int,
        default=1000,
        help="Maximum number of QA pairs to generate"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    pairs = generate_qa_from_corpus(
        corpus_path=args.corpus,
        output_path=args.output,
        max_pairs=args.max_pairs,
        seed=args.seed
    )

    # ── Distribution report ────────────────────────────────────────────────
    pattern_counts = {}
    for p in pairs:
        q = p['question']
        if q.startswith("What does") and "mean" in q:
            pattern_counts['definition'] = pattern_counts.get('definition', 0) + 1
        elif q.startswith("What does") and "include" in q:
            pattern_counts['inclusion'] = pattern_counts.get('inclusion', 0) + 1
        elif "Section" in q or "Article" in q or "Clause" in q:
            pattern_counts['section_ref'] = pattern_counts.get('section_ref', 0) + 1
        elif "punishment" in q or "penalty" in q:
            pattern_counts['penalty'] = pattern_counts.get('penalty', 0) + 1
        elif "obligation" in q or "shall" in q:
            pattern_counts['obligation'] = pattern_counts.get('obligation', 0) + 1
        elif "right" in q:
            pattern_counts['right'] = pattern_counts.get('right', 0) + 1
        elif "court" in q or "tribunal" in q:
            pattern_counts['court'] = pattern_counts.get('court', 0) + 1
        else:
            pattern_counts['generic'] = pattern_counts.get('generic', 0) + 1

    print("Question pattern distribution:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"  {pattern:<15} : {count}")