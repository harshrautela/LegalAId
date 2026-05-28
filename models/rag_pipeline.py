"""
rag_pipeline.py — Extractive RAG Pipeline for LegalAId
=======================================================
WHAT CHANGED IN THIS VERSION (summarization update)
────────────────────────────────────────────────────
1. extractive_summary() is now TF-IDF-weighted, not just overlap-weighted.
   This matches the scoring used in generate_synthetic_summarization.py
   so retrieval and extraction stay consistent.

2. retrieve_for_summarization() uses a different query construction
   strategy — it extracts high-IDF key phrases from the input context
   and uses those as the FAISS query, instead of using a generic
   "Summarize this legal text:" prefix.

3. run_batch() now auto-detects task from input file fields if --task
   is not passed explicitly.

4. QA pipeline is UNCHANGED — all QA logic is identical to before.

5. answer_query() now returns a structured dict with:
   - answer
   - prediction
   - evidence (list of evidence items)
   - evidence_text
   - groundedness
   - confidence
   - unsupported_warning
   - warning
   so the UI can render comparison + evidence correctly.

FIXES APPLIED (Part 3 Fix Plan)
────────────────────────────────
Step 2: answer_question_rag() now accepts doc_chunks (list of dicts with
        'text' and 'source') from the uploaded PDF. When doc_chunks are
        provided they are used as the PRIMARY evidence — FAISS is only
        used when doc_chunks is empty (general knowledge query).

Step 5: Removed evidence_text[:800] truncation in groundedness scoring
        inside both answer_query() and answer_question_rag().
        The groundedness scorer now sees the FULL evidence string.
        GROUNDEDNESS_THRESHOLD should be set to 0.35 in config.py.

Modes:
  extractive  — copy exact sentences from retrieved chunks (DEFAULT)
  generative  — TinyLlama with strict evidence-only prompt
  hybrid      — extractive first; generative fallback if too short

Tasks:
  qa            — question answering (unchanged)
  summarization — TF-IDF extractive summarization (new)
"""

import json
import re
import math
import pickle
import logging
from pathlib import Path
from typing import Optional, List, Dict
from collections import Counter
import sys

import numpy as np
# NOTE: faiss, sentence_transformers, transformers and torch are all imported
# inside load_embedder() / load_generator() respectively to avoid import
# failures when packages are not installed, and to prevent the massive
# __path__ warning flood that crashes the Streamlit WebSocket on startup.

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    EMBEDDING_MODEL, BASE_MODEL_NAME, TOP_K, FAISS_INDEX_PATH,
    FAISS_METADATA_PATH, DEVICE, RAG_MODE, EXTRACTIVE_TOP_N,
    GROUNDEDNESS_THRESHOLD
)

# Optional — fall back to defaults if not in your config yet
try:
    from config import RERANK_TOP_N
except ImportError:
    RERANK_TOP_N = 3

try:
    from config import SUM_SENTENCES
except ImportError:
    SUM_SENTENCES = 3   # sentences per extractive summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Globals (lazy-loaded singletons)
# ─────────────────────────────────────────────────────────────────────────────
_embedder = None   # Optional[SentenceTransformer] — lazy loaded
_faiss_index = None
_metadata: Optional[list] = None
_tokenizer = None
_model = None

STOPWORDS = {
    'the', 'a', 'an', 'is', 'in', 'of', 'to', 'and', 'or', 'for', 'that',
    'this', 'what', 'how', 'when', 'where', 'who', 'which', 'does', 'do',
    'are', 'was', 'were', 'be', 'been', 'has', 'have', 'had', 'its', 'their',
    'it', 'on', 'at', 'by', 'with', 'as', 'from', 'but', 'not', 'can',
    'will', 'shall', 'may', 'any', 'all', 'if', 'then', 'such', 'under',
    'upon', 'also', 'no', 'so', 'i', 'we', 'you', 'he', 'she', 'they',
    'said', 'into', 'than', 'only', 'more', 'these', 'those', 'could',
    'would', 'should', 'being', 'about', 'after', 'before', 'other',
    'each', 'every', 'both', 'between', 'through', 'during', 'without'
}

LEGAL_SIGNALS = [
    'shall', 'means', 'includes', 'section', 'act', 'court', 'penalty',
    'punish', 'offence', 'liable', 'right', 'duty', 'obligation',
    'authority', 'government', 'article', 'clause', 'provision',
    'constitution', 'tribunal', 'appeal', 'conviction', 'imprisonment',
    'fine', 'order', 'decree', 'judgment', 'regulation', 'statute',
    'plaintiff', 'defendant', 'commission', 'notwithstanding', 'whereas',
    'thereof', 'herein', 'pursuant', 'hereby', 'enactment'
]


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_embedder():
    """Lazy-load the SentenceTransformer embedder."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        logger.info(f"Loading embedder: {EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def load_faiss_index():
    """Lazy-load the FAISS index and metadata."""
    import faiss  # lazy import
    global _faiss_index, _metadata
    if _faiss_index is None:
        logger.info(f"Loading FAISS index: {FAISS_INDEX_PATH}")
        _faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_METADATA_PATH, 'rb') as f:
            _metadata = pickle.load(f)
        logger.info(
            f"FAISS loaded — {_faiss_index.ntotal} vectors, "
            f"{len(_metadata)} metadata entries"
        )
    return _faiss_index, _metadata


def load_generator():
    global _tokenizer, _model
    if _tokenizer is None:
        # Lazy imports — keep transformers/torch out of module-level scope
        # so Streamlit doesn't trigger the __path__ warning flood on startup
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        logger.info(f"Loading generator: {BASE_MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        _model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto" if DEVICE == "cuda" else None
        )
        if DEVICE != "cuda":
            _model = _model.to(DEVICE)
        _model.eval()
    return _tokenizer, _model


# ─────────────────────────────────────────────────────────────────────────────
# Token / text helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_set(text: str) -> set:
    """Meaningful tokens as a set (for overlap scoring)."""
    return {
        t for t in re.findall(r'\b[a-z]{3,}\b', text.lower())
        if t not in STOPWORDS
    }


def _tokenize_list(text: str) -> list:
    """Meaningful tokens as a list (for TF-IDF counting)."""
    return [
        t for t in re.findall(r'\b[a-z]{3,}\b', text.lower())
        if t not in STOPWORDS
    ]


def _overlap(a: str, b: str) -> float:
    """Jaccard overlap between two token sets."""
    ta, tb = _tokenize_set(a), _tokenize_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _split_sentences(text: str, min_len: int = 30, max_len: int = 350) -> list:
    """Split text into filtered, non-trivial sentences."""
    raw = re.split(r'(?<=[.!?;])\s+', text.strip())
    return [s.strip() for s in raw if min_len <= len(s.strip()) <= max_len]


def _has_legal_signal(text: str) -> bool:
    tl = text.lower()
    return any(kw in tl for kw in LEGAL_SIGNALS)


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF within a chunk (for summarization scoring)
# ─────────────────────────────────────────────────────────────────────────────

def _tfidf_score_sentences(sentences: list) -> list:
    """
    Score each sentence by mean TF-IDF within the chunk.
    Returns [(score, sentence), ...] sorted descending.
    """
    if not sentences:
        return []
    N = len(sentences)
    tokenized = [_tokenize_list(s) for s in sentences]

    df = Counter()
    for tokens in tokenized:
        for tok in set(tokens):
            df[tok] += 1

    scored = []
    for sent, tokens in zip(sentences, tokenized):
        if not tokens:
            scored.append((0.0, sent))
            continue
        tf = Counter(tokens)
        vals = [
            (tf[t] / len(tokens)) * math.log((N + 1) / (df.get(t, 0) + 1))
            for t in tokens
        ]
        score = sum(vals) / len(vals) if vals else 0.0
        scored.append((score, sent))

    return sorted(scored, key=lambda x: x[0], reverse=True)


def _extract_key_phrases(text: str, top_n: int = 8) -> str:
    """
    Extract top-N high-IDF terms from a text block.
    Used to build a retrieval query for summarization.
    """
    tokens = _tokenize_list(text)
    if not tokens:
        return text[:200]

    # Use term frequency as a proxy for importance
    freq = Counter(tokens)
    top_terms = [term for term, _ in freq.most_common(top_n)]
    return ' '.join(top_terms)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(query: str, context: str = "", top_k: int = None) -> list:
    """
    QA retrieval: embed (question + context snippet), search FAISS,
    rerank by combined FAISS + lexical score.

    For broad definitional queries (e.g. "What is IPC?") we:
    1. Expand query with topic-specific anchor terms
    2. Fetch more candidates
    3. Hard-filter candidates to only those containing the query's core topic keywords
       This prevents specific sub-section chunks (e.g. IPC §354, §509) from
       dominating when the question is about the Act as a whole.
    """
    if top_k is None:
        top_k = TOP_K

    embedder = load_embedder()
    index, metadata = load_faiss_index()

    # Detect definitional/broad query
    q_lower = query.strip().lower()
    _definitional_starters = (
        "what is", "what are", "define", "definition of",
        "explain the", "describe the", "tell me about",
        "what does", "what do", "meaning of", "overview of",
    )
    is_definitional = any(q_lower.startswith(s) for s in _definitional_starters)

    # Extract core topic keywords from the query (strip question words)
    _q_stopwords = {
        'what', 'is', 'are', 'the', 'a', 'an', 'of', 'in', 'for', 'under',
        'does', 'do', 'define', 'definition', 'explain', 'describe', 'tell',
        'me', 'about', 'meaning', 'overview', 'how', 'why', 'when', 'where',
        'which', 'who', 'please', 'can', 'you', 'give', 'provide',
    }
    raw_tokens    = set(re.findall(r'\b[a-z]{2,}\b', q_lower))
    topic_kws     = raw_tokens - _q_stopwords

    # Abbreviation expansion for common Indian legal acronyms
    _abbr_expand = {
        'ipc':  {'ipc', 'indian', 'penal', 'code'},
        'crpc': {'crpc', 'criminal', 'procedure', 'code'},
        'cpc':  {'cpc', 'civil', 'procedure', 'code'},
        'iea':  {'evidence', 'act'},
    }
    for abbr, expansion in _abbr_expand.items():
        if abbr in topic_kws:
            topic_kws |= expansion

    if is_definitional:
        # Fetch more candidates for definitional queries
        top_k = max(top_k, 12)
        # Build a topic-specific anchor — use the actual topic words, not generic ones
        topic_anchor = " ".join(sorted(topic_kws)[:6])
        anchor = f"definition overview introduction purpose history {topic_anchor}"
        combined_query = f"{query.strip()} {anchor}"
    else:
        combined_query = query.strip()
        if context:
            combined_query = f"{query.strip()} {context.strip()[:200]}"

    q_vec = embedder.encode(
        [combined_query], convert_to_numpy=True, normalize_embeddings=True
    )

    fetch_k = min(top_k * 4, index.ntotal)
    distances, indices = index.search(q_vec.astype(np.float32), fetch_k)

    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        chunk = dict(metadata[idx])
        chunk['faiss_score'] = float(dist)
        candidates.append(chunk)

    # ── Topic relevance hard-filter for broad/definitional queries ────────────
    # Problem: FAISS cosine similarity is topic-agnostic — for "What is IPC?"
    # it returns chunks about §354 / §509 because they mention "IPC" and score
    # high on semantic similarity. These chunks anchor Groq to wrong sub-topics.
    # Fix: score each candidate by % of topic keywords it contains. Chunks that
    # don't cover the actual question subject are removed before reranking.
    if is_definitional and topic_kws:
        def _kw_score(chunk_text: str) -> float:
            cl = chunk_text.lower()
            hits = sum(1 for kw in topic_kws if kw in cl)
            return hits / len(topic_kws)

        # Compute keyword scores
        for c in candidates:
            c['kw_score'] = _kw_score(c.get('text', c.get('content', '')))

        # Keep candidates with meaningful topic overlap
        # Raised from 0.2 → 0.35: forces chunks to actually cover the question subject.
        # 0.2 was too loose — §354/§509 chunks that merely mention "IPC" passed easily.
        relevant = [c for c in candidates if c['kw_score'] >= 0.35]

        if relevant:
            candidates = relevant
        # If nothing passes the threshold, keep top-3 by kw_score as fallback
        else:
            candidates = sorted(candidates, key=lambda x: x['kw_score'], reverse=True)[:3]

    rerank_top = RERANK_TOP_N + 2 if is_definitional else RERANK_TOP_N
    reranked = _lexical_rerank(query, candidates, top_n=rerank_top)

    # ── Source-diversity guard ─────────────────────────────────────────────
    # Hard-caps any single section number to 2 chunks in the final set.
    if is_definitional and len(reranked) > 2:
        from collections import Counter as _Counter
        _sec_re2 = re.compile(r'\bsection\s+(\d+\w*)\b', re.IGNORECASE)

        def _dominant_section(chunk_text: str):
            hits = _sec_re2.findall(chunk_text)
            if not hits:
                return None
            return _Counter(hits).most_common(1)[0][0].lower()

        sec_count: dict = {}
        diverse: list = []
        for chunk in reranked:
            text = chunk.get('text', chunk.get('content', ''))
            dom = _dominant_section(text)
            if dom is None:
                diverse.append(chunk)
            elif sec_count.get(dom, 0) < 2:
                diverse.append(chunk)
                sec_count[dom] = sec_count.get(dom, 0) + 1

        reranked = diverse if diverse else reranked

    return reranked


def retrieve_for_summarization(context: str, top_k: int = None) -> list:
    """
    Summarization retrieval: extracts key legal phrases from the input
    context and uses them as the FAISS query.
    """
    if top_k is None:
        top_k = TOP_K

    embedder = load_embedder()
    index, metadata = load_faiss_index()

    key_phrases = _extract_key_phrases(context, top_n=10)
    first_sentence = _split_sentences(context)
    anchor = first_sentence[0] if first_sentence else context[:120]
    query_str = f"{anchor} {key_phrases}"

    q_vec = embedder.encode(
        [query_str], convert_to_numpy=True, normalize_embeddings=True
    )

    fetch_k = min(top_k * 3, index.ntotal)
    distances, indices = index.search(q_vec.astype(np.float32), fetch_k)

    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        chunk = dict(metadata[idx])
        chunk['faiss_score'] = float(dist)
        candidates.append(chunk)

    return _lexical_rerank_for_summary(context, candidates, top_n=RERANK_TOP_N)


def _lexical_rerank(query: str, candidates: list, top_n: int = 3) -> list:
    """
    QA reranking: overlap with question tokens.

    For broad/definitional queries (e.g. 'What is the IPC?') we use a
    TOPIC-KEYWORD scoring mode instead of plain Jaccard overlap.
    Plain Jaccard fails here because §354/§509 chunks share tokens like
    'indian', 'penal', 'code' with the query and score equally high.
    Topic-keyword mode measures how much of the query's subject vocabulary
    (after stripping question words) appears in the chunk — chunks about
    the actual subject score higher than chunks that merely mention the Act name.
    """
    if not candidates:
        return []

    q_lower = query.strip().lower()
    _definitional_starters = (
        "what is", "what are", "define", "definition of",
        "explain the", "describe the", "tell me about",
        "what does", "what do", "meaning of", "overview of",
    )
    is_definitional = any(q_lower.startswith(s) for s in _definitional_starters)

    if is_definitional:
        # ── Topic-keyword reranking for broad queries ─────────────────────────
        # Strip question words to get only the subject words
        _q_sw = {
            'what', 'is', 'are', 'the', 'a', 'an', 'of', 'in', 'for', 'under',
            'does', 'do', 'define', 'definition', 'explain', 'describe', 'tell',
            'me', 'about', 'meaning', 'overview', 'how', 'why', 'when', 'where',
            'which', 'who', 'please', 'can', 'you', 'give', 'provide',
        }
        topic_kws = set(re.findall(r'\b[a-z]{2,}\b', q_lower)) - _q_sw

        # Expand abbreviations
        _abbr = {
            'ipc':  {'ipc', 'indian', 'penal', 'code'},
            'crpc': {'crpc', 'criminal', 'procedure', 'code'},
            'cpc':  {'cpc', 'civil', 'procedure', 'code'},
            'iea':  {'evidence', 'act'},
        }
        for abbr, exp in _abbr.items():
            if abbr in topic_kws:
                topic_kws |= exp

        # Penalise chunks that are clearly about a specific sub-section
        # (e.g. contain "section 354" or "section 509") when not asked for them
        _section_re = re.compile(r'\bsection\s+\d+\w*\b', re.IGNORECASE)
        asked_section = re.search(r'\bsection\s+(\d+\w*)\b', query, re.IGNORECASE)
        asked_sec_num = asked_section.group(1).lower() if asked_section else None

        for chunk in candidates:
            text = chunk.get('text', chunk.get('content', ''))
            cl   = text.lower()

            # Topic coverage: what fraction of subject keywords appear?
            kw_hits  = sum(1 for kw in topic_kws if kw in cl)
            kw_score = kw_hits / len(topic_kws) if topic_kws else 1.0

            # Penalise if chunk is dominated by a specific section not asked about
            sec_mentions = _section_re.findall(text)
            if sec_mentions and not asked_sec_num:
                # Raised penalty: 0.12 per mention, cap 0.60 (was 0.08/0.40 — too weak)
                penalty = min(0.60, len(sec_mentions) * 0.12)
            else:
                penalty = 0.0

            chunk['lex_score']     = max(0.0, kw_score - penalty)
            chunk['combined_score'] = (
                # Flip weight: lex (topic relevance) 0.7 > FAISS cosine 0.3
                # Old was 0.4/0.6 — FAISS dominated and pulled in wrong chunks
                chunk.get('faiss_score', 0.0) * 0.3 +
                chunk['lex_score'] * 0.7
            )
    else:
        # ── Standard Jaccard overlap for specific queries ──────────────────────
        query_tokens = _tokenize_set(query)
        for chunk in candidates:
            text = chunk.get('text', chunk.get('content', ''))
            text_tokens = _tokenize_set(text)
            if not query_tokens or not text_tokens:
                chunk['lex_score'] = 0.0
            else:
                overlap = len(query_tokens & text_tokens)
                union   = len(query_tokens | text_tokens)
                chunk['lex_score'] = overlap / union
            chunk['combined_score'] = (
                chunk.get('faiss_score', 0.0) * 0.6 +
                chunk['lex_score'] * 0.4
            )

    return sorted(candidates, key=lambda x: x['combined_score'], reverse=True)[:top_n]


def _lexical_rerank_for_summary(
    context: str, candidates: list, top_n: int = 3
) -> list:
    """
    Summarization reranking: overlap with the full input context.
    Uses first 200 tokens of context to avoid dominance by long inputs.
    """
    context_tokens = _tokenize_set(context[:800])
    if not context_tokens:
        return candidates[:top_n]
    for chunk in candidates:
        text = chunk.get('text', chunk.get('content', ''))
        text_tokens = _tokenize_set(text)
        if not text_tokens:
            chunk['lex_score'] = 0.0
        else:
            overlap = len(context_tokens & text_tokens)
            union = len(context_tokens | text_tokens)
            chunk['lex_score'] = overlap / union
        chunk['combined_score'] = (
            chunk.get('faiss_score', 0.0) * 0.5 +
            chunk['lex_score'] * 0.5
        )
    return sorted(candidates, key=lambda x: x['combined_score'], reverse=True)[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# QA — Extractive answer (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def extractive_answer(question: str, chunks: list) -> tuple:
    """
    Find top-N sentences from retrieved chunks matching the question.
    Returns (answer_string, evidence_text).
    """
    top_n = EXTRACTIVE_TOP_N
    question_tokens = _tokenize_set(question)
    evidence_text = ""
    scored_sentences = []

    for chunk in chunks:
        text = chunk.get('text', chunk.get('content', ''))
        evidence_text += text + " "
        for sent in _split_sentences(text):
            sent_tokens = _tokenize_set(sent)
            if not sent_tokens:
                continue
            overlap = len(question_tokens & sent_tokens)
            scored_sentences.append((overlap, sent))

    if not scored_sentences:
        return "Answer not found in legal documents.", evidence_text.strip()

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top_scored = [(sc, s) for sc, s in scored_sentences if sc > 0]

    if not top_scored:
        text = chunks[0].get('text', chunks[0].get('content', ''))
        fallback = _split_sentences(text)
        answer = ' '.join(fallback[:2]) if fallback else "Answer not found."
        return answer, evidence_text.strip()

    seen, selected = set(), []
    for _, sent in top_scored:
        if sent not in seen:
            seen.add(sent)
            selected.append(sent)
        if len(selected) >= top_n:
            break

    return ' '.join(selected), evidence_text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Summarization — TF-IDF extractive summary (NEW / IMPROVED)
# ─────────────────────────────────────────────────────────────────────────────

def extractive_summary(
    context: str,
    chunks: list,
    n_sentences: int = None
) -> tuple:
    """
    Build an extractive summary from retrieved chunks using TF-IDF scoring.
    """
    if n_sentences is None:
        n_sentences = SUM_SENTENCES

    evidence_text = ""
    all_sentences = []
    sentence_source = {}

    for ci, chunk in enumerate(chunks):
        text = chunk.get('text', chunk.get('content', ''))
        evidence_text += text + " "
        for sent in _split_sentences(text):
            if sent not in sentence_source:
                sentence_source[sent] = (ci, len(all_sentences))
                all_sentences.append(sent)

    if not all_sentences:
        text = chunks[0].get('text', chunks[0].get('content', '')) if chunks else ""
        return text[:300], evidence_text.strip()

    scored = _tfidf_score_sentences(all_sentences)

    legal_scored = [(sc, s) for sc, s in scored if _has_legal_signal(s)]

    seen, selected = set(), []

    legal_slots = max(1, n_sentences // 2 + 1)
    for sc, sent in legal_scored:
        if sent not in seen:
            seen.add(sent)
            selected.append(sent)
        if len(selected) >= legal_slots:
            break

    for sc, sent in scored:
        if len(selected) >= n_sentences:
            break
        if sent not in seen:
            seen.add(sent)
            selected.append(sent)

    def get_position(sent):
        if sent in sentence_source:
            ci, pos = sentence_source[sent]
            return ci * 10000 + pos
        return 99999

    selected_ordered = sorted(selected, key=get_position)
    summary = ' '.join(selected_ordered)
    return summary, evidence_text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Generative fallback (unchanged, used only in hybrid/generative mode)
# ─────────────────────────────────────────────────────────────────────────────

def _build_strict_prompt(
    question: str,
    chunks: list,
    task: str = "qa"
) -> str:
    evidence = "\n\n".join(
        chunk.get('text', chunk.get('content', ''))
        for chunk in chunks
    ).strip()[:1200]

    if task == "qa":
        return (
            "<|system|>You are a precise legal assistant. "
            "Answer strictly using the Evidence. Copy exact words. "
            "One or two sentences only. "
            "If the answer is not in the evidence, say: "
            "'This information is not in the provided legal text.'<|endoftext|>\n"
            f"<|user|>Evidence:\n{evidence}\n\n"
            f"Question: {question}\n"
            "Answer (copy exact legal text):<|endoftext|>\n"
            "<|assistant|>"
        )
    else:
        return (
            "<|system|>You are a precise legal assistant. "
            "Summarise using ONLY the Evidence below. "
            "Use exact legal terms from the text. "
            "Three sentences, no additions.<|endoftext|>\n"
            f"<|user|>Evidence:\n{evidence}\n\n"
            "Summary (use exact legal terminology):<|endoftext|>\n"
            "<|assistant|>"
        )


def _generate(prompt: str, max_new_tokens: int = 150) -> str:
    import torch  # lazy import — torch is not imported at module level
    tokenizer, model = load_generator()
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_ids = out[0][inputs['input_ids'].shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    for stop in ['<|', '\n\n', 'Question:', 'Evidence:']:
        if stop in text:
            text = text[:text.index(stop)].strip()
    return text or "Answer not found in legal documents."


# ─────────────────────────────────────────────────────────────────────────────
# Evidence helpers for UI
# ─────────────────────────────────────────────────────────────────────────────

def _build_evidence_items(chunks: list) -> list:
    evidence_items = []
    for rank, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", chunk.get("content", ""))
        meta = {
            k: chunk.get(k)
            for k in [
                "doc_id",
                "parent_doc_id",
                "chunk_id",
                "chunk_index",
                "case_name",
                "court",
                "year",
                "source",
                "task",
            ]
            if chunk.get(k) is not None
        }

        evidence_items.append(
            {
                "rank": rank,
                "text": text[:1200],
                "score": round(float(chunk.get("combined_score", chunk.get("faiss_score", 0.0))), 4),
                "faiss_score": round(float(chunk.get("faiss_score", 0.0)), 4),
                "lex_score": round(float(chunk.get("lex_score", 0.0)), 4),
                "metadata": meta,
            }
        )
    return evidence_items


def _doc_chunks_to_faiss_like(doc_chunks: List[Dict]) -> list:
    """
    Convert app.py-style doc_chunks (list of {'text':..., 'source':...})
    into the same dict shape that FAISS retrieval returns, so all downstream
    helpers (_build_evidence_items, extractive_answer, etc.) work unchanged.
    """
    converted = []
    for i, dc in enumerate(doc_chunks):
        text = dc.get("text", "")
        source = dc.get("source", "uploaded_document")
        converted.append({
            "text": text,
            "content": text,
            "source": source,
            "doc_id": f"upload_{i}",
            "chunk_index": i,
            "faiss_score": 1.0,   # treat uploaded chunks as perfect match
            "lex_score": 1.0,
            "combined_score": 1.0,
        })
    return converted


# ─────────────────────────────────────────────────────────────────────────────
# Groundedness metric
# ─────────────────────────────────────────────────────────────────────────────

def groundedness_score(answer: str, evidence: str) -> float:
    """
    Semantic groundedness: cosine similarity between answer sentences
    and the evidence pool using the already-loaded sentence embedder.
    Falls back to token overlap if embedder unavailable.

    Why semantic? Groq paraphrases fluently — token overlap with
    extractive FAISS chunks is near zero even when the answer is
    perfectly grounded. Semantic similarity correctly captures this.
    """
    if not answer or not evidence:
        return 0.0

    try:
        embedder = load_embedder()

        # Split answer into sentences; score each against full evidence block
        ans_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if len(s.strip()) > 15]
        if not ans_sents:
            ans_sents = [answer[:500]]

        # Encode answer sentences and the evidence (truncated to 5000 chars to keep it fast)
        evidence_trunc = evidence[:5000]
        all_texts = ans_sents + [evidence_trunc]
        vecs = embedder.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True)

        ans_vecs = vecs[:len(ans_sents)]
        ev_vec   = vecs[-1]

        # Cosine similarity for each sentence (dot product of unit vectors)
        sims = [float(np.dot(av, ev_vec)) for av in ans_vecs]

        # Mean similarity across all answer sentences
        score = float(np.mean(sims))
        # Clamp to [0, 1]
        return round(max(0.0, min(1.0, score)), 4)

    except Exception:
        # Fallback: token overlap (original method)
        answer_tokens = [
            t for t in re.findall(r'\b\w+\b', answer.lower())
            if t not in STOPWORDS and len(t) > 2
        ]
        if not answer_tokens:
            return 0.0
        evidence_tokens = set(re.findall(r'\b\w+\b', evidence.lower()))
        grounded = sum(1 for t in answer_tokens if t in evidence_tokens)
        return round(grounded / len(answer_tokens), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def answer_query(
    question: str,
    context: str = "",
    task: str = "qa",
    mode: str = None,
    top_k: int = None
) -> dict:
    """
    Full RAG pipeline for a single query.
    Returns a structured dict so the UI can show answer + evidence + warning.
    """
    if mode is None:
        mode = RAG_MODE
    if top_k is None:
        top_k = TOP_K

    # ── 1. Retrieve ──────────────────────────────────────────────────────────
    if task == "summarization":
        query_context = context if context else question
        chunks = retrieve_for_summarization(query_context, top_k=top_k)
    else:
        chunks = retrieve(question, context=context, top_k=top_k)

    if not chunks:
        return {
            "question": question,
            "context": context,
            "answer": "No relevant legal documents found.",
            "prediction": "No relevant legal documents found.",
            "evidence": [],
            "evidence_text": "",
            "groundedness": 0.0,
            "confidence": 0.0,
            "unsupported_warning": True,
            "warning": "No relevant legal documents found.",
            "mode": mode,
            "task": task,
            "retrieved_chunks": [],
        }

    evidence_text = " ".join(c.get('text', c.get('content', '')) for c in chunks)
    evidence_items = _build_evidence_items(chunks)

    # ── 2. Generate answer ───────────────────────────────────────────────────
    if mode == "extractive":
        if task == "qa":
            answer, evidence_text = extractive_answer(question, chunks)
        else:
            query_context = context if context else question
            answer, evidence_text = extractive_summary(query_context, chunks)

    elif mode == "generative":
        prompt = _build_strict_prompt(question, chunks, task=task)
        answer = _generate(prompt)

    else:  # hybrid
        if task == "qa":
            answer, evidence_text = extractive_answer(question, chunks)
        else:
            query_context = context if context else question
            answer, evidence_text = extractive_summary(query_context, chunks)

        if len(answer.split()) < 8 or answer.lower().startswith("answer not found"):
            prompt = _build_strict_prompt(question, chunks, task=task)
            answer = _generate(prompt)

    # ── 3. Groundedness ──────────────────────────────────────────────────────
    # FIX Step 5: pass FULL evidence_text — no [:800] truncation
    gs = groundedness_score(answer, evidence_text)
    unsupported = gs < GROUNDEDNESS_THRESHOLD

    return {
        "question": question,
        "context": context,
        "answer": answer,
        "prediction": answer,
        "evidence": evidence_items,
        # FIX Step 5: store full evidence_text in return dict too
        "evidence_text": evidence_text,
        "groundedness": gs,
        "confidence": gs,
        "unsupported_warning": unsupported,
        "warning": "Weakly grounded answer: evidence overlap is low." if unsupported else "",
        "mode": mode,
        "task": task,
        "retrieved_chunks": chunks,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PATCH — fixed UI-facing alias + doc_chunks primary evidence (Step 2 Fix)
# ─────────────────────────────────────────────────────────────────────────────

def answer_question_rag(
    question: str,
    doc_text: str = "",
    context: str = "",
    task: str = "qa",
    mode: str = None,
    top_k: int = None,
    doc_chunks: Optional[List[Dict]] = None,   # ← NEW: Step 2
) -> dict:
    """
    UI-facing wrapper for the RAG pipeline.

    Step 2 change:
      When doc_chunks (list of {'text': ..., 'source': ...}) are provided,
      they are used as the PRIMARY evidence and FAISS retrieval is SKIPPED.
      This grounds every answer in the actual uploaded PDF.
      FAISS is only used when doc_chunks is None/empty (general knowledge
      query with no uploaded document).

    Accepts both 'doc_text' (from app.py) and 'context' for backwards compat.
    """
    # merge doc_text + context (doc_text takes priority)
    combined_context = (doc_text or context or "").strip()

    # ── Step 2: Decide evidence source ────────────────────────────────────────
    if doc_chunks:
        # PRIMARY PATH — uploaded PDF chunks take priority over FAISS
        logger.info(
            f"answer_question_rag: using {len(doc_chunks)} uploaded doc_chunks "
            "as primary evidence (FAISS skipped)"
        )
        chunks = _doc_chunks_to_faiss_like(doc_chunks)
    else:
        # FALLBACK PATH — no uploaded document, use FAISS corpus
        # Only attempt FAISS if the index file actually exists.
        # Loading the embedder (SentenceTransformer) when the index is missing
        # takes 10-15s and kills the Streamlit WebSocket connection.
        _faiss_exists = Path(str(FAISS_INDEX_PATH)).exists()
        chunks = []
        if _faiss_exists:
            try:
                if task == "summarization":
                    query_context = combined_context if combined_context else question
                    chunks = retrieve_for_summarization(query_context, top_k=top_k or TOP_K)
                else:
                    chunks = retrieve(question, context=combined_context, top_k=top_k or TOP_K)
            except Exception as exc:
                logger.warning(f"FAISS retrieval failed: {exc}")
                chunks = []

    evidence_items = _build_evidence_items(chunks) if chunks else []
    evidence_text = " ".join(
        c.get("text", c.get("content", "")) for c in chunks
    )

    # ── Groq generation (fast, grounded in evidence) ──────────────────────────
    try:
        from models.groq_client import groq_answer, groq_summarize

        # ── SECONDARY TOPIC FILTER in answer_question_rag ────────────────────
        # Even after retrieve()'s filtering, do a final check here before
        # sending to Groq. With RERANK_TOP_N=3 (config.py), sometimes all 3
        # returned chunks are about a specific sub-section. This catches that.

        if chunks and not doc_chunks:
            _q_sw2 = {
                'what', 'is', 'are', 'the', 'a', 'an', 'of', 'in', 'for',
                'under', 'does', 'do', 'define', 'definition', 'explain',
                'describe', 'tell', 'me', 'about', 'meaning', 'overview',
                'how', 'why', 'when', 'where', 'which', 'who', 'please',
                'can', 'you', 'give', 'provide',
            }
            q_lower2      = question.lower()
            topic_kws2    = set(re.findall(r'\b[a-z]{2,}\b', q_lower2)) - _q_sw2
            _abbr2 = {
                'ipc':  {'ipc', 'indian', 'penal', 'code'},
                'crpc': {'crpc', 'criminal', 'procedure', 'code'},
                'cpc':  {'cpc', 'civil', 'procedure', 'code'},
            }
            for a, e in _abbr2.items():
                if a in topic_kws2:
                    topic_kws2 |= e

            # Detect if question is broad (no specific section number asked)
            _is_broad = (
                not re.search(r'\bsection\s+\d+\b', question, re.IGNORECASE)
                and not re.search(r'\barticle\s+\d+\b', question, re.IGNORECASE)
            )
            _is_definitional2 = any(
                q_lower2.strip().startswith(s) for s in (
                    "what is", "what are", "define", "definition of",
                    "explain the", "describe the", "tell me about",
                    "what does", "what do", "meaning of", "overview of",
                )
            )

            if topic_kws2 and _is_broad and _is_definitional2:
                # For broad queries: score chunks and drop dominated-by-specific-section ones
                _sec_re2 = re.compile(r'\bsection\s+\d+\w*\b', re.IGNORECASE)

                def _chunk_relevance(ch):
                    txt = ch.get('text', ch.get('content', ''))
                    cl  = txt.lower()
                    kw_hits  = sum(1 for kw in topic_kws2 if kw in cl)
                    kw_score = kw_hits / len(topic_kws2)
                    sec_hits = len(_sec_re2.findall(txt))
                    # Raised to match retrieve() + groq_client: 0.12/cap 0.60
                    # (was 0.1/0.5 — inconsistently weak vs the other two layers)
                    penalty  = min(0.60, sec_hits * 0.12)
                    return max(0.0, kw_score - penalty)

                scored2   = [(c, _chunk_relevance(c)) for c in chunks]
                top_score = max(s for _, s in scored2)

                if top_score > 0:
                    # Keep chunks within 35% of top score (raised from 40% gap).
                    # 0.6 threshold was too loose — a chunk scoring 0.54 when top
                    # is 0.9 still passed, but 0.54 can still be an off-topic chunk.
                    relevant2 = [(c, s) for c, s in scored2 if s >= top_score * 0.65]
                    relevant2.sort(key=lambda x: x[1], reverse=True)
                    # For broad queries give Groq up to 5 chunks for richer answers
                    chunks = [c for c, _ in relevant2[:5]]
                else:
                    # All chunks scored 0 → clear evidence, use general knowledge
                    logger.info(
                        f"answer_question_rag: secondary filter cleared all chunks "
                        f"for '{question[:60]}' — Groq will use general knowledge"
                    )
                    chunks = []

                evidence_items = _build_evidence_items(chunks)
                evidence_text  = " ".join(
                    c.get("text", c.get("content", "")) for c in chunks
                )

        # Build context string to pass to Groq.
        # IMPORTANT: Only prepend combined_context when it is actual document text
        # (i.e. the user uploaded a PDF and doc_chunks were provided). When
        # doc_chunks is empty, combined_context comes from app.py as an empty
        # string or as the query question itself — do NOT inject it as evidence.
        # Injecting the question as context was reinforcing query-topic bias.
        groq_context = evidence_text.strip()
        if doc_chunks and combined_context and combined_context not in groq_context:
            # Only merge when we actually have a real user-uploaded document
            groq_context = combined_context[:1500] + "\n\n" + groq_context

        if task == "summarization":
            answer = groq_summarize(
                text=combined_context or question,
                mode="rag",
                evidence=evidence_text,
            )
        else:
            answer = groq_answer(
                query=question,
                context=groq_context,
                mode="rag",
            )
    except Exception as exc:
        logger.warning(f"Groq generation failed, falling back to extractive: {exc}")
        # Fallback: use original extractive pipeline
        if chunks:
            if task == "summarization":
                answer, _ = extractive_summary(
                    combined_context or question, chunks
                )
            else:
                answer, _ = extractive_answer(question, chunks)
        else:
            answer = f"[RAG generation error: {exc}]"

    # ── Groundedness ───────────────────────────────────────────────────────────
    # Score against the full context Groq actually saw (evidence + doc text),
    # not just the retrieved chunk text, so general legal facts in the answer
    # that come from the uploaded document are counted as grounded.
    scoring_text = evidence_text
    if combined_context and combined_context not in scoring_text:
        scoring_text = combined_context + " " + scoring_text
    gs = groundedness_score(answer, scoring_text) if scoring_text else 0.0

    return {
        "answer": answer,
        "prediction": answer,
        "evidence": evidence_items,
        # FIX Step 5: store full evidence_text — no [:800] truncation
        "evidence_text": evidence_text,
        "groundedness": gs,
        "confidence": gs,
        "unsupported_warning": gs < GROUNDEDNESS_THRESHOLD,
        "warning": (
            "Weakly grounded answer." if gs < GROUNDEDNESS_THRESHOLD else ""
        ),
        "task": task,
        "mode": mode or RAG_MODE,
    }


def summarize_rag(text: str) -> str:
    """
    Public summarization entry used by app.py:
        from models.rag_pipeline import summarize_rag
    """
    result = answer_question_rag(
        question="Summarize this legal document",
        doc_text=text,
        task="summarization",
    )
    if isinstance(result, dict):
        return result.get("answer", "")
    return str(result)


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def _detect_task(sample: dict) -> str:
    """Auto-detect task from sample fields."""
    if 'question' in sample and sample.get('question'):
        return 'qa'
    if 'summary' in sample:
        return 'summarization'
    if 'context' in sample and 'question' not in sample:
        return 'summarization'
    return 'qa'


def run_batch(
    input_path: str,
    output_path: str,
    task: str = None,
    mode: str = None,
    top_k: int = None,
    limit: int = None
) -> list:
    """
    Run RAG in batch mode on a JSONL input file.
    """
    if mode is None:
        mode = RAG_MODE

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if limit:
        samples = samples[:limit]

    if task is None and samples:
        task = _detect_task(samples[0])
        logger.info(f"Auto-detected task: {task}")

    logger.info(
        f"Batch RAG — {len(samples)} samples | task={task} | mode={mode}"
    )

    results = []
    for i, sample in enumerate(samples):
        try:
            if task == "qa":
                question = sample.get('question', '')
                context = sample.get('context', '')
                reference = sample.get('answer', sample.get('reference', ''))
            else:
                context = sample.get('context', sample.get('text', ''))
                question = ""
                reference = sample.get('summary', sample.get('reference', ''))

            result = answer_query(
                question=question,
                context=context,
                task=task,
                mode=mode,
                top_k=top_k
            )

            results.append({
                "task": task,
                "question": question if task == "qa" else "",
                "context": context,
                "reference": reference,
                "prediction": result["answer"],
                "answer": result["answer"],
                "evidence": result["evidence"],
                "evidence_text": result["evidence_text"],
                "groundedness": result["groundedness"],
                "confidence": result["confidence"],
                "unsupported_warning": result["unsupported_warning"],
                "warning": result["warning"],
                "mode": mode,
            })

        except Exception as exc:
            logger.warning(f"Sample {i} failed: {exc}")
            results.append({
                "task": task,
                "question": sample.get('question', ''),
                "context": sample.get('context', ''),
                "reference": sample.get(
                    'answer', sample.get('summary', sample.get('reference', ''))
                ),
                "prediction": "Error in RAG pipeline.",
                "answer": "Error in RAG pipeline.",
                "evidence": [],
                "evidence_text": "",
                "groundedness": 0.0,
                "confidence": 0.0,
                "unsupported_warning": True,
                "warning": str(exc),
                "mode": mode,
            })

        if (i + 1) % 50 == 0:
            avg_gs = sum(r['groundedness'] for r in results) / len(results)
            logger.info(
                f"  {i + 1}/{len(samples)} | avg groundedness: {avg_gs:.4f}"
            )

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    avg_gs = (
        sum(r['groundedness'] for r in results) / len(results)
        if results else 0
    )
    logger.info(f"Saved {len(results)} predictions -> {output_path}")
    logger.info(f"Final average groundedness: {avg_gs:.4f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Interactive mode
# ─────────────────────────────────────────────────────────────────────────────

def interactive():
    print("=" * 60)
    print("  LegalAId RAG Pipeline — Interactive Mode")
    print(f"  Default mode: {RAG_MODE}")
    print("  Commands:")
    print("    'quit'                    — exit")
    print("    'mode:<extractive|hybrid>' — switch generation mode")
    print("    'task:<qa|summarization>'  — switch task")
    print("=" * 60 + "\n")

    mode = RAG_MODE
    task = "qa"

    while True:
        prompt_label = f"[{task}] Question" if task == "qa" else "[summarization] Context"
        user_input = input(f"{prompt_label}: ").strip()

        if not user_input:
            continue
        if user_input.lower() == 'quit':
            break
        if user_input.lower().startswith('mode:'):
            mode = user_input.split(':', 1)[1].strip()
            print(f"  → Mode: {mode}\n")
            continue
        if user_input.lower().startswith('task:'):
            task = user_input.split(':', 1)[1].strip()
            print(f"  → Task: {task}\n")
            continue

        result = answer_query(
            question=user_input if task == "qa" else "",
            context=user_input if task == "summarization" else "",
            task=task,
            mode=mode
        )
        label = "Summary" if task == "summarization" else "Answer"
        print(f"\n{label}       : {result['answer']}")
        print(f"Groundedness: {result['groundedness']:.4f}")
        print(f"Evidence    : {result['evidence_text'][:250]}...\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LegalAId RAG Pipeline")
    parser.add_argument(
        '--mode',
        choices=['interactive', 'batch'],
        default='interactive',
        help="Run mode"
    )
    parser.add_argument(
        '--task',
        choices=['qa', 'summarization'],
        default=None,
        help="Task (auto-detected from input fields if not set)"
    )
    parser.add_argument(
        '--rag-mode',
        choices=['extractive', 'generative', 'hybrid'],
        default=None,
        help="Answer generation strategy (overrides config.RAG_MODE)"
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help="Input JSONL for batch mode"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/rag_predictions.jsonl',
        help="Output JSONL for batch mode"
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help="Max samples to process in batch mode"
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help="Override TOP_K for this run"
    )
    args = parser.parse_args()

    if args.mode == 'interactive':
        interactive()
    else:
        if not args.input:
            print("ERROR: --input is required for batch mode")
            sys.exit(1)
        run_batch(
            input_path=args.input,
            output_path=args.output,
            task=args.task,
            mode=args.rag_mode,
            top_k=args.top_k,
            limit=args.limit
        )