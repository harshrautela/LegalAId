"""
scripts/preprocess.py
─────────────────────
Preprocess LegalAId raw datasets into clean, model-ready JSONL files.

What this script does:
1. Loads raw summarisation, QA, and RAG corpus datasets.
2. Cleans legal text using shared utilities.
3. Removes duplicates and low-quality rows.
4. Normalizes and validates fields.
5. Chunks long RAG documents into retrieval-friendly passages only when needed.
6. Saves processed datasets to data/processed/.
7. Writes document-level train/val/test splits.
8. Writes a preprocessing summary report.

Expected raw input files:
- data/raw/summarisation_raw.jsonl
- data/raw/qa_raw.jsonl
- data/raw/rag_corpus_raw.jsonl

Output files:
- data/processed/summarisation_processed.jsonl
- data/processed/qa_processed.jsonl
- data/processed/rag_corpus_processed.jsonl
- data/processed/summarisation_train.jsonl
- data/processed/summarisation_val.jsonl
- data/processed/summarisation_test.jsonl
- data/processed/qa_train.jsonl
- data/processed/qa_val.jsonl
- data/processed/qa_test.jsonl
- data/processed/rag_train.jsonl
- data/processed/rag_val.jsonl
- data/processed/rag_test.jsonl
- data/processed/preprocess_report.json
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from loguru import logger

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import RAW_DIR, PROCESSED_DIR, RANDOM_SEED, CHUNK_SIZE, CHUNK_OVERLAP
from utils.text_utils import (
    clean_legal_text,
    deduplicate_paragraphs,
    extract_citations,
    truncate_tokens,
    count_tokens_approx,
)
from utils.chunker import LegalChunker

# ─────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)

RAW_SUM_PATH = RAW_DIR / "summarisation_raw.jsonl"
RAW_QA_PATH = RAW_DIR / "qa_raw.jsonl"
RAW_RAG_PATH = RAW_DIR / "rag_corpus_raw.jsonl"

OUT_SUM_PATH = PROCESSED_DIR / "summarisation_processed.jsonl"
OUT_QA_PATH = PROCESSED_DIR / "qa_processed.jsonl"
OUT_RAG_PATH = PROCESSED_DIR / "rag_corpus_processed.jsonl"

OUT_SUM_TRAIN = PROCESSED_DIR / "summarisation_train.jsonl"
OUT_SUM_VAL = PROCESSED_DIR / "summarisation_val.jsonl"
OUT_SUM_TEST = PROCESSED_DIR / "summarisation_test.jsonl"

OUT_QA_TRAIN = PROCESSED_DIR / "qa_train.jsonl"
OUT_QA_VAL = PROCESSED_DIR / "qa_val.jsonl"
OUT_QA_TEST = PROCESSED_DIR / "qa_test.jsonl"

OUT_RAG_TRAIN = PROCESSED_DIR / "rag_train.jsonl"
OUT_RAG_VAL = PROCESSED_DIR / "rag_val.jsonl"
OUT_RAG_TEST = PROCESSED_DIR / "rag_test.jsonl"

REPORT_PATH = PROCESSED_DIR / "preprocess_report.json"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
logger.add(str(PROCESSED_DIR / "preprocess.log"), rotation="5 MB")


# ─────────────────────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def save_jsonl(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records):,} records → {path}")


def normalize_field(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = value.replace("\u00a0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def stable_dedupe(records: List[dict], keys: List[str]) -> List[dict]:
    seen = set()
    out = []
    for rec in records:
        signature = tuple(
            " ".join(normalize_field(rec.get(k, "")).lower().split())[:1000]
            for k in keys
        )
        if signature in seen:
            continue
        seen.add(signature)
        out.append(rec)
    return out


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def add_citation_metadata(text: str) -> List[str]:
    citations = extract_citations(text)
    return citations if citations else []


def brief_preview(text: str, limit: int = 120) -> str:
    text = normalize_field(text)
    return text[:limit] + ("..." if len(text) > limit else "")


def is_chunked_rag_record(rec: dict) -> bool:
    keys = set(rec.keys())
    chunk_signals = {
        "chunk_id", "chunk_index", "parent_doc_id", "source_doc_id",
        "start_char", "end_char", "chunk_start", "chunk_end",
        "segment_id", "passage_id",
    }
    if keys.intersection(chunk_signals):
        return True

    text = normalize_field(rec.get("text", ""))
    # If raw file already looks like retrieved passages, don't chunk again.
    # This avoids exploding the corpus size.
    if len(text) <= CHUNK_SIZE + 800 and len(text) >= 100:
        return True

    return False


def split_by_group(
    records: List[dict],
    group_key_fn,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Split by group so one document (or one parent document) never leaks across splits.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6

    groups: Dict[str, List[dict]] = {}
    for rec in records:
        group_key = group_key_fn(rec)
        groups.setdefault(group_key, []).append(rec)

    group_ids = list(groups.keys())
    random.Random(RANDOM_SEED).shuffle(group_ids)

    n = len(group_ids)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    n_test = max(1, n - n_train - n_val)

    train_groups = set(group_ids[:n_train])
    val_groups = set(group_ids[n_train:n_train + n_val])
    test_groups = set(group_ids[n_train + n_val:])

    train, val, test = [], [], []
    for gid, items in groups.items():
        if gid in train_groups:
            train.extend(items)
        elif gid in val_groups:
            val.extend(items)
        else:
            test.extend(items)

    return train, val, test


def enrich_split_field(records: List[dict], split_name: str) -> List[dict]:
    out = []
    for rec in records:
        new_rec = dict(rec)
        new_rec["split"] = split_name
        out.append(new_rec)
    return out


# ─────────────────────────────────────────────────────────────
# Summarisation preprocessing
# ─────────────────────────────────────────────────────────────

def preprocess_summarisation(records: List[dict]) -> Tuple[List[dict], dict]:
    cleaned = []
    stats = Counter()

    for i, rec in enumerate(records):
        stats["raw_rows"] += 1

        text = normalize_field(rec.get("text", ""))
        summary = normalize_field(rec.get("summary", ""))

        text = clean_legal_text(text, remove_boilerplate_lines=True)
        text = deduplicate_paragraphs(text)
        summary = clean_legal_text(summary, remove_boilerplate_lines=False)

        if len(text) < 300:
            stats["dropped_too_short_text"] += 1
            continue
        if len(summary) < 30:
            stats["dropped_too_short_summary"] += 1
            continue

        text = truncate_tokens(text, max_chars=12000)
        summary = truncate_tokens(summary, max_chars=3000)

        doc_id = normalize_field(rec.get("doc_id")) or f"sum_{i:05d}"
        source = normalize_field(rec.get("source")) or "unknown"

        citations = add_citation_metadata(text + " " + summary)

        cleaned.append(
            {
                "doc_id": doc_id,
                "text": text,
                "summary": summary,
                "source": source,
                "task": "summarisation",
                "text_len": len(text),
                "summary_len": len(summary),
                "token_approx": count_tokens_approx(text),
                "citations": citations,
            }
        )
        stats["kept"] += 1

    cleaned = stable_dedupe(cleaned, keys=["text", "summary"])
    stats["after_dedup"] = len(cleaned)

    return cleaned, dict(stats)


# ─────────────────────────────────────────────────────────────
# QA preprocessing
# ─────────────────────────────────────────────────────────────

def preprocess_qa(records: List[dict]) -> Tuple[List[dict], dict]:
    cleaned = []
    stats = Counter()

    for i, rec in enumerate(records):
        stats["raw_rows"] += 1

        context = normalize_field(rec.get("context", ""))
        question = normalize_field(rec.get("question", ""))
        answer = normalize_field(rec.get("answer", ""))

        context = clean_legal_text(context, remove_boilerplate_lines=True)
        context = deduplicate_paragraphs(context)
        question = clean_legal_text(question, remove_boilerplate_lines=False)
        answer = clean_legal_text(answer, remove_boilerplate_lines=False)

        if not question:
            question = "What is the main legal issue discussed in this passage?"

        if len(context) < 250:
            stats["dropped_too_short_context"] += 1
            continue
        if len(answer) < 15:
            stats["dropped_too_short_answer"] += 1
            continue

        context = truncate_tokens(context, max_chars=12000)
        answer = truncate_tokens(answer, max_chars=3000)

        doc_id = normalize_field(rec.get("doc_id")) or f"qa_{i:05d}"
        source = normalize_field(rec.get("source")) or "unknown"

        citations = add_citation_metadata(context + " " + answer)

        cleaned.append(
            {
                "doc_id": doc_id,
                "context": context,
                "question": question,
                "answer": answer,
                "source": source,
                "task": "qa",
                "context_len": len(context),
                "question_len": len(question),
                "answer_len": len(answer),
                "token_approx": count_tokens_approx(context),
                "citations": citations,
            }
        )
        stats["kept"] += 1

    cleaned = stable_dedupe(cleaned, keys=["context", "question", "answer"])
    stats["after_dedup"] = len(cleaned)

    return cleaned, dict(stats)


# ─────────────────────────────────────────────────────────────
# RAG corpus preprocessing
# ─────────────────────────────────────────────────────────────

def preprocess_rag(records: List[dict]) -> Tuple[List[dict], dict]:
    """
    Clean the retrieval corpus.
    Important:
    - If the raw data is already chunked, we do NOT chunk again.
    - If the raw data is long documents, we chunk them once.
    """
    stats = Counter()
    cleaned_docs = []

    chunker = LegalChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, mode="para")
    already_chunked = sum(1 for r in records if is_chunked_rag_record(r)) > max(1, len(records) // 2)

    logger.info(
        "RAG preprocessing mode: {}",
        "already chunked → cleaning only" if already_chunked else "raw docs → chunking"
    )

    for i, rec in enumerate(records):
        stats["raw_rows"] += 1

        text = normalize_field(rec.get("text", ""))
        case_name = normalize_field(rec.get("case_name", "Unknown"))
        court = normalize_field(rec.get("court", "Supreme Court of India"))
        year = normalize_field(rec.get("year", "N/A"))
        source = normalize_field(rec.get("source", "unknown"))
        doc_id = normalize_field(rec.get("doc_id")) or f"corpus_{i:05d}"

        text = clean_legal_text(text, remove_boilerplate_lines=True)
        text = deduplicate_paragraphs(text)

        if len(text) < 100:
            stats["dropped_too_short_text"] += 1
            continue

        # If the raw file already contains chunks, keep them as retrieval passages.
        if already_chunked:
            chunk_text = truncate_tokens(text, max_chars=6000)
            chunk_dict = {
                "doc_id": doc_id,
                "parent_doc_id": normalize_field(rec.get("parent_doc_id")) or doc_id,
                "chunk_id": normalize_field(rec.get("chunk_id")) or doc_id,
                "chunk_index": safe_int(rec.get("chunk_index"), default=0),
                "text": chunk_text,
                "case_name": case_name,
                "court": court,
                "year": year,
                "source": source,
                "task": "rag_corpus",
                "token_approx": count_tokens_approx(chunk_text),
                "citations": add_citation_metadata(chunk_text),
            }
            cleaned_docs.append(chunk_dict)
            stats["chunks_kept"] += 1
            stats["docs_kept"] += 1
            continue

        # Otherwise chunk long documents once
        text = truncate_tokens(text, max_chars=15000)

        chunks = chunker.chunk_document(
            text=text,
            doc_id=doc_id,
            metadata={
                "case_name": case_name,
                "court": court,
                "year": year,
                "source": source,
                "task": "rag_corpus",
            },
        )

        for c in chunks:
            chunk_dict = c.to_dict()
            chunk_text = normalize_field(chunk_dict.get("text", ""))
            if len(chunk_text) < 100:
                continue

            chunk_dict["text"] = chunk_text
            chunk_dict["source"] = source
            chunk_dict["case_name"] = case_name
            chunk_dict["court"] = court
            chunk_dict["year"] = year
            chunk_dict["task"] = "rag_corpus"
            chunk_dict["token_approx"] = count_tokens_approx(chunk_text)
            chunk_dict["citations"] = add_citation_metadata(chunk_text)
            cleaned_docs.append(chunk_dict)
            stats["chunks_kept"] += 1

        stats["docs_kept"] += 1

    cleaned_docs = stable_dedupe(cleaned_docs, keys=["text", "doc_id"])
    stats["after_dedup"] = len(cleaned_docs)

    return cleaned_docs, dict(stats)


# ─────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────

def write_report(report: dict) -> None:
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved preprocessing report → {REPORT_PATH}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    logger.info("═" * 70)
    logger.info(" LegalAId – Preprocessing ")
    logger.info("═" * 70)

    if not RAW_SUM_PATH.exists():
        raise FileNotFoundError(f"Missing raw summarisation file: {RAW_SUM_PATH}")
    if not RAW_QA_PATH.exists():
        raise FileNotFoundError(f"Missing raw QA file: {RAW_QA_PATH}")
    if not RAW_RAG_PATH.exists():
        raise FileNotFoundError(f"Missing raw RAG file: {RAW_RAG_PATH}")

    raw_sum = load_jsonl(RAW_SUM_PATH)
    raw_qa = load_jsonl(RAW_QA_PATH)
    raw_rag = load_jsonl(RAW_RAG_PATH)

    logger.info(f"Loaded raw summarisation rows: {len(raw_sum):,}")
    logger.info(f"Loaded raw QA rows: {len(raw_qa):,}")
    logger.info(f"Loaded raw RAG rows: {len(raw_rag):,}")

    sum_processed, sum_stats = preprocess_summarisation(raw_sum)
    qa_processed, qa_stats = preprocess_qa(raw_qa)
    rag_processed, rag_stats = preprocess_rag(raw_rag)

    # Save full processed files
    save_jsonl(sum_processed, OUT_SUM_PATH)
    save_jsonl(qa_processed, OUT_QA_PATH)
    save_jsonl(rag_processed, OUT_RAG_PATH)

    # Split by document / parent document so there is no leakage
    sum_train, sum_val, sum_test = split_by_group(
        sum_processed,
        group_key_fn=lambda r: r.get("doc_id", ""),
    )
    qa_train, qa_val, qa_test = split_by_group(
        qa_processed,
        group_key_fn=lambda r: r.get("doc_id", ""),
    )
    rag_train, rag_val, rag_test = split_by_group(
        rag_processed,
        group_key_fn=lambda r: r.get("parent_doc_id") or r.get("doc_id", ""),
    )

    # Add split labels before saving
    save_jsonl(enrich_split_field(sum_train, "train"), OUT_SUM_TRAIN)
    save_jsonl(enrich_split_field(sum_val, "val"), OUT_SUM_VAL)
    save_jsonl(enrich_split_field(sum_test, "test"), OUT_SUM_TEST)

    save_jsonl(enrich_split_field(qa_train, "train"), OUT_QA_TRAIN)
    save_jsonl(enrich_split_field(qa_val, "val"), OUT_QA_VAL)
    save_jsonl(enrich_split_field(qa_test, "test"), OUT_QA_TEST)

    save_jsonl(enrich_split_field(rag_train, "train"), OUT_RAG_TRAIN)
    save_jsonl(enrich_split_field(rag_val, "val"), OUT_RAG_VAL)
    save_jsonl(enrich_split_field(rag_test, "test"), OUT_RAG_TEST)

    report = {
        "summarisation": {
            "input_rows": len(raw_sum),
            "output_rows": len(sum_processed),
            "split_rows": {
                "train": len(sum_train),
                "val": len(sum_val),
                "test": len(sum_test),
            },
            "stats": sum_stats,
        },
        "qa": {
            "input_rows": len(raw_qa),
            "output_rows": len(qa_processed),
            "split_rows": {
                "train": len(qa_train),
                "val": len(qa_val),
                "test": len(qa_test),
            },
            "stats": qa_stats,
        },
        "rag_corpus": {
            "input_rows": len(raw_rag),
            "output_rows": len(rag_processed),
            "split_rows": {
                "train": len(rag_train),
                "val": len(rag_val),
                "test": len(rag_test),
            },
            "stats": rag_stats,
        },
        "settings": {
            "random_seed": RANDOM_SEED,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
    }

    write_report(report)

    logger.info("\nPreprocessing summary:")
    logger.info(f"  Summarisation: {len(sum_processed):,} rows")
    logger.info(f"  QA           : {len(qa_processed):,} rows")
    logger.info(f"  RAG corpus   : {len(rag_processed):,} rows/chunks")
    logger.info("Preprocessing complete ✓")


if __name__ == "__main__":
    main()