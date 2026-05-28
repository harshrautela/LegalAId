from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

# Default threshold used across the project
GROUNDEDNESS_THRESHOLD = 0.35


# ─────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────

def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_words(text: str) -> List[str]:
    text = normalize_text(text).lower()
    return re.findall(r"\b[a-zA-Z]{3,}\b", text)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction).lower() == normalize_text(reference).lower())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_words(prediction)
    ref_tokens = tokenize_words(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    common = len(pred_set & ref_set)
    if common == 0:
        return 0.0
    precision = common / len(pred_set)
    recall = common / len(ref_set)
    return 2 * precision * recall / (precision + recall + 1e-9)


def jaccard_similarity(a: str, b: str) -> float:
    sa = set(tokenize_words(a))
    sb = set(tokenize_words(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ─────────────────────────────────────────────────────────────
# Generation metrics
# ─────────────────────────────────────────────────────────────

def compute_generation_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Corpus-level generation metrics.
    Returns ROUGE-1 / ROUGE-2 / ROUGE-L (F1) and BLEU.
    """
    if not predictions or not references:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    bleu = BLEU(tokenize="13a")

    totals = defaultdict(float)
    n = min(len(predictions), len(references))

    for pred, ref in zip(predictions[:n], references[:n]):
        ref = normalize_text(ref)
        pred = normalize_text(pred)
        scores = scorer.score(ref, pred)
        totals["rouge1"] += scores["rouge1"].fmeasure
        totals["rouge2"] += scores["rouge2"].fmeasure
        totals["rougeL"] += scores["rougeL"].fmeasure

    bleu_score = bleu.corpus_score(predictions[:n], [references[:n]]).score / 100.0

    return {
        "rouge1": round(totals["rouge1"] / n, 4),
        "rouge2": round(totals["rouge2"] / n, 4),
        "rougeL": round(totals["rougeL"] / n, 4),
        "bleu": round(float(bleu_score), 4),
    }


def compute_qa_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Lightweight QA metrics:
    - exact match
    - token F1
    - ROUGE-L
    - BLEU
    """
    if not predictions or not references:
        return {"exact_match": 0.0, "f1": 0.0, "rougeL": 0.0, "bleu": 0.0}

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    bleu = BLEU(tokenize="13a")

    ems = []
    f1s = []
    rouge_ls = []

    n = min(len(predictions), len(references))
    for pred, ref in zip(predictions[:n], references[:n]):
        pred = normalize_text(pred)
        ref = normalize_text(ref)
        ems.append(exact_match(pred, ref))
        f1s.append(token_f1(pred, ref))
        rouge_ls.append(scorer.score(ref, pred)["rougeL"].fmeasure)

    bleu_score = bleu.corpus_score(predictions[:n], [references[:n]]).score / 100.0

    return {
        "exact_match": round(float(np.mean(ems)), 4),
        "f1": round(float(np.mean(f1s)), 4),
        "rougeL": round(float(np.mean(rouge_ls)), 4),
        "bleu": round(float(bleu_score), 4),
    }


# ─────────────────────────────────────────────────────────────
# Groundedness / hallucination
# ─────────────────────────────────────────────────────────────

def groundedness_score(answer: str, evidence_texts: List[str]) -> float:
    """
    Lightweight proxy groundedness score.
    Measures lexical overlap between answer and evidence.
    """
    answer_tokens = set(tokenize_words(answer))
    if not answer_tokens:
        return 0.0

    evidence_tokens = set()
    for text in evidence_texts:
        evidence_tokens.update(tokenize_words(text))

    if not evidence_tokens:
        return 0.0

    return len(answer_tokens & evidence_tokens) / len(answer_tokens)


def hallucination_proxy(answer: str, evidence_texts: List[str], threshold: float = GROUNDEDNESS_THRESHOLD) -> float:
    """
    Returns 1.0 if the answer is likely hallucinated, else 0.0.
    """
    return float(groundedness_score(answer, evidence_texts) < threshold)


def compute_groundedness_metrics(
    answers: List[str],
    evidence_sources: List[List[str]],
    threshold: float = GROUNDEDNESS_THRESHOLD,
) -> Dict[str, float]:
    """
    Returns:
    - groundedness: mean overlap score
    - hallucination_rate: fraction below threshold
    """
    if not answers:
        return {"groundedness": 0.0, "hallucination_rate": 0.0}

    scores = [
        groundedness_score(ans, ev)
        for ans, ev in zip(answers, evidence_sources)
    ]

    groundedness = float(np.mean(scores)) if scores else 0.0
    hallucination_rate = float(np.mean([1.0 if s < threshold else 0.0 for s in scores])) if scores else 0.0

    return {
        "groundedness": round(groundedness, 4),
        "hallucination_rate": round(hallucination_rate, 4),
    }


# ─────────────────────────────────────────────────────────────
# Retrieval metrics
# ─────────────────────────────────────────────────────────────

def compute_retrieval_metrics(
    retrieved_rank_lists: List[List[int]],
    relevant_rank_lists: List[List[int]],
    k_values: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, float]:
    """
    Computes retrieval metrics from ranked retrieval results.

    Inputs:
    - retrieved_rank_lists: list of retrieved ranks for each query
    - relevant_rank_lists: list of relevant ranks for each query
    """
    if not retrieved_rank_lists:
        out = {f"recall@{k}": 0.0 for k in k_values}
        out["mrr"] = 0.0
        return out

    n = len(retrieved_rank_lists)
    recalls = {k: 0 for k in k_values}
    rr_total = 0.0

    for retrieved_ranks, relevant_ranks in zip(retrieved_rank_lists, relevant_rank_lists):
        relevant_set = set(relevant_ranks)

        first_rel = None
        for r in retrieved_ranks:
            if r in relevant_set:
                first_rel = r
                break

        for k in k_values:
            if any(r <= k for r in relevant_set):
                recalls[k] += 1

        if first_rel is not None:
            rr_total += 1.0 / first_rel

    out = {
        f"recall@{k}": round(recalls[k] / n, 4)
        for k in k_values
    }
    out["mrr"] = round(rr_total / n, 4)
    return out


def compute_retrieval_metrics_from_scores(
    scores: List[List[float]],
    relevant_indices: List[List[int]],
    k_values: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, float]:
    """
    Alternative retrieval metric helper if you have scores/indices from FAISS.
    """
    if not scores:
        out = {f"recall@{k}": 0.0 for k in k_values}
        out["mrr"] = 0.0
        return out

    n = len(scores)
    recalls = {k: 0 for k in k_values}
    rr_total = 0.0

    for retrieved_indices, gold_indices in zip(scores, relevant_indices):
        gold_set = set(gold_indices)

        first_rel_pos = None
        for pos, idx in enumerate(retrieved_indices, start=1):
            if idx in gold_set:
                first_rel_pos = pos
                break

        for k in k_values:
            if any(pos <= k for pos, idx in enumerate(retrieved_indices, start=1) if idx in gold_set):
                recalls[k] += 1

        if first_rel_pos is not None:
            rr_total += 1.0 / first_rel_pos

    out = {f"recall@{k}": round(recalls[k] / n, 4) for k in k_values}
    out["mrr"] = round(rr_total / n, 4)
    return out