from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import (
    compute_generation_metrics,
    compute_groundedness_metrics,
)

BASELINE_PATH = ROOT / "outputs" / "results" / "baseline" / "baseline_predictions.jsonl"
RAG_QA_PATH = ROOT / "outputs" / "results" / "rag_predictions.jsonl"
RAG_SUMMARY_PATH = ROOT / "outputs" / "results" / "rag_summarization_predictions.jsonl"

SAVE_PATH = ROOT / "outputs" / "evaluation_results.json"


def load_jsonl(path: Path) -> List[dict]:
    data = []
    if not path.exists():
        logger.warning(f"{path} not found")
        return data

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def evaluate_records(records: List[dict]) -> Dict[str, dict]:
    summary_pred, summary_ref, summary_evidence = [], [], []
    qa_pred, qa_ref, qa_evidence = [], [], []

    for r in records:
        task = str(r.get("task", "")).lower()

        pred = str(r.get("prediction", r.get("answer", "")))
        ref = str(r.get("reference", r.get("summary", r.get("answer", ""))))
        context = str(r.get("context", r.get("input_context", "")))

        if task in ["summarisation", "summarization"]:
            summary_pred.append(pred)
            summary_ref.append(ref)
            summary_evidence.append([context])

        elif task == "qa":
            qa_pred.append(pred)
            qa_ref.append(ref)
            qa_evidence.append([context])

    results = {}

    # ✅ summarisation
    if summary_pred:
        results["summarisation"] = compute_generation_metrics(summary_pred, summary_ref)

        if any(ev and ev[0].strip() for ev in summary_evidence):
            results["summarisation"].update(
                compute_groundedness_metrics(summary_pred, summary_evidence)
            )

    # ✅ QA
    if qa_pred:
        results["qa"] = compute_generation_metrics(qa_pred, qa_ref)
        results["qa"].update(
            compute_groundedness_metrics(qa_pred, qa_evidence)
        )

    return results


def main():
    logger.info("Running Evaluation...")

    baseline_data = load_jsonl(BASELINE_PATH)
    rag_qa_data = load_jsonl(RAG_QA_PATH)
    rag_summary_data = load_jsonl(RAG_SUMMARY_PATH)

    results = {}

    if baseline_data:
        results["baseline"] = evaluate_records(baseline_data)

    rag_results = {}

    if rag_qa_data:
        rag_results.update(evaluate_records(rag_qa_data))

    if rag_summary_data:
        rag_results.update(evaluate_records(rag_summary_data))

    if rag_results:
        results["rag"] = rag_results

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\nEvaluation complete ✓")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()