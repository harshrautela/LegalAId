"""
scripts/split_dataset.py
────────────────────────
Splits dataset into train / val / test.

Supports:
- QA dataset (context, question, answer)
- Summarization dataset (text, summary)

Usage:
    python scripts/split_dataset.py --task qa --input data/processed/synthetic_qa.jsonl
    python scripts/split_dataset.py --task summarization --input data/processed/synthetic_summarization.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError("Dataset is empty.")

    return records


def save_jsonl(records: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _clean_text(v) -> str:
    return str(v or "").strip()


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

def validate_qa(records: List[Dict]) -> List[Dict]:
    """
    Keeps rows that have enough QA signal.
    Accepts a few fallback keys in case the generator uses different names.
    Normalizes output to: context, question, answer
    """
    clean = []

    for r in records:
        context = _clean_text(r.get("context") or r.get("text") or r.get("document") or r.get("passage"))
        question = _clean_text(r.get("question") or r.get("instruction") or r.get("prompt") or r.get("query"))
        answer = _clean_text(r.get("answer") or r.get("response") or r.get("output") or r.get("summary"))

        if len(context) > 50 and len(question) > 5 and len(answer) > 5:
            clean.append({
                "doc_id": r.get("doc_id", ""),
                "context": context,
                "question": question,
                "answer": answer,
                "source": r.get("source", ""),
                "task": "qa",
            })

    return clean


def validate_summarization(records: List[Dict]) -> List[Dict]:
    """
    Keeps rows that have enough summarization signal.
    Accepts fallback keys like document/context/output/answer.
    Normalizes output to: text, summary
    """
    clean = []

    for r in records:
        text = _clean_text(r.get("text") or r.get("document") or r.get("context") or r.get("passage"))
        summary = _clean_text(r.get("summary") or r.get("output") or r.get("answer") or r.get("response"))

        if len(text) > 50 and len(summary) > 10:
            clean.append({
                "doc_id": r.get("doc_id", ""),
                "text": text,
                "summary": summary,
                "source": r.get("source", ""),
                "task": "summarization",
            })

    return clean


# ─────────────────────────────────────────────────────────────
# Split logic
# ─────────────────────────────────────────────────────────────

def split_dataset(records: List[Dict]):
    random.seed(RANDOM_SEED)
    random.shuffle(records)

    n = len(records)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]

    return train, val, test


# ─────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["qa", "summarization"])
    parser.add_argument("--input", type=str, required=True)

    args = parser.parse_args()

    input_path = Path(args.input)
    data = load_jsonl(input_path)

    print(f"Loaded {len(data)} records")

    if args.task == "qa":
        data = validate_qa(data)
    else:
        data = validate_summarization(data)

    print(f"After cleaning: {len(data)} records")

    if len(data) < 50:
        raise ValueError("Dataset too small after cleaning.")

    train, val, test = split_dataset(data)

    base_dir = input_path.parent

    save_jsonl(train, base_dir / f"{args.task}_train.jsonl")
    save_jsonl(val, base_dir / f"{args.task}_val.jsonl")
    save_jsonl(test, base_dir / f"{args.task}_test.jsonl")

    print("\n✅ Split complete:")
    print(f"Train: {len(train)}")
    print(f"Val:   {len(val)}")
    print(f"Test:  {len(test)}")


if __name__ == "__main__":
    main()