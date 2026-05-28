from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    BASE_MODEL_NAME,
    MAX_SEQ_LEN,
    PROCESSED_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
)

from evaluation.metrics import (
    compute_generation_metrics,
    compute_qa_metrics,
    normalize_text,
)

# ─────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_DIR = RESULTS_DIR / "baseline"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = BASELINE_DIR / "baseline.log"
SUMMARY_JSON = BASELINE_DIR / "baseline_summary.json"
SUMMARY_CSV = BASELINE_DIR / "baseline_summary.csv"

logger.add(str(LOG_PATH), rotation="5 MB")


# ─────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def save_jsonl(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records):,} rows → {path}")


# ─────────────────────────────────────────────────────────────
# Text / prompt helpers
# ─────────────────────────────────────────────────────────────
def is_seq2seq_model(model_name: str) -> bool:
    low = model_name.lower()
    return any(x in low for x in ["t5", "bart", "pegasus", "flan"])


def get_device_map():
    return "auto" if torch.cuda.is_available() else None


def clean_output(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_prompt(task: str, text: str, question: str = "") -> str:
    text = clean_output(text)
    question = clean_output(question)

    if task == "summarisation":
        return (
            "Summarize the following Indian legal judgment concisely and accurately.\n\n"
            f"{text}\n\nSummary:"
        )

    if task == "qa":
        return (
            "Answer the question using the legal context.\n"
            "If the answer is not explicitly supported, say 'Insufficient evidence'.\n\n"
            f"Context:\n{text}\n\nQuestion: {question}\nAnswer:"
        )

    return text


def truncate_for_model(text: str, max_chars: int = 9000) -> str:
    text = clean_output(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# ─────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────
def load_test_or_fallback(split_name: str) -> List[dict]:
    """
    Prefer *_test.jsonl. If not available, fall back to processed file
    and sample a small subset.
    """
    test_path = PROCESSED_DIR / f"{split_name}_test.jsonl"
    processed_path = PROCESSED_DIR / f"{split_name}_processed.jsonl"

    data = load_jsonl(test_path)
    if data:
        return data

    data = load_jsonl(processed_path)
    if not data:
        return []

    # Fallback sample for quick baseline run
    random.seed(RANDOM_SEED)
    sample_size = min(50, len(data))
    return random.sample(data, sample_size)


# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────
def load_model_and_tokenizer():
    model_name = os.getenv("LEGALAID_BASE_MODEL", BASE_MODEL_NAME)

    logger.info(f"Loading baseline model: {model_name}")
    seq2seq = is_seq2seq_model(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        if seq2seq:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=get_device_map(),
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=get_device_map(),
                low_cpu_mem_usage=True,
            )
            model.config.pad_token_id = tokenizer.pad_token_id

        model.eval()
        logger.info("Baseline model loaded successfully.")
        return tokenizer, model, seq2seq

    except Exception as e:
        logger.warning(f"Could not load {model_name}: {e}")
        fallback = "google/flan-t5-base"
        logger.info(f"Trying fallback model: {fallback}")

        tokenizer = AutoTokenizer.from_pretrained(fallback, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

        model = AutoModelForSeq2SeqLM.from_pretrained(
            fallback,
            torch_dtype=dtype,
            device_map=get_device_map(),
            low_cpu_mem_usage=True,
        )
        model.eval()
        logger.info("Fallback model loaded successfully.")
        return tokenizer, model, True


# ─────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────
def generate_text(
    tokenizer,
    model,
    seq2seq: bool,
    task: str,
    text: str,
    question: str = "",
) -> str:
    prompt = build_prompt(task=task, text=text, question=question)
    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")

    with torch.inference_mode():
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if seq2seq:
            outputs = model.generate(
                **inputs,
                max_new_tokens=160 if task == "summarisation" else 128,
                do_sample=False,
                num_beams=2,
                early_stopping=True,
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return clean_output(decoded)

        outputs = model.generate(
            **inputs,
            max_new_tokens=160 if task == "summarisation" else 128,
            do_sample=False,
            num_beams=2,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        decoded = clean_output(decoded)

        # Remove prompt echo if model repeats it
        prompt_head = clean_output(prompt[:200])
        if decoded.startswith(prompt_head):
            decoded = decoded[len(prompt_head):].strip()

        return decoded


# ─────────────────────────────────────────────────────────────
# Evaluation loops
# ─────────────────────────────────────────────────────────────
def run_summarisation_baseline(tokenizer, model, seq2seq: bool, samples: List[dict]) -> Tuple[List[dict], Dict[str, float]]:
    records = []
    preds = []
    refs = []

    for rec in samples:
        text = truncate_for_model(rec.get("text", ""))
        ref = clean_output(rec.get("summary", ""))

        pred = generate_text(tokenizer, model, seq2seq, "summarisation", text=text)
        preds.append(pred)
        refs.append(ref)

        records.append(
            {
                "task": "summarisation",
                "doc_id": rec.get("doc_id", ""),
                "reference": ref,
                "prediction": pred,
                "input": text,
                "source": rec.get("source", ""),
            }
        )

    metrics = compute_generation_metrics(preds, refs)
    metrics["task"] = "summarisation"
    metrics["model"] = "baseline"
    metrics["n"] = len(samples)
    return records, metrics


def run_qa_baseline(tokenizer, model, seq2seq: bool, samples: List[dict]) -> Tuple[List[dict], Dict[str, float]]:
    records = []
    preds = []
    refs = []

    for rec in samples:
        context = truncate_for_model(rec.get("context", ""))
        question = clean_output(rec.get("question", ""))
        ref = clean_output(rec.get("answer", ""))

        pred = generate_text(tokenizer, model, seq2seq, "qa", text=context, question=question)
        preds.append(pred)
        refs.append(ref)

        records.append(
            {
                "task": "qa",
                "doc_id": rec.get("doc_id", ""),
                "question": question,
                "context": context,
                "reference": ref,
                "prediction": pred,
                "source": rec.get("source", ""),
            }
        )

    metrics = compute_qa_metrics(preds, refs)
    metrics["task"] = "qa"
    metrics["model"] = "baseline"
    metrics["n"] = len(samples)
    return records, metrics


# ─────────────────────────────────────────────────────────────
# PATCH — append this block to the BOTTOM of models/baseline.py
# (or replace the existing answer_question / summarize stubs if they exist)
#
# These two functions are what app.py imports:
#     from models.baseline import answer_question
#     from models.baseline import summarize
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# Groq-powered public interface  (used by ui/app.py)
# ─────────────────────────────────────────────────────────────

def answer_question(
    query: str,
    context: str = "",
    model=None,          # kept for API compat — not used with Groq
) -> str:
    """
    Baseline answer: minimal prompt, no Indian-law specialisation.
    Simulates a raw (untuned) model for comparison purposes.
    """
    try:
        from models.groq_client import groq_answer
        return groq_answer(query, context=context, mode="baseline")
    except Exception as exc:
        return f"[Baseline Groq error: {exc}]"


def summarize(text: str, model=None) -> str:
    """Baseline summarization via Groq (minimal prompt)."""
    try:
        from models.groq_client import groq_summarize
        return groq_summarize(text, mode="baseline")
    except Exception as exc:
        return f"[Baseline summarize error: {exc}]"


# allow direct run: python models/baseline.py
if __name__ == "__main__":
    ans = answer_question("What is Article 21 of the Indian Constitution?")
    print("Answer:", ans)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    logger.info("═" * 70)
    logger.info(" LegalAId – Baseline Run ")
    logger.info("═" * 70)

    sum_test = load_test_or_fallback("summarisation")
    qa_test = load_test_or_fallback("qa")

    if not sum_test:
        logger.warning("No summarisation data found.")
    if not qa_test:
        logger.warning("No QA data found.")

    logger.info(f"Summarisation samples: {len(sum_test)}")
    logger.info(f"QA samples: {len(qa_test)}")

    tokenizer, model, seq2seq = load_model_and_tokenizer()

    all_records = []
    summary_rows = []

    if sum_test:
        sum_records, sum_metrics = run_summarisation_baseline(tokenizer, model, seq2seq, sum_test)
        all_records.extend(sum_records)
        summary_rows.append(sum_metrics)
        save_jsonl(sum_records, BASELINE_DIR / "summarisation_baseline.jsonl")
        logger.info(f"[summarisation] {sum_metrics}")

    if qa_test:
        qa_records, qa_metrics = run_qa_baseline(tokenizer, model, seq2seq, qa_test)
        all_records.extend(qa_records)
        summary_rows.append(qa_metrics)
        save_jsonl(qa_records, BASELINE_DIR / "qa_baseline.jsonl")
        logger.info(f"[qa] {qa_metrics}")

    # Save combined outputs
    save_jsonl(all_records, BASELINE_DIR / "baseline_predictions.jsonl")

    df = pd.DataFrame(summary_rows)
    df.to_csv(SUMMARY_CSV, index=False)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved baseline summary → {SUMMARY_CSV}")
    logger.info(f"Saved baseline summary → {SUMMARY_JSON}")
    logger.info("Baseline run complete ✓")

    print("\nBaseline run complete ✓")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()