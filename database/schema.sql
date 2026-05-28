"""
models/baseline.py
──────────────────
Baseline inference for LegalAId.

This script provides:
1. Pretrained model only baseline
2. Prompt-engineered baseline
3. Fine-tuned LoRA baseline
4. Fine-tuned + RAG baseline hook

Use this file for quick manual testing and for evaluation scripts.

Run:
    python models/baseline.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

# ─────────────────────────────────────────────────────────────
# Paths / project setup
# ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    BASE_MODEL_NAME,
    CHECKPOINTS_DIR,
    FINETUNED_MODEL_DIR,
    MAX_SEQ_LEN,
    RANDOM_SEED,
)

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_OUT_DIR = OUTPUT_DIR / "baseline"
BASELINE_OUT_DIR.mkdir(parents=True, exist_ok=True)

FINETUNED_ADAPTER_DIR = Path(FINETUNED_MODEL_DIR)
DEFAULT_MODEL_NAME = BASE_MODEL_NAME

SYSTEM_PROMPT = (
    "You are LegalAId, a careful assistant for Indian legal documents. "
    "Answer only from the given text, keep responses grounded, and avoid unsupported claims."
)

# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────

def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_name: str):
    dtype = pick_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def load_finetuned_model(base_model_name: str, adapter_dir: Path):
    if PeftModel is None:
        raise RuntimeError("peft is not installed or could not be imported.")

    base_model = load_base_model(base_model_name)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Fine-tuned adapter not found: {adapter_dir}")

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    return model


def build_prompt(question: str, context: Optional[str] = None, mode: str = "qa") -> str:
    question = normalize_text(question)
    context = normalize_text(context or "")

    if mode == "summarize":
        return (
            f"{SYSTEM_PROMPT}\n\n"
            "### Instruction:\n"
            "Summarize this Indian legal judgment in a concise, formal way.\n\n"
            "### Input:\n"
            f"{context}\n\n"
            "### Response:\n"
        )

    if mode == "qa":
        return (
            f"{SYSTEM_PROMPT}\n\n"
            "### Instruction:\n"
            "Answer the legal question using only the provided context.\n\n"
            "### Input:\n"
            f"Context:\n{context}\n\nQuestion:\n{question}\n\n"
            "### Response:\n"
        )

    if mode == "argument":
        return (
            f"{SYSTEM_PROMPT}\n\n"
            "### Instruction:\n"
            "Draft a short legal argument grounded in the provided text.\n\n"
            "### Input:\n"
            f"{context}\n\nQuestion:\n{question}\n\n"
            "### Response:\n"
        )

    return (
        f"{SYSTEM_PROMPT}\n\n"
        "### Instruction:\n"
        "Answer the legal task using the provided text.\n\n"
        "### Input:\n"
        f"{context}\n\nQuestion:\n{question}\n\n"
        "### Response:\n"
    )


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Try to remove prompt echo if present
    if prompt in decoded:
        decoded = decoded.split(prompt, 1)[-1].strip()

    return normalize_text(decoded)


def maybe_save_result(data: Dict[str, Any], filename: str) -> None:
    path = BASELINE_OUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────

def run_pretrained_baseline(question: str, context: Optional[str] = None, task: str = "qa") -> Dict[str, Any]:
    tokenizer = load_tokenizer(DEFAULT_MODEL_NAME)
    model = load_base_model(DEFAULT_MODEL_NAME)

    prompt = build_prompt(question=question, context=context, mode=task)
    answer = generate_text(model, tokenizer, prompt)

    result = {
        "baseline": "pretrained_only",
        "task": task,
        "question": question,
        "context": context,
        "prompt": prompt,
        "answer": answer,
    }
    maybe_save_result(result, "pretrained_only.json")
    return result


def run_prompt_engineered_baseline(question: str, context: Optional[str] = None, task: str = "qa") -> Dict[str, Any]:
    tokenizer = load_tokenizer(DEFAULT_MODEL_NAME)
    model = load_base_model(DEFAULT_MODEL_NAME)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "### Instruction:\n"
        "Follow the instruction carefully and answer in short, clear legal language.\n"
        "If the answer is not supported by the context, say you are unsure.\n\n"
        "### Input:\n"
        f"Context:\n{normalize_text(context or '')}\n\nQuestion:\n{normalize_text(question)}\n\n"
        "### Response:\n"
    )
    answer = generate_text(model, tokenizer, prompt)

    result = {
        "baseline": "prompt_engineered",
        "task": task,
        "question": question,
        "context": context,
        "prompt": prompt,
        "answer": answer,
    }
    maybe_save_result(result, "prompt_engineered.json")
    return result


def run_finetuned_baseline(question: str, context: Optional[str] = None, task: str = "qa") -> Dict[str, Any]:
    tokenizer = load_tokenizer(DEFAULT_MODEL_NAME)
    model = load_finetuned_model(DEFAULT_MODEL_NAME, FINETUNED_ADAPTER_DIR)

    prompt = build_prompt(question=question, context=context, mode=task)
    answer = generate_text(model, tokenizer, prompt)

    result = {
        "baseline": "fine_tuned_lora",
        "task": task,
        "question": question,
        "context": context,
        "prompt": prompt,
        "answer": answer,
    }
    maybe_save_result(result, "fine_tuned_lora.json")
    return result


# ─────────────────────────────────────────────────────────────
# Manual demo
# ─────────────────────────────────────────────────────────────

def main():
    logger.info("═" * 60)
    logger.info(" LegalAId – Baseline Runner ")
    logger.info("═" * 60)

    samples = [
        {
            "task": "qa",
            "question": "What is Article 21?",
            "context": "Article 21 of the Indian Constitution protects the right to life and personal liberty."
        },
        {
            "task": "qa",
            "question": "What is bail under Indian law?",
            "context": "Bail is the release of an accused person from custody pending trial, subject to conditions imposed by the court."
        },
        {
            "task": "summarize",
            "question": "Summarize the judgment",
            "context": "The court held that the order was passed without hearing the affected party and therefore violated principles of natural justice."
        },
    ]

    results = []

    for sample in samples:
        task = sample["task"]
        question = sample["question"]
        context = sample["context"]

        logger.info(f"Running sample: {question}")

        pretrained = run_pretrained_baseline(question, context, task=task)
        prompt_based = run_prompt_engineered_baseline(question, context, task=task)
        finetuned = run_finetuned_baseline(question, context, task=task)

        results.append(
            {
                "question": question,
                "context": context,
                "pretrained_only": pretrained["answer"],
                "prompt_engineered": prompt_based["answer"],
                "fine_tuned": finetuned["answer"],
            }
        )

    maybe_save_result({"samples": results}, "baseline_comparison_demo.json")

    print("\n===== BASELINE DEMO COMPLETE =====\n")
    for item in results:
        print(f"Q: {item['question']}")
        print(f"Pretrained: {item['pretrained_only']}")
        print(f"Prompt-engineered: {item['prompt_engineered']}")
        print(f"Fine-tuned: {item['fine_tuned']}")
        print("-" * 80)


if __name__ == "__main__":
    main()