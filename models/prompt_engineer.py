"""
models/prompt_engineer.py
──────────────────────────
Prompt engineering utilities for LegalAId.

This file provides:
1. Strong legal prompt templates
2. Task-specific prompt builders
3. A simple text generation helper
4. A small demo runner for manual testing

Run:
    python models/prompt_engineer.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────
# Project setup
# ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    BASE_MODEL_NAME,
    MAX_SEQ_LEN,
)

OUTPUT_DIR = ROOT / "outputs"
PROMPT_OUT_DIR = OUTPUT_DIR / "prompt_engineered"
PROMPT_OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are LegalAId, an assistant specialized in Indian legal documents. "
    "You must answer carefully, clearly, and only from the provided context. "
    "If the context is insufficient, say so honestly. "
    "Do not invent facts, cases, statutes, or legal outcomes."
)

# ─────────────────────────────────────────────────────────────
# Helpers
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


def load_tokenizer(model_name: str = BASE_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_name: str = BASE_MODEL_NAME):
    dtype = pick_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 220,
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

    # Remove prompt echo if present
    if prompt in decoded:
        decoded = decoded.split(prompt, 1)[-1].strip()

    return normalize_text(decoded)


# ─────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────

def build_summary_prompt(text: str) -> str:
    text = normalize_text(text)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "### Task:\n"
        "Summarize the following Indian legal judgment.\n\n"
        "### Instructions:\n"
        "- Write a concise legal summary.\n"
        "- Keep the main facts, issue, reasoning, and outcome.\n"
        "- Do not add unsupported details.\n\n"
        "### Input:\n"
        f"{text}\n\n"
        "### Response:\n"
    )


def build_qa_prompt(context: str, question: str) -> str:
    context = normalize_text(context)
    question = normalize_text(question)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "### Task:\n"
        "Answer the legal question using the provided context.\n\n"
        "### Instructions:\n"
        "- Use only the context.\n"
        "- If the context is insufficient, say you cannot answer confidently.\n"
        "- Keep the answer precise and legally grounded.\n\n"
        "### Context:\n"
        f"{context}\n\n"
        "### Question:\n"
        f"{question}\n\n"
        "### Response:\n"
    )


def build_argument_prompt(context: str, issue: str) -> str:
    context = normalize_text(context)
    issue = normalize_text(issue)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "### Task:\n"
        "Draft a short evidence-backed legal argument.\n\n"
        "### Instructions:\n"
        "- Use the provided context only.\n"
        "- Present a clear argument with legal reasoning.\n"
        "- Mention if the evidence is weak or incomplete.\n\n"
        "### Context:\n"
        f"{context}\n\n"
        "### Issue:\n"
        f"{issue}\n\n"
        "### Response:\n"
    )


def build_clause_prompt(context: str, request: str) -> str:
    context = normalize_text(context)
    request = normalize_text(request)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "### Task:\n"
        "Identify relevant clauses, sections, or legal points from the text.\n\n"
        "### Instructions:\n"
        "- Return only the most relevant items.\n"
        "- Explain briefly why they matter.\n"
        "- Do not hallucinate clauses not present in the text.\n\n"
        "### Text:\n"
        f"{context}\n\n"
        "### Request:\n"
        f"{request}\n\n"
        "### Response:\n"
    )


def build_key_issues_prompt(context: str) -> str:
    context = normalize_text(context)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "### Task:\n"
        "Identify the key legal issues in the following text.\n\n"
        "### Instructions:\n"
        "- Return 3 to 5 bullet points.\n"
        "- Be concise and grounded in the text.\n"
        "- If the text is too short, say so.\n\n"
        "### Text:\n"
        f"{context}\n\n"
        "### Response:\n"
    )


# ─────────────────────────────────────────────────────────────
# Public inference helpers
# ─────────────────────────────────────────────────────────────

def prompt_engineered_summary(text: str, model=None, tokenizer=None) -> str:
    close_model = False
    close_tokenizer = False

    if model is None:
        model = load_model(BASE_MODEL_NAME)
        close_model = True
    if tokenizer is None:
        tokenizer = load_tokenizer(BASE_MODEL_NAME)
        close_tokenizer = True

    prompt = build_summary_prompt(text)
    out = generate_text(model, tokenizer, prompt, max_new_tokens=220)

    if close_model:
        del model
    if close_tokenizer:
        del tokenizer

    return out


def prompt_engineered_qa(context: str, question: str, model=None, tokenizer=None) -> str:
    close_model = False
    close_tokenizer = False

    if model is None:
        model = load_model(BASE_MODEL_NAME)
        close_model = True
    if tokenizer is None:
        tokenizer = load_tokenizer(BASE_MODEL_NAME)
        close_tokenizer = True

    prompt = build_qa_prompt(context, question)
    out = generate_text(model, tokenizer, prompt, max_new_tokens=220)

    if close_model:
        del model
    if close_tokenizer:
        del tokenizer

    return out


def prompt_engineered_argument(context: str, issue: str, model=None, tokenizer=None) -> str:
    close_model = False
    close_tokenizer = False

    if model is None:
        model = load_model(BASE_MODEL_NAME)
        close_model = True
    if tokenizer is None:
        tokenizer = load_tokenizer(BASE_MODEL_NAME)
        close_tokenizer = True

    prompt = build_argument_prompt(context, issue)
    out = generate_text(model, tokenizer, prompt, max_new_tokens=240)

    if close_model:
        del model
    if close_tokenizer:
        del tokenizer

    return out


def prompt_engineered_clauses(context: str, request: str, model=None, tokenizer=None) -> str:
    close_model = False
    close_tokenizer = False

    if model is None:
        model = load_model(BASE_MODEL_NAME)
        close_model = True
    if tokenizer is None:
        tokenizer = load_tokenizer(BASE_MODEL_NAME)
        close_tokenizer = True

    prompt = build_clause_prompt(context, request)
    out = generate_text(model, tokenizer, prompt, max_new_tokens=220)

    if close_model:
        del model
    if close_tokenizer:
        del tokenizer

    return out


def prompt_engineered_key_issues(context: str, model=None, tokenizer=None) -> str:
    close_model = False
    close_tokenizer = False

    if model is None:
        model = load_model(BASE_MODEL_NAME)
        close_model = True
    if tokenizer is None:
        tokenizer = load_tokenizer(BASE_MODEL_NAME)
        close_tokenizer = True

    prompt = build_key_issues_prompt(context)
    out = generate_text(model, tokenizer, prompt, max_new_tokens=180)

    if close_model:
        del model
    if close_tokenizer:
        del tokenizer

    return out


# ─────────────────────────────────────────────────────────────
# Demo runner
# ─────────────────────────────────────────────────────────────

def main():
    logger.info("═" * 60)
    logger.info(" LegalAId – Prompt Engineering Demo ")
    logger.info("═" * 60)

    samples = [
        {
            "mode": "qa",
            "question": "What is Article 21?",
            "context": "Article 21 of the Indian Constitution protects the right to life and personal liberty."
        },
        {
            "mode": "qa",
            "question": "What is bail under Indian law?",
            "context": "Bail is the release of an accused person from custody pending trial, subject to conditions imposed by the court."
        },
        {
            "mode": "summary",
            "text": "The court held that the order was passed without hearing the affected party and therefore violated principles of natural justice."
        },
    ]

    tokenizer = load_tokenizer(BASE_MODEL_NAME)
    model = load_model(BASE_MODEL_NAME)

    results = []

    for sample in samples:
        if sample["mode"] == "qa":
            question = sample["question"]
            context = sample["context"]
            answer = prompt_engineered_qa(context, question, model=model, tokenizer=tokenizer)
            results.append(
                {
                    "mode": "qa",
                    "question": question,
                    "context": context,
                    "answer": answer,
                }
            )
            print("\nQ:", question)
            print("A:", answer)

        elif sample["mode"] == "summary":
            text = sample["text"]
            answer = prompt_engineered_summary(text, model=model, tokenizer=tokenizer)
            results.append(
                {
                    "mode": "summary",
                    "text": text,
                    "answer": answer,
                }
            )
            print("\nTEXT:", text)
            print("SUMMARY:", answer)

    out_path = PROMPT_OUT_DIR / "prompt_engineering_demo.json"
    save_json({"results": results}, out_path)
    logger.info(f"Saved demo results → {out_path}")


if __name__ == "__main__":
    main()