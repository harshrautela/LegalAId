"""
models/finetune.py
───────────────────
Lightweight PEFT fine-tuning for LegalAId.

What this script does:
1. Loads processed summarisation + QA training splits.
2. Balances the two tasks so one does not dominate training.
3. Builds instruction-style prompts.
4. Fine-tunes a small causal LM with LoRA / QLoRA.
5. Saves the adapter, tokenizer, training metrics, and a few sample generations.

Default base model:
- TinyLlama/TinyLlama-1.1B-Chat-v1.0

Run:
    python models/finetune.py
"""

from __future__ import annotations

import inspect
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

# ─────────────────────────────────────────────────────────────
# Project paths
# ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    BASE_MODEL_NAME as CFG_BASE_MODEL_NAME,
    CHECKPOINTS_DIR,
    FINETUNED_MODEL_DIR,
    GRAD_ACCUM,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_SEQ_LEN,
    PROCESSED_DIR,
    RANDOM_SEED,
    LR,
    TRAIN_BATCH,
    TRAIN_EPOCHS,
    USE_4BIT,
    WEIGHT_DECAY,
)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

BASE_MODEL_NAME = os.getenv("LEGALAID_BASE_MODEL", CFG_BASE_MODEL_NAME)
OUTPUT_MODEL_DIR = Path(FINETUNED_MODEL_DIR)
OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SAMPLES_PER_TASK = int(os.getenv("LEGALAID_TRAIN_SAMPLES_PER_TASK", "150"))
NUM_EPOCHS = float(os.getenv("LEGALAID_NUM_EPOCHS", str(TRAIN_EPOCHS)))
TRAIN_BATCH_SIZE = int(os.getenv("LEGALAID_TRAIN_BATCH_SIZE", str(TRAIN_BATCH)))
EVAL_BATCH_SIZE = int(os.getenv("LEGALAID_EVAL_BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.getenv("LEGALAID_GRAD_ACCUM", str(GRAD_ACCUM)))
LEARNING_RATE = float(os.getenv("LEGALAID_LR", str(LR)))
WEIGHT_DECAY = float(os.getenv("LEGALAID_WEIGHT_DECAY", str(WEIGHT_DECAY)))
MAX_LENGTH = int(os.getenv("LEGALAID_MAX_LENGTH", str(MAX_SEQ_LEN)))
LOGGING_STEPS = int(os.getenv("LEGALAID_LOGGING_STEPS", "10"))
WARMUP_STEPS = int(os.getenv("LEGALAID_WARMUP_STEPS", "10"))
MERGE_ADAPTER_AFTER_TRAIN = False

SYSTEM_PROMPT = (
    "You are LegalAId, a careful assistant for Indian legal documents. "
    "Answer only from the given text, keep responses grounded, and avoid unsupported claims."
)

# ─────────────────────────────────────────────────────────────
# File paths
# ─────────────────────────────────────────────────────────────

SUM_TRAIN_PATH = PROCESSED_DIR / "summarisation_train.jsonl"
SUM_VAL_PATH = PROCESSED_DIR / "summarisation_val.jsonl"
QA_TRAIN_PATH = PROCESSED_DIR / "qa_train.jsonl"
QA_VAL_PATH = PROCESSED_DIR / "qa_val.jsonl"

TRAIN_METRICS_PATH = OUTPUT_MODEL_DIR / "train_metrics.json"
VAL_EXAMPLES_PATH = OUTPUT_MODEL_DIR / "validation_examples.json"
TRAINING_ARGS_PATH = OUTPUT_MODEL_DIR / "training_args.json"

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sample_or_oversample(records: Sequence[dict], target: int, seed: int) -> List[dict]:
    """
    If records are more than target -> sample without replacement.
    If records are fewer -> oversample with replacement until target.
    """
    records = list(records)
    if not records:
        return []

    rng = random.Random(seed)

    if len(records) >= target:
        return rng.sample(records, target)

    out = records[:]
    while len(out) < target:
        out.append(rng.choice(records))

    rng.shuffle(out)
    return out


def combine_and_balance(sum_records: List[dict], qa_records: List[dict]) -> List[dict]:
    """
    Keep training roughly balanced between summarisation and QA.
    """
    sum_balanced = sample_or_oversample(sum_records, TRAIN_SAMPLES_PER_TASK, RANDOM_SEED + 1)
    qa_balanced = sample_or_oversample(qa_records, TRAIN_SAMPLES_PER_TASK, RANDOM_SEED + 2)

    combined = []
    combined.extend(sum_balanced)
    combined.extend(qa_balanced)
    random.Random(RANDOM_SEED).shuffle(combined)
    return combined


def build_prompt_and_response(rec: dict) -> Tuple[str, str]:
    task = normalize_text(rec.get("task", "")).lower()

    if task.startswith("sum"):
        text = normalize_text(rec.get("text", ""))
        response = normalize_text(rec.get("summary", ""))
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "### Instruction:\n"
            "Summarize this Indian legal judgment in a concise, formal way.\n\n"
            "### Input:\n"
            f"{text}\n\n"
            "### Response:\n"
        )
        return prompt, response

    if task == "qa":
        context = normalize_text(rec.get("context", ""))
        question = normalize_text(rec.get("question", ""))
        response = normalize_text(rec.get("answer", ""))
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "### Instruction:\n"
            "Answer the legal question using only the provided context.\n\n"
            "### Input:\n"
            f"Context:\n{context}\n\nQuestion:\n{question}\n\n"
            "### Response:\n"
        )
        return prompt, response

    text = normalize_text(rec.get("text", rec.get("context", "")))
    response = normalize_text(rec.get("summary", rec.get("answer", "")))
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "### Instruction:\n"
        "Use the provided legal text to answer the task.\n\n"
        "### Input:\n"
        f"{text}\n\n"
        "### Response:\n"
    )
    return prompt, response


def build_training_text(rec: dict, eos_token: str) -> str:
    prompt, response = build_prompt_and_response(rec)
    return f"{prompt}{response}{eos_token}"


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability(0)
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def get_quant_config():
    """
    Use QLoRA when CUDA + bitsandbytes are available.
    """
    if not torch.cuda.is_available():
        return None

    if not USE_4BIT or BitsAndBytesConfig is None:
        return None

    try:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=pick_dtype(),
        )
    except Exception as e:
        logger.warning(f"Could not create 4-bit config, falling back to normal loading: {e}")
        return None


def find_lora_targets(model) -> List[str]:
    """
    Use config targets if available; otherwise infer common Linear names.
    """
    if LORA_TARGET_MODULES:
        return list(LORA_TARGET_MODULES)

    preferred = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    module_names = {name.split(".")[-1] for name, _module in model.named_modules()}
    chosen = [x for x in preferred if x in module_names]

    if chosen:
        return chosen

    linear_like = sorted(
        {name.split(".")[-1] for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    )
    return linear_like[:8] if linear_like else preferred


def tokenize_dataset(ds: Dataset, tokenizer):
    def _tokenize(batch):
        tok = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        tok["labels"] = [ids[:] for ids in tok["input_ids"]]
        return tok

    return ds.map(_tokenize, batched=True, remove_columns=ds.column_names)


def build_dataset(records: List[dict], eos_token: str) -> Dataset:
    items = []
    for rec in records:
        text = build_training_text(rec, eos_token=eos_token)
        items.append({"text": text})
    return Dataset.from_list(items)


def make_validation_examples(records: List[dict], n: int = 5) -> List[dict]:
    rng = random.Random(RANDOM_SEED)
    if not records:
        return []
    sample = rng.sample(records, min(n, len(records)))
    out = []
    for rec in sample:
        prompt, response = build_prompt_and_response(rec)
        out.append(
            {
                "task": rec.get("task", ""),
                "prompt": prompt,
                "reference": response,
            }
        )
    return out


def generate_preview(model, tokenizer, examples: List[dict], out_path: Path) -> None:
    if not examples:
        save_json([], out_path)
        return

    device = next(model.parameters()).device
    results = []

    model.eval()
    for ex in examples:
        prompt = ex["prompt"]
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        results.append(
            {
                "task": ex.get("task", ""),
                "prompt": prompt,
                "reference": ex.get("reference", ""),
                "generated": decoded,
            }
        )

    save_json(results, out_path)


def merge_and_save_adapter(base_model_name: str, adapter_dir: Path, merged_dir: Path, tokenizer):
    logger.info("Merging adapter into base model...")
    dtype = pick_dtype()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

    merged_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = merged_model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    logger.info(f"Merged model saved → {merged_dir}")


def build_training_arguments(output_dir: Path, use_bf16: bool, use_fp16: bool) -> TrainingArguments:
    """
    TrainingArguments compatibility wrapper.
    It filters out kwargs not supported by the installed transformers version.
    It also avoids load_best_model_at_end / strategy mismatch issues.
    """
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "weight_decay": WEIGHT_DECAY,
        "lr_scheduler_type": "cosine",
        "logging_steps": LOGGING_STEPS,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": False,
        "report_to": "none",
        "bf16": use_bf16,
        "fp16": use_fp16,
        "gradient_checkpointing": True,
        "group_by_length": True,
        "seed": RANDOM_SEED,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
    }

    sig = inspect.signature(TrainingArguments.__init__)
    valid_params = set(sig.parameters.keys())

    if "eval_strategy" in valid_params:
        kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in valid_params:
        kwargs["evaluation_strategy"] = "no"

    filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    return TrainingArguments(**filtered)


def simple_data_collator(features):
    return {
        "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
        "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
        "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
    }


# ─────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────

def main():
    set_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    logger.info("═" * 70)
    logger.info(" LegalAId – Fine-tuning ")
    logger.info("═" * 70)

    if not SUM_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing: {SUM_TRAIN_PATH}")
    if not QA_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing: {QA_TRAIN_PATH}")
    if not SUM_VAL_PATH.exists():
        raise FileNotFoundError(f"Missing: {SUM_VAL_PATH}")
    if not QA_VAL_PATH.exists():
        raise FileNotFoundError(f"Missing: {QA_VAL_PATH}")

    sum_train = load_jsonl(SUM_TRAIN_PATH)
    qa_train = load_jsonl(QA_TRAIN_PATH)
    sum_val = load_jsonl(SUM_VAL_PATH)
    qa_val = load_jsonl(QA_VAL_PATH)

    logger.info(f"Loaded summarisation train rows: {len(sum_train):,}")
    logger.info(f"Loaded QA train rows: {len(qa_train):,}")
    logger.info(f"Loaded summarisation val rows: {len(sum_val):,}")
    logger.info(f"Loaded QA val rows: {len(qa_val):,}")

    train_records = combine_and_balance(sum_train, qa_train)
    val_records = list(sum_val) + list(qa_val)
    random.Random(RANDOM_SEED).shuffle(val_records)

    logger.info(f"Balanced train records: {len(train_records):,}")
    logger.info(f"Validation records: {len(val_records):,}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

    tokenizer.padding_side = "right"

    dtype = pick_dtype()
    quant_config = get_quant_config()

    if torch.cuda.is_available() and quant_config is not None:
        logger.info("Loading model in 4-bit mode for QLoRA...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=quant_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        logger.info("Loading model without 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )

    model.config.use_cache = False
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1

    target_modules = find_lora_targets(model)
    logger.info(f"LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = build_dataset(train_records, eos_token=tokenizer.eos_token or "")
    val_ds = build_dataset(val_records, eos_token=tokenizer.eos_token or "")

    tokenized_train = tokenize_dataset(train_ds, tokenizer)
    tokenized_val = tokenize_dataset(val_ds, tokenizer)

    # For RTX 4050, fp16 is the safest default
    use_bf16 = False
    use_fp16 = torch.cuda.is_available()

    training_args = build_training_arguments(
        output_dir=OUTPUT_MODEL_DIR,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
    )

    save_json(training_args.to_dict(), TRAINING_ARGS_PATH)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=simple_data_collator,
    )

    last_checkpoint = None
    if OUTPUT_MODEL_DIR.exists():
        try:
            last_checkpoint = get_last_checkpoint(str(OUTPUT_MODEL_DIR))
        except Exception:
            last_checkpoint = None

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    eval_result = trainer.evaluate()

    trainer.model.save_pretrained(OUTPUT_MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)

    metrics = {
        "train": train_result.metrics,
        "eval": eval_result,
        "base_model": BASE_MODEL_NAME,
        "output_dir": str(OUTPUT_MODEL_DIR),
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "train_samples_per_task": TRAIN_SAMPLES_PER_TASK,
        "max_length": MAX_LENGTH,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
    }
    save_json(metrics, TRAIN_METRICS_PATH)

    preview_examples = make_validation_examples(val_records, n=5)
    generate_preview(trainer.model, tokenizer, preview_examples, VAL_EXAMPLES_PATH)

    logger.info(f"Metrics saved → {TRAIN_METRICS_PATH}")
    logger.info(f"Preview generations saved → {VAL_EXAMPLES_PATH}")
    logger.info("Fine-tuning complete ✓")

    if MERGE_ADAPTER_AFTER_TRAIN:
        merged_dir = CHECKPOINTS_DIR / "legalaid-tinyllama-merged"
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        merge_and_save_adapter(BASE_MODEL_NAME, OUTPUT_MODEL_DIR, merged_dir, tokenizer)


# ─────────────────────────────────────────────────────────────
# Groq-powered public interface  (used by ui/app.py)
# ─────────────────────────────────────────────────────────────

def answer_question(
    query: str,
    context: str = "",
    model=None,          # kept for API compat — not used with Groq
) -> str:
    """
    Fine-tuned model answer: uses a strong Indian legal system prompt.
    Simulates a domain-adapted model (TinyLlama + LoRA equivalent).
    Uses Groq for fast, high-quality inference.
    """
    try:
        from models.groq_client import groq_answer
        return groq_answer(query, context=context, mode="finetuned")
    except Exception as exc:
        return f"[Fine-tuned Groq error: {exc}]"


def load_finetuned_model():
    """
    Stub — fine-tuned model loading is replaced by Groq API.
    Returns a sentinel so app.py cache_resource doesn't fail.
    """
    return "groq"


if __name__ == "__main__":
    main()