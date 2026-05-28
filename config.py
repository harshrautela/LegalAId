"""
LegalAId – Central Configuration
All paths, model names, and hyperparameters live here.
Change once; everything else picks it up automatically.
"""

from pathlib import Path
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Root / base paths
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
BASE_DIR = ROOT  # alias used by some scripts / Claude output

# ── Directory layout ─────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FAISS_DIR = DATA_DIR / "faiss_index"
SAMPLES_DIR = DATA_DIR / "samples"

OUTPUT_DIR = ROOT / "outputs"
OUTPUTS_DIR = OUTPUT_DIR  # backward-friendly alias
LOGS_DIR = OUTPUT_DIR / "logs"
RESULTS_DIR = OUTPUT_DIR / "results"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"

DATABASE_DIR = ROOT / "database"

for _d in [
    RAW_DIR,
    PROCESSED_DIR,
    FAISS_DIR,
    SAMPLES_DIR,
    OUTPUT_DIR,
    LOGS_DIR,
    RESULTS_DIR,
    CHECKPOINTS_DIR,
    DATABASE_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# File path aliases (Claude-compatible)
# ─────────────────────────────────────────────────────────────────────────────

FAISS_INDEX_PATH = FAISS_DIR / "legal_index.faiss"
FAISS_METADATA_PATH = FAISS_DIR / "metadata.pkl"
RAG_CORPUS_PROCESSED = PROCESSED_DIR / "rag_corpus_processed.jsonl"
SYNTHETIC_QA_PATH = PROCESSED_DIR / "synthetic_qa.jsonl"

# Optional convenience aliases
QA_TRAIN_PATH = PROCESSED_DIR / "qa_train.jsonl"
QA_VAL_PATH = PROCESSED_DIR / "qa_val.jsonl"
QA_TEST_PATH = PROCESSED_DIR / "qa_test.jsonl"

SUMMARISATION_TRAIN_PATH = PROCESSED_DIR / "summarisation_train.jsonl"
SUMMARISATION_VAL_PATH = PROCESSED_DIR / "summarisation_val.jsonl"
SUMMARISATION_TEST_PATH = PROCESSED_DIR / "summarisation_test.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# SQLite
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = DATABASE_DIR / "legalaid.db"

# ─────────────────────────────────────────────────────────────────────────────
# Hardware / device
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Embedding / base model
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FALLBACK_MODEL = "google/flan-t5-base"

# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuned model output path
# ─────────────────────────────────────────────────────────────────────────────

FINETUNED_MODEL_DIR = CHECKPOINTS_DIR / "legalaid-tinyllama-lora"

# ─────────────────────────────────────────────────────────────────────────────
# QLoRA / LoRA hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

TRAIN_EPOCHS = 3
TRAIN_BATCH = 1
GRAD_ACCUM = 16
LR = 2e-4
MAX_SEQ_LEN = 384
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
USE_4BIT = True

# ─────────────────────────────────────────────────────────────────────────────
# RAG settings
# ─────────────────────────────────────────────────────────────────────────────

TOP_K = 8                # initial retrieval pool
RERANK_TOP_N = 3         # keep only best 3 after reranking
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

RAG_MODE = "extractive"  # "extractive" | "generative" | "hybrid"
EXTRACTIVE_TOP_N = 2     # number of sentences/chunks to keep for extractive answer
SUM_SENTENCES = 3        # sentences per extractive summary

GROUNDEDNESS_THRESHOLD = 0.50   # semantic cosine similarity scale (was 0.35 for token overlap)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset fractions
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

EVAL_BATCH_SIZE = 8
ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

APP_TITLE = "LegalAId — AI-Powered Indian Legal Assistant"
APP_SUBTITLE = "Summarise · Answer · Argue · Cite"
ACCENT_COLOR = "#1a3a5c"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

LOG_LEVEL = "INFO"

print("[config] LegalAId configuration loaded OK")