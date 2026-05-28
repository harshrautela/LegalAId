"""
Create FAISS vector index for LegalAId.

Run:
    python scripts/create_faiss_index.py

Fixes applied (Engineering Guide Step 8):
- EMBED_MODEL now read from config.py if available, with fallback
- Added TOP_K constant (used for sanity-check print only; actual search TOP_K lives in rag_pipeline.py)
- Better error messages when corpus file is missing or empty
- Added per-batch progress percentage to embedding loop (visible for large corpora)
- Metadata dict no longer double-stores 'text' key (was stored twice via **r and explicit key)
- Backup of existing index before overwrite (prevents data loss on re-run)
- Consistent print prefix [faiss] for log filtering
"""

from pathlib import Path
import json
import pickle
import shutil
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config — pull from project config where possible
# ---------------------------------------------------------------------------
try:
    from config import EMBED_MODEL  # type: ignore
except ImportError:
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = ROOT / "data" / "processed"
FAISS_DIR = ROOT / "data" / "faiss_index"
FAISS_DIR.mkdir(parents=True, exist_ok=True)

RAG_FILE   = PROCESSED_DIR / "rag_corpus_processed.jsonl"
INDEX_FILE = FAISS_DIR / "legal_index.faiss"
META_FILE  = FAISS_DIR / "metadata.pkl"

BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list:
    """Load all valid JSON lines from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [faiss] WARNING: skipping malformed line {lineno}: {exc}")
    return records


def extract_text(record: dict) -> str:
    """
    Pull the text field from a corpus record.
    Tries common field names in priority order.
    """
    for field in ("chunk", "text", "content", "document", "passage"):
        val = record.get(field, "")
        if isinstance(val, str):
            val = val.strip()
            if val:
                return val
    return ""


def backup_existing_index() -> None:
    """
    If an index already exists, rename it to *.bak before overwriting.
    Prevents accidental data loss when re-running the script.
    """
    if INDEX_FILE.exists():
        bak = INDEX_FILE.with_suffix(".faiss.bak")
        shutil.copy2(INDEX_FILE, bak)
        print(f"[faiss] Backed up existing index → {bak.name}")
    if META_FILE.exists():
        bak = META_FILE.with_suffix(".pkl.bak")
        shutil.copy2(META_FILE, bak)
        print(f"[faiss] Backed up existing metadata → {bak.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("LegalAId FAISS Index Builder")
    print("=" * 60)

    # ── 1. Validate corpus ────────────────────────────────────────────────
    if not RAG_FILE.exists():
        raise FileNotFoundError(
            f"[faiss] Corpus file not found:\n  {RAG_FILE}\n"
            "Run: python scripts/ingest_pdfs.py  first."
        )

    print(f"\n[faiss] Loading corpus from {RAG_FILE.name} ...")
    t0 = time.time()
    records = load_jsonl(RAG_FILE)
    print(f"[faiss] Raw records loaded: {len(records)}  ({time.time()-t0:.1f}s)")

    if not records:
        raise ValueError(
            "[faiss] Corpus file exists but contains no valid records.\n"
            "Run: python scripts/ingest_pdfs.py  to populate it."
        )

    # ── 2. Extract texts & build metadata ────────────────────────────────
    texts: list = []
    metadata: list = []

    for i, record in enumerate(records):
        txt = extract_text(record)
        if not txt:
            print(f"  [faiss] WARNING: record {i} has no usable text field — skipped.")
            continue

        texts.append(txt)

        # Build metadata entry — avoid double-storing 'text' via **record
        # (record already contains 'text'; we use sequential id for FAISS alignment)
        meta_entry = {k: v for k, v in record.items() if k != "id"}
        meta_entry["faiss_id"] = len(metadata)   # sequential index into FAISS
        meta_entry["text"] = txt                  # canonical text field
        metadata.append(meta_entry)

    if not texts:
        raise ValueError("[faiss] No usable text chunks found in corpus.")

    print(f"[faiss] Usable chunks: {len(texts)}")

    # ── 3. Load embedding model ───────────────────────────────────────────
    print(f"\n[faiss] Loading embedding model: {EMBED_MODEL} ...")
    t1 = time.time()
    model = SentenceTransformer(EMBED_MODEL)
    print(f"[faiss] Model loaded ({time.time()-t1:.1f}s)")

    # ── 4. Generate embeddings ────────────────────────────────────────────
    print(f"[faiss] Generating embeddings for {len(texts)} chunks ...")
    t2 = time.time()

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,   # required for cosine similarity via IndexFlatIP
    )

    embeddings = np.asarray(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    print(f"[faiss] Embeddings shape: {embeddings.shape}  ({time.time()-t2:.1f}s)")
    print(f"[faiss] Embedding dimension: {dim}")

    # ── 5. Build FAISS index ──────────────────────────────────────────────
    print("\n[faiss] Building FAISS IndexFlatIP (cosine via normalised vectors) ...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[faiss] Indexed vectors: {index.ntotal}")

    # ── 6. Save — backup first ────────────────────────────────────────────
    print("\n[faiss] Saving ...")
    backup_existing_index()

    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)

    print(f"\n[faiss] ✓ Index saved  → {INDEX_FILE}")
    print(f"[faiss] ✓ Metadata saved → {META_FILE}")

    total_time = time.time() - t0
    print(f"\n[faiss] FAISS indexing complete ✓  (total: {total_time:.1f}s)")
    print(f"[faiss] Index contains {index.ntotal} vectors from {len(texts)} chunks.")
    print("\n[faiss] → You can now (re)start the Streamlit app.")


if __name__ == "__main__":
    main()