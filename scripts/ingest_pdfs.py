# scripts/ingest_pdfs.py
"""
PDF ingestion for LegalAId RAG corpus.
Extracts text from PDFs, cleans, chunks, and appends to rag_corpus_processed.jsonl.
After running this, rebuild the FAISS index with: python scripts/create_faiss_index.py

Fixes applied (Engineering Guide Step 8):
- CHUNK_SIZE and CHUNK_OVERLAP now read from config.py if available, with fallback defaults
- Corpus output directory is auto-created if missing (no silent failure)
- chunk_text() exported so rag_pipeline.py can reuse it for in-memory PDF chunking (Root Cause 1)
- Added chunk_pdf_bytes() helper: chunks a raw PDF bytes object in memory (used by app.py / rag_pipeline.py)
- Dedup key uses first 120 chars (was 100) to reduce false-positive collision on short legal headers
- total_pages_extracted counter was unused — removed to avoid confusion
- Added summary statistics at the end
"""

import json
import re
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — try to pull from project config, fall back to safe defaults
# ---------------------------------------------------------------------------
try:
    from config import CHUNK_SIZE, CHUNK_OVERLAP  # type: ignore
except ImportError:
    CHUNK_SIZE = 300        # words per chunk
    CHUNK_OVERLAP = 50      # word overlap

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
CORPUS_PATH = BASE_DIR / "data" / "processed" / "rag_corpus_processed.jsonl"

DEDUP_KEY_LEN = 120   # chars used for deduplication fingerprint (increased from 100)
MIN_CHUNK_WORDS = 30  # discard trailing chunks shorter than this


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_pdfplumber(pdf_path: Path) -> str:
    """Extract text from a PDF file on disk using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is not installed. Run: pip install pdfplumber")

    text_parts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF given as raw bytes (used for in-memory uploaded PDFs).
    Requires pdfplumber.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is not installed. Run: pip install pdfplumber")

    import io
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalise and clean extracted PDF text."""
    # Remove standalone page numbers
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces / tabs
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove horizontal rules (dashes / underscores / equals)
    text = re.sub(r"^[-_=]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Normalise curly quotes to straight quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking  (exported so rag_pipeline.py can import and reuse — Root Cause 1 fix)
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list:
    """
    Paragraph-aware sliding-window chunking.

    Splits the text into paragraphs first, then accumulates words into
    chunks of `chunk_size` words with `overlap` words of context carry-over.

    Returns a list of chunk strings.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    words_buffer: list = []
    chunks: list = []
    step = chunk_size - overlap

    for para in paragraphs:
        words_buffer.extend(para.split())

        while len(words_buffer) >= chunk_size:
            chunk = " ".join(words_buffer[:chunk_size])
            chunks.append(chunk)
            words_buffer = words_buffer[step:]

    # Flush any remaining words as a final partial chunk
    if words_buffer:
        chunk = " ".join(words_buffer)
        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)

    return chunks


def chunk_pdf_bytes(
    pdf_bytes: bytes,
    source_name: str = "uploaded_pdf",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list:
    """
    High-level helper: extract + clean + chunk a PDF given as raw bytes.

    Returns a list of dicts compatible with the FAISS metadata format:
        [{"text": "...", "source": "filename.pdf", "type": "uploaded"}, ...]

    Used by app.py and rag_pipeline.py to build in-memory doc_chunks
    (Root Cause 1 fix — uploaded PDF is now the primary evidence for RAG).
    """
    raw_text = extract_text_from_bytes(pdf_bytes)
    cleaned = clean_text(raw_text)

    if len(cleaned.split()) < MIN_CHUNK_WORDS:
        return []

    text_chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)

    return [
        {
            "id": str(uuid.uuid4()),
            "text": chunk,
            "source": source_name,
            "type": "uploaded",
        }
        for chunk in text_chunks
    ]


# ---------------------------------------------------------------------------
# Corpus deduplication
# ---------------------------------------------------------------------------

def load_existing_corpus() -> set:
    """
    Return a set of deduplication fingerprints (first DEDUP_KEY_LEN chars of each chunk).
    Prevents re-adding the same text when re-running ingest.
    """
    existing: set = set()
    if CORPUS_PATH.exists():
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        existing.add(entry.get("text", "")[:DEDUP_KEY_LEN])
                    except json.JSONDecodeError:
                        continue
    return existing


# ---------------------------------------------------------------------------
# Main ingestion routine
# ---------------------------------------------------------------------------

def ingest_pdfs() -> None:
    """
    Scan PDF_DIR, extract + chunk every PDF, and append new chunks to
    the JSONL corpus file.  Skips chunks already present (dedup).
    """
    # Ensure output directory exists (silent failure fix)
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not PDF_DIR.exists():
        PDF_DIR.mkdir(parents=True)
        print(f"[ingest] Created {PDF_DIR}")
        print("[ingest] Place your legal PDFs there and re-run.")
        return

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[ingest] No PDFs found in {PDF_DIR}")
        return

    print(f"[ingest] Found {len(pdf_files)} PDF(s): {[p.name for p in pdf_files]}")

    existing_keys = load_existing_corpus()
    print(f"[ingest] Existing corpus dedup fingerprints: {len(existing_keys)}")

    new_chunks: list = []
    stats: dict = {}

    for pdf_path in pdf_files:
        print(f"\n[ingest] Processing: {pdf_path.name}")
        try:
            raw_text = extract_text_pdfplumber(pdf_path)
        except Exception as exc:
            print(f"  ERROR extracting text: {exc}")
            continue

        cleaned = clean_text(raw_text)
        word_count = len(cleaned.split())

        if word_count < 50:
            print(f"  WARNING: only {word_count} words extracted — skipping (likely scanned/image PDF).")
            continue

        chunks = chunk_text(cleaned)
        print(f"  ~{word_count} words → {len(chunks)} chunks")

        added = 0
        skipped = 0
        for chunk in chunks:
            key = chunk[:DEDUP_KEY_LEN]
            if key in existing_keys:
                skipped += 1
                continue
            entry = {
                "id": str(uuid.uuid4()),
                "text": chunk,
                "source": pdf_path.name,
                "type": "pdf_ingested",
            }
            new_chunks.append(entry)
            existing_keys.add(key)
            added += 1

        stats[pdf_path.name] = {"added": added, "skipped": skipped}
        print(f"  Added {added} new chunks, skipped {skipped} duplicates")

    if not new_chunks:
        print("\n[ingest] No new chunks to add. Corpus is up-to-date.")
        return

    with open(CORPUS_PATH, "a", encoding="utf-8") as f:
        for entry in new_chunks:
            f.write(json.dumps(entry) + "\n")

    print(f"\n[ingest] ✓ Appended {len(new_chunks)} new chunks → {CORPUS_PATH}")
    print("\n[ingest] Per-file summary:")
    for name, s in stats.items():
        print(f"  {name}: +{s['added']} added, {s['skipped']} skipped")

    print("\n[ingest] → Next step: python scripts/create_faiss_index.py")


if __name__ == "__main__":
    ingest_pdfs()