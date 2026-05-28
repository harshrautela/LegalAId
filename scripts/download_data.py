"""
scripts/download_data.py
────────────────────────
Downloads and caches the three legal datasets you asked for:

1) viber1/indian-law-dataset
   - used as the RAG corpus
   - FULL download by default (~24.6k rows)
   - can be capped if you switch FULL_RAG_DOWNLOAD = False

2) ai4bharat/IndicQA
   - QA dataset
   - loads all supported language configs
   - keeps only English-like rows
   - if not enough English rows exist, fills the target with synthetic English QA

3) SaiCharanChetpelly/legal-summarization
   - summarization dataset
   - keeps all real rows (20 rows)
   - fills the target with synthetic legal summaries

Usage
─────
    python scripts/download_data.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from itertools import product
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR

# ── output / control flags ───────────────────────────────────────────────────
FORCE_REBUILD = False

# Summarization: keep all real rows (20) and fill with synthetic examples
TARGET_SUMMARISATION = 240

# QA: keep English-only rows, then fill to this size with synthetic English QA
TARGET_QA = 800

# RAG corpus: FULL viber1 dataset by default
FULL_RAG_DOWNLOAD = True
TARGET_RAG_CORPUS = 600  # used only if FULL_RAG_DOWNLOAD = False

# If you want smaller files for quick tests, change these:
# TARGET_SUMMARISATION = 120
# TARGET_QA = 600
# FULL_RAG_DOWNLOAD = False
# TARGET_RAG_CORPUS = 600

# ── sources ───────────────────────────────────────────────────────────────────
SUMMARISATION_SOURCE = "SaiCharanChetpelly/legal-summarization"
QA_SOURCE = "ai4bharat/IndicQA"
RAG_SOURCE = "viber1/indian-law-dataset"

# IndicQA configs from the dataset script
QA_LANGS = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]

# ── determinism ───────────────────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

RAW_DIR.mkdir(parents=True, exist_ok=True)
logger.add(str(RAW_DIR / "download.log"), rotation="5 MB")

# ── helpers ───────────────────────────────────────────────────────────────────
INDIC_SCRIPT_RE = re.compile(
    r"[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F"
    r"\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0E00-\u0E7F"
    r"\u4E00-\u9FFF\u3040-\u30FF]"
)


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records):,} records → {path}")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_str(v: Any, default: str = "") -> str:
    text = normalize_text(v)
    return text if text else default


def safe_len(x: Any) -> int:
    return len(x) if isinstance(x, str) else 0


def truncate(text: str, limit: int) -> str:
    text = normalize_text(text)
    return text[:limit] if len(text) > limit else text


def english_like(text: str) -> bool:
    """
    Strong heuristic for English-only filtering.
    Rejects rows containing Indic / CJK scripts and requires ASCII-letter dominance.
    """
    text = normalize_text(text)
    if not text:
        return False

    if INDIC_SCRIPT_RE.search(text):
        return False

    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 12:
        return False

    ascii_letters = sum(ch.isascii() for ch in letters)
    return (ascii_letters / len(letters)) >= 0.95


def extract_text_value(value: Any) -> str:
    """
    Robustly extract a string from strings, lists, or nested dicts.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return normalize_text(value)

    if isinstance(value, dict):
        preferred_keys = [
            "text", "answer", "answers", "output", "response",
            "content", "body", "value", "label", "summary"
        ]
        for key in preferred_keys:
            if key in value:
                extracted = extract_text_value(value.get(key))
                if extracted:
                    return extracted
        for _, v in value.items():
            extracted = extract_text_value(v)
            if extracted:
                return extracted
        return ""

    if isinstance(value, (list, tuple)):
        for item in value:
            extracted = extract_text_value(item)
            if extracted:
                return extracted
        return ""

    return normalize_text(value)


def pick_first(row: dict, keys: list[str], default: str = "") -> str:
    for key in keys:
        if key not in row:
            continue
        value = extract_text_value(row.get(key))
        if value:
            return value
    return default


def dedupe_records(records: list[dict], key_fields: list[str]) -> list[dict]:
    seen = set()
    out = []
    for rec in records:
        key = tuple(normalize_text(safe_str(rec.get(k, "")))[:700] for k in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def shuffle_records(records: list[dict]) -> list[dict]:
    records = records[:]
    random.shuffle(records)
    return records


def iter_all_rows(ds_obj) -> Iterable[dict]:
    """
    Accepts Dataset, DatasetDict, or any list-like object.
    """
    if isinstance(ds_obj, DatasetDict):
        for split_name in ds_obj.keys():
            try:
                for row in ds_obj[split_name]:
                    yield row
            except Exception as e:
                logger.warning(f"Could not iterate split '{split_name}': {e}")
    elif isinstance(ds_obj, Dataset):
        for row in ds_obj:
            yield row
    else:
        try:
            for row in ds_obj:
                yield row
        except Exception:
            return


def load_hf_dataset(name: str, config: str | None = None):
    if config:
        return load_dataset(name, config, trust_remote_code=True)
    return load_dataset(name, trust_remote_code=True)


# ── synthetic generators ─────────────────────────────────────────────────────
def _make_summarisation_record(i: int, topic: str, forum: str, statute: str, remedy: str, principle: str) -> dict:
    case_tag = f"{forum} matter on {topic.lower()}"
    text = (
        f"In a dispute concerning {topic}, the petitioner invoked {statute}. "
        f"The {forum} examined the pleadings, the record, and the arguments on fairness and legality. "
        f"It noted that {remedy} was sought because the previous order was alleged to be contrary to {principle}. "
        f"After considering the materials, the Court held that {remedy.lower()} was justified."
    )
    summary = (
        f"{forum} case on {topic}: the court applied {principle} and held that {remedy.lower()} was justified."
    )
    return {
        "doc_id": f"sum_syn_{i:05d}",
        "text": normalize_text(text),
        "summary": normalize_text(summary),
        "source": "synthetic",
        "task": "summarisation",
        "language": "en",
        "case_tag": case_tag,
    }


def _synthetic_summarisation_seed(target: int) -> list[dict]:
    topics = [
        "bail cancellation",
        "service termination",
        "tender rejection",
        "municipal demolition",
        "maintenance dispute",
        "passport refusal",
        "land acquisition compensation",
        "disciplinary inquiry",
        "consumer refund claim",
        "tax assessment challenge",
        "writ petition on delay",
        "appointment dispute",
    ]
    forums = ["Supreme Court", "High Court", "Tribunal", "Family Court"]
    statutes = [
        "Article 14 of the Constitution",
        "Article 21 of the Constitution",
        "the Arms Act, 1959",
        "the Consumer Protection Act, 2019",
        "the Limitation Act, 1963",
        "the Code of Civil Procedure, 1908",
        "the Indian Contract Act, 1872",
        "the service rules applicable to the department",
    ]
    remedies = [
        "the writ petition",
        "the appeal",
        "interim relief",
        "regular bail",
        "reinstatement",
        "compensation",
        "quashing of the order",
        "fresh consideration of the claim",
    ]
    principles = [
        "the rule of natural justice",
        "procedural fairness",
        "the doctrine of proportionality",
        "judicial review",
        "reasoned decision-making",
        "the right to livelihood",
        "constitutional equality",
        "legality and fairness",
    ]

    combos = list(product(topics, forums, statutes, remedies, principles))
    random.shuffle(combos)

    out = []
    for i, (topic, forum, statute, remedy, principle) in enumerate(combos[:target]):
        out.append(_make_summarisation_record(i, topic, forum, statute, remedy, principle))
    return out


def _make_qa_record(i: int, topic: str, forum: str, statute: str, focus: str, outcome: str) -> dict:
    context = (
        f"In a dispute concerning {topic}, the {forum} considered whether {focus}. "
        f"The record referred to {statute}, and the parties argued over the legality of the earlier order. "
        f"After reviewing the facts, the court held that {outcome}."
    )
    question_options = [
        f"What did the {forum} decide in the matter involving {topic}?",
        f"Why was the earlier order challenged in the case about {topic}?",
        f"What legal issue did the court examine in the dispute over {topic}?",
        f"What was the result of the {forum} review in the {topic} case?",
        f"Which principle guided the court in the {topic} dispute?",
    ]
    answer = (
        f"The {forum} held that {outcome}. "
        f"It relied on {statute} and applied {focus} while deciding the matter."
    )
    return {
        "doc_id": f"qa_syn_{i:05d}",
        "context": normalize_text(context),
        "question": normalize_text(question_options[i % len(question_options)]),
        "answer": normalize_text(answer),
        "source": "synthetic",
        "task": "qa",
        "language": "en",
    }


def _synthetic_qa_seed(target: int) -> list[dict]:
    topics = [
        "bail",
        "maintenance",
        "passport renewal",
        "tender evaluation",
        "service termination",
        "land acquisition",
        "tax reassessment",
        "consumer refund",
        "disciplinary action",
        "writ jurisdiction",
        "school admission",
        "demolition notice",
        "arbitration award",
        "employment benefits",
        "custody dispute",
    ]
    forums = ["Supreme Court", "High Court", "District Court", "Tribunal"]
    statutes = [
        "Article 14 of the Constitution",
        "Article 19 of the Constitution",
        "Article 21 of the Constitution",
        "the Limitation Act, 1963",
        "the Consumer Protection Act, 2019",
        "the Code of Criminal Procedure, 1973",
        "the Code of Civil Procedure, 1908",
        "the relevant service rules",
    ]
    focus_points = [
        "the order satisfied the requirements of fairness",
        "the previous authority had given a proper hearing",
        "the evidence supported the claimant's request",
        "the delay was sufficiently explained",
        "the action was proportionate",
        "the authority acted within jurisdiction",
        "the affected person had a valid legal remedy",
        "the impugned order could not stand",
    ]
    outcomes = [
        "the petition was allowed",
        "the appeal was dismissed",
        "interim relief was granted",
        "the order was quashed",
        "the application was rejected",
        "the matter was remanded for fresh consideration",
        "the relief was confined to costs only",
        "the court directed compliance within four weeks",
    ]

    combos = list(product(topics, forums, statutes, focus_points, outcomes))
    random.shuffle(combos)

    out = []
    for i, (topic, forum, statute, focus, outcome) in enumerate(combos[:target]):
        out.append(_make_qa_record(i, topic, forum, statute, focus, outcome))
    return out


def _make_rag_synthetic_record(i: int, issue: str, court: str, principle: str, disposition: str) -> dict:
    text = (
        f"{court} judgment on {issue}. The bench analyzed pleadings, precedent, and statutory interpretation. "
        f"It emphasized {principle} and concluded that {disposition}. "
        f"The ruling is useful for retrieval because it links facts, legal reasoning, and final relief."
    )
    return {
        "doc_id": f"rag_syn_{i:05d}",
        "text": normalize_text(text),
        "case_name": f"Synthetic {court} Matter on {issue}",
        "court": court,
        "year": "N/A",
        "source": "synthetic",
        "language": "en",
    }


def _synthetic_rag_seed(target: int) -> list[dict]:
    issues = [
        "natural justice",
        "bail conditions",
        "service regularization",
        "tax exemption",
        "land possession",
        "consumer deficiency",
        "tender cancellation",
        "maintenance allowance",
        "writ maintainability",
        "disciplinary inquiry",
        "arbitration enforcement",
        "demolition notice",
    ]
    courts = ["Supreme Court of India", "Delhi High Court", "Bombay High Court", "National Tribunal"]
    principles = [
        "fair hearing before adverse action",
        "proportionality in administrative decisions",
        "the requirement of reasons in orders",
        "the right to livelihood and dignity",
        "the need for jurisdictional compliance",
        "the limits of executive power",
        "the importance of precedent",
        "the balance between public interest and individual rights",
    ]
    dispositions = [
        "the petition was allowed",
        "the appeal was dismissed",
        "the order was set aside",
        "the matter was remanded",
        "interim protection was granted",
        "costs were imposed",
        "the challenge failed",
        "fresh hearing was directed",
    ]

    combos = list(product(issues, courts, principles, dispositions))
    random.shuffle(combos)

    out = []
    for i, (issue, court, principle, disposition) in enumerate(combos[:target]):
        out.append(_make_rag_synthetic_record(i, issue, court, principle, disposition))
    return out


# ── Dataset 1 – summarisation ────────────────────────────────────────────────
def download_summarisation() -> list[dict]:
    out = RAW_DIR / "summarisation_raw.jsonl"
    if out.exists() and not FORCE_REBUILD:
        logger.info("Summarisation data already present – skipping download.")
        return load_jsonl(out)

    logger.info("Downloading legal summarisation dataset …")
    records: list[dict] = []

    try:
        ds = load_hf_dataset(SUMMARISATION_SOURCE)
        total_rows = len(ds) if not isinstance(ds, DatasetDict) else sum(len(ds[k]) for k in ds.keys())
        logger.info(f"Loaded {SUMMARISATION_SOURCE}: {total_rows} rows")
    except Exception as e:
        logger.warning(f"Could not load summarisation dataset: {e}")
        ds = None

    if ds is not None:
        for row in iter_all_rows(ds):
            text = pick_first(
                row,
                ["text", "judgment", "judgement", "input", "article", "content", "body", "document", "passage", "instruction"]
            )
            summary = pick_first(
                row,
                ["summary", "output", "target", "abstract", "answer", "label", "response", "completion"]
            )

            text = normalize_text(text)
            summary = normalize_text(summary)

            if safe_len(text) >= 80 and safe_len(summary) >= 20:
                records.append({
                    "doc_id": f"sum_real_{len(records):05d}",
                    "text": truncate(text, 8000),
                    "summary": truncate(summary, 2000),
                    "source": SUMMARISATION_SOURCE,
                    "task": "summarisation",
                    "language": "en" if english_like(text + " " + summary) else "unknown",
                })

    records = dedupe_records(records, ["text", "summary"])

    if len(records) < TARGET_SUMMARISATION:
        need = TARGET_SUMMARISATION - len(records)
        logger.warning(
            f"Summarisation has only {len(records):,} real rows; adding {need:,} synthetic rows."
        )
        synth = _synthetic_summarisation_seed(need)
        records.extend(synth)

    records = dedupe_records(records, ["text", "summary"])
    records = shuffle_records(records)

    if len(records) > TARGET_SUMMARISATION:
        records = records[:TARGET_SUMMARISATION]

    save_jsonl(records, out)
    return records


# ── Dataset 2 – QA (English only) ────────────────────────────────────────────
def download_qa() -> list[dict]:
    out = RAW_DIR / "qa_raw.jsonl"
    if out.exists() and not FORCE_REBUILD:
        logger.info("QA data already present – skipping download.")
        return load_jsonl(out)

    logger.info("Downloading IndicQA and filtering to English-only rows …")
    records: list[dict] = []

    for lang in QA_LANGS:
        cfg = f"indicqa.{lang}"
        try:
            ds = load_hf_dataset(QA_SOURCE, cfg)
            total_rows = len(ds) if not isinstance(ds, DatasetDict) else sum(len(ds[k]) for k in ds.keys())
            logger.info(f"Loaded {QA_SOURCE}/{cfg}: {total_rows} rows")

            for row in iter_all_rows(ds):
                context = pick_first(row, ["context", "text", "input", "passage", "judgment", "judgement", "article", "document", "content", "body"])
                question = pick_first(row, ["question", "instruction", "prompt", "query"])
                answer = pick_first(row, ["answer", "output", "summary", "label", "target", "response", "completion", "answers"])

                context = normalize_text(context)
                question = normalize_text(question)
                answer = normalize_text(answer)

                if not (context and question and answer):
                    continue

                # Keep only English-like rows.
                if not (english_like(context) and english_like(question) and english_like(answer)):
                    continue

                records.append({
                    "doc_id": f"qa_real_{len(records):05d}",
                    "context": truncate(context, 9000),
                    "question": truncate(question, 1500),
                    "answer": truncate(answer, 3000),
                    "source": f"{QA_SOURCE}/{cfg}",
                    "task": "qa",
                    "language": "en",
                    "source_lang": lang,
                })
        except Exception as e:
            logger.warning(f"QA dataset unavailable ({cfg}): {e}")

    records = dedupe_records(records, ["context", "question", "answer"])

    if len(records) < TARGET_QA:
        need = TARGET_QA - len(records)
        logger.warning(
            f"English QA rows found: {len(records):,}. Adding {need:,} synthetic English legal QA rows."
        )
        synth = _synthetic_qa_seed(need)
        records.extend(synth)

    records = dedupe_records(records, ["context", "question", "answer"])
    records = shuffle_records(records)

    if len(records) > TARGET_QA:
        records = records[:TARGET_QA]

    save_jsonl(records, out)
    return records


# ── Dataset 3 – RAG corpus ───────────────────────────────────────────────────
def _build_rag_text(row: dict) -> str:
    instruction = pick_first(row, ["instruction", "Instruction", "question", "prompt", "input", "query"])
    response = pick_first(row, ["response", "Response", "output", "answer", "completion", "target"])
    text = "\n\n".join(part for part in [instruction, response] if part)
    if text:
        return normalize_text(text)

    # fallback if the dataset uses different column names
    text = pick_first(
        row,
        ["text", "facts", "judgment", "judgement", "content", "body", "document", "passage", "summary"]
    )
    return normalize_text(text)


def download_rag_corpus() -> list[dict]:
    out = RAW_DIR / "rag_corpus_raw.jsonl"
    if out.exists() and not FORCE_REBUILD:
        logger.info("RAG corpus already present – skipping download.")
        return load_jsonl(out)

    logger.info("Downloading RAG corpus from viber1/indian-law-dataset …")
    records: list[dict] = []

    try:
        ds = load_hf_dataset(RAG_SOURCE)
        total_rows = len(ds) if not isinstance(ds, DatasetDict) else sum(len(ds[k]) for k in ds.keys())
        logger.info(f"Loaded {RAG_SOURCE}: {total_rows} rows")
    except Exception as e:
        logger.warning(f"Could not load RAG corpus dataset: {e}")
        ds = None

    if ds is not None:
        for row in iter_all_rows(ds):
            text = _build_rag_text(row)
            if safe_len(text) < 120:
                continue

            case_name = pick_first(row, ["case_name", "name", "title", "case", "heading"], "Unknown")
            court = pick_first(row, ["court", "forum"], "Supreme Court of India")
            year = pick_first(row, ["year", "date", "decision_year", "judgment_year"], "N/A")

            records.append({
                "doc_id": f"rag_real_{len(records):05d}",
                "text": truncate(text, 12000),
                "case_name": case_name,
                "court": court,
                "year": year,
                "source": RAG_SOURCE,
                "task": "rag",
                "language": "en" if english_like(text) else "unknown",
            })

            if not FULL_RAG_DOWNLOAD and len(records) >= TARGET_RAG_CORPUS:
                break

    records = dedupe_records(records, ["text", "case_name", "year"])

    # Only use synthetic fallback if the corpus is unexpectedly tiny.
    if not FULL_RAG_DOWNLOAD and len(records) < TARGET_RAG_CORPUS:
        need = TARGET_RAG_CORPUS - len(records)
        logger.warning(
            f"RAG corpus has only {len(records):,} rows in sample mode; adding {need:,} synthetic corpus rows."
        )
        synth = _synthetic_rag_seed(need)
        records.extend(synth)
        records = dedupe_records(records, ["text", "case_name", "year"])

    records = shuffle_records(records)

    if not FULL_RAG_DOWNLOAD and len(records) > TARGET_RAG_CORPUS:
        records = records[:TARGET_RAG_CORPUS]

    save_jsonl(records, out)
    return records


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    logger.info("═" * 60)
    logger.info(" LegalAId – Data Download ")
    logger.info("═" * 60)

    sum_records = download_summarisation()
    qa_records = download_qa()
    rag_records = download_rag_corpus()

    logger.info(
        f"\nDownload summary:\n"
        f"  Summarisation : {len(sum_records):,} records\n"
        f"  QA            : {len(qa_records):,} records\n"
        f"  RAG corpus    : {len(rag_records):,} records\n"
        f"  Total         : {len(sum_records) + len(qa_records) + len(rag_records):,} records"
    )
    logger.info("Download complete ✓")


if __name__ == "__main__":
    main()