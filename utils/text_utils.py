"""
utils/text_utils.py
───────────────────
Text-cleaning helpers shared across all pipeline stages.
Handles OCR noise, legal boilerplate, and normalization.
"""

import re
import unicodedata
from typing import List, Optional

# ── OCR & noise patterns ──────────────────────────────────────────────────────
_PAGE_HEADER_RE  = re.compile(
    r"(?im)^(page\s+\d+\s+of\s+\d+|www\.\S+|all rights reserved.*|"
    r"[\u00a9\u00ae]\s*\d{4}.*|printed\s+on\s+.*|generated\s+by\s+.*)$"
)
_PAGE_NUMBER_RE  = re.compile(r"(?m)^\s*-?\s*\d+\s*-?\s*$")
_OCR_ARTIFACTS   = re.compile(r"[^\x00-\x7F\u0900-\u097F\s]")  # non-ASCII non-Devanagari
_MULTI_SPACE_RE  = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE   = re.compile(r"\n{3,}")
_REPEATED_PUNCT  = re.compile(r"([.,;:\-–—])\1{2,}")
_ORDINAL_FIX     = re.compile(r"(\d)(st|nd|rd|th)\b", re.I)
_SMART_QUOTES    = str.maketrans({"\u2018": "'", "\u2019": "'",
                                  "\u201c": '"', "\u201d": '"',
                                  "\u2013": "-", "\u2014": "-",
                                  "\u00a0": " "})

# Common Indian legal boilerplate lines to strip
_BOILERPLATE_FRAGMENTS = [
    "reportable", "not reportable", "for private circulation only",
    "in the supreme court of india", "civil appellate jurisdiction",
    "criminal appellate jurisdiction", "writ petition",
    "judgment reserved on", "judgment delivered on",
    "signature of the judge", "bench strength",
]


def normalize_unicode(text: str) -> str:
    """NFC-normalise and replace smart punctuation."""
    text = unicodedata.normalize("NFC", text)
    return text.translate(_SMART_QUOTES)


def remove_ocr_noise(text: str) -> str:
    """Strip non-printable characters, OCR artefacts, etc."""
    text = _OCR_ARTIFACTS.sub(" ", text)
    text = _REPEATED_PUNCT.sub(r"\1", text)
    return text


def remove_headers_footers(text: str) -> str:
    """Remove page headers, footers, and standalone page numbers."""
    text = _PAGE_HEADER_RE.sub("", text)
    text = _PAGE_NUMBER_RE.sub("", text)
    return text


def remove_boilerplate(text: str) -> str:
    """Remove common Indian court boilerplate lines."""
    lines = text.split("\n")
    clean = []
    for line in lines:
        lower = line.lower().strip()
        if any(frag in lower for frag in _BOILERPLATE_FRAGMENTS):
            continue
        clean.append(line)
    return "\n".join(clean)


def normalize_whitespace(text: str) -> str:
    """Collapse multi-spaces and excessive newlines."""
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def clean_legal_text(text: str, remove_boilerplate_lines: bool = True) -> str:
    """
    Full cleaning pipeline for raw Indian legal text.

    Steps:
      1. Unicode normalisation
      2. OCR noise removal
      3. Header / footer stripping
      4. Optional boilerplate removal
      5. Whitespace normalisation
    """
    text = normalize_unicode(text)
    text = remove_ocr_noise(text)
    text = remove_headers_footers(text)
    if remove_boilerplate_lines:
        text = remove_boilerplate(text)
    text = normalize_whitespace(text)
    return text


def deduplicate_paragraphs(text: str, min_len: int = 30) -> str:
    """
    Remove duplicate paragraphs (e.g. repeated legal disclaimer blocks).
    Only paragraphs longer than min_len are checked.
    """
    seen: set = set()
    paras = text.split("\n\n")
    result = []
    for para in paras:
        key = " ".join(para.lower().split())
        if len(key) >= min_len:
            if key in seen:
                continue
            seen.add(key)
        result.append(para)
    return "\n\n".join(result)


def truncate_tokens(text: str, max_chars: int = 2000) -> str:
    """Simple character-based truncation for display / embedding limits."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Break at last sentence boundary
    for end_char in (".", "?", "!", "\n"):
        pos = cut.rfind(end_char)
        if pos > max_chars // 2:
            return cut[: pos + 1].strip()
    return cut.strip() + " …"


def extract_citations(text: str) -> List[str]:
    """
    Heuristically extract Indian legal citation strings like:
    AIR 2004 SC 123 · (2010) 5 SCC 200 · WP 456/2022
    """
    patterns = [
        r"\bAIR\s+\d{4}\s+[A-Z]+\s+\d+",
        r"\(\d{4}\)\s+\d+\s+SCC\s+\d+",
        r"\bWP(?:\s*\(C\))?\s+\d+/\d{4}",
        r"\bCRL\.?\s*APP?\.?\s+\d+/\d{4}",
        r"\bCA\s+No\.?\s+\d+\s+of\s+\d{4}",
    ]
    results = []
    for pat in patterns:
        results.extend(re.findall(pat, text, re.I))
    return list(set(results))


def count_tokens_approx(text: str) -> int:
    """Rough token count: ~1.3 chars per token for English legal text."""
    return max(1, len(text) // 4)


def split_into_sentences(text: str) -> List[str]:
    """
    Sentence splitter aware of legal abbreviations like
    Sec., Cl., Art., Hon'ble, v., etc.
    """
    abbrevs = {"sec", "cl", "art", "vs", "v", "hon'ble", "nos",
               "no", "para", "col", "vol", "pg", "pp", "ibid", "id",
               "cf", "op", "cit", "et", "al"}
    # Split on '. ' or '.\n' but not after known abbreviations
    sentences = []
    current = []
    words = text.split()
    for i, word in enumerate(words):
        current.append(word)
        if word.endswith(".") or word.endswith("?") or word.endswith("!"):
            bare = word.rstrip(".?!").lower()
            if bare not in abbrevs:
                sentences.append(" ".join(current))
                current = []
    if current:
        sentences.append(" ".join(current))
    return [s.strip() for s in sentences if s.strip()]


def highlight_query_terms(text: str, query: str, window: int = 200) -> str:
    """
    Return a snippet of `text` centred around the first hit of any
    query term, for evidence display in the UI.
    """
    terms = [t.lower() for t in query.split() if len(t) > 3]
    lower_text = text.lower()
    best_pos = len(text)
    for term in terms:
        pos = lower_text.find(term)
        if 0 <= pos < best_pos:
            best_pos = pos
    if best_pos == len(text):
        return text[:window]
    start = max(0, best_pos - window // 2)
    end   = min(len(text), best_pos + window // 2)
    snippet = text[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet