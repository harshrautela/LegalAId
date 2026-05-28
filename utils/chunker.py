"""
utils/chunker.py
────────────────
Splits long legal documents into overlapping chunks suitable for
FAISS indexing and retrieval-augmented generation.

Design choice: token-overlap sliding window rather than sentence boundaries,
because Indian legal text often has run-on sentences that are semantically
inseparable. Overlap ensures no clause is cut off mid-thought.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from utils.text_utils import count_tokens_approx


@dataclass
class Chunk:
    """A single passage extracted from a legal document."""
    chunk_id:   str
    doc_id:     str
    text:       str
    char_start: int
    char_end:   int
    token_approx: int = 0
    metadata:   dict  = field(default_factory=dict)

    def __post_init__(self):
        self.token_approx = count_tokens_approx(self.text)

    def to_dict(self) -> dict:
        return {
            "chunk_id":     self.chunk_id,
            "doc_id":       self.doc_id,
            "text":         self.text,
            "char_start":   self.char_start,
            "char_end":     self.char_end,
            "token_approx": self.token_approx,
            **self.metadata,
        }


class LegalChunker:
    """
    Sliding-window chunker with two modes:
      - 'char'   : fixed character window  (fast, good for embedding)
      - 'para'   : paragraph-respecting    (better semantic boundaries)

    Parameters
    ----------
    chunk_size    : target chunk size in characters (default 1 600 ≈ 400 tokens)
    chunk_overlap : overlap between consecutive chunks (default 320 chars)
    mode          : 'char' | 'para'
    """

    DEFAULT_CHUNK_CHARS   = 1_600
    DEFAULT_OVERLAP_CHARS = 320

    def __init__(
        self,
        chunk_size: int    = DEFAULT_CHUNK_CHARS,
        chunk_overlap: int = DEFAULT_OVERLAP_CHARS,
        mode: str          = "para",
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.mode          = mode

    # ── public API ────────────────────────────────────────────────────────────

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[dict] = None,
    ) -> List[Chunk]:
        """
        Split `text` into overlapping chunks and return a list of Chunk objects.
        Each chunk carries the doc_id and optional metadata (case_name, court …).
        """
        metadata = metadata or {}
        if self.mode == "para":
            spans = self._para_spans(text)
        else:
            spans = self._char_spans(text)

        chunks: List[Chunk] = []
        for i, (start, end) in enumerate(spans):
            passage = text[start:end].strip()
            if len(passage) < 60:          # skip trivially short fragments
                continue
            chunks.append(Chunk(
                chunk_id   = f"{doc_id}__c{i:04d}",
                doc_id     = doc_id,
                text       = passage,
                char_start = start,
                char_end   = end,
                metadata   = dict(metadata, chunk_index=i),
            ))
        return chunks

    def chunk_batch(
        self,
        records: List[dict],
        text_key: str  = "text",
        id_key:   str  = "doc_id",
    ) -> List[Chunk]:
        """Convenience wrapper to chunk a list of document dicts."""
        all_chunks = []
        for rec in records:
            doc_id   = str(rec.get(id_key, "doc_unknown"))
            text     = rec.get(text_key, "")
            metadata = {k: v for k, v in rec.items()
                        if k not in (text_key, id_key)}
            all_chunks.extend(self.chunk_document(text, doc_id, metadata))
        return all_chunks

    # ── private helpers ───────────────────────────────────────────────────────

    def _char_spans(self, text: str) -> List[tuple]:
        """Simple sliding window over characters."""
        step   = self.chunk_size - self.chunk_overlap
        spans  = []
        start  = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            spans.append((start, end))
            if end == len(text):
                break
            start += step
        return spans

    def _para_spans(self, text: str) -> List[tuple]:
        """
        Paragraph-respecting chunker.
        Accumulate paragraphs until the chunk_size is reached,
        then start a new chunk with an overlap of the last paragraph(s).
        """
        # Split on double-newline paragraph boundaries
        raw_paras = re.split(r"\n{2,}", text)
        # Filter empty
        paras = [(p.strip(), text.find(p.strip())) for p in raw_paras if p.strip()]

        spans: List[tuple] = []
        buf_start = 0
        buf_chars = 0
        buf_paras: List[tuple] = []   # (text, original_start)

        for para_text, para_start in paras:
            para_len = len(para_text)

            # If a single paragraph exceeds chunk_size, hard-split it
            if para_len > self.chunk_size:
                # Flush current buffer first
                if buf_paras:
                    start = buf_paras[0][1]
                    end   = buf_paras[-1][1] + len(buf_paras[-1][0])
                    spans.append((start, end))
                    buf_paras = []
                    buf_chars = 0
                # Hard-split the big paragraph
                for sub_start, sub_end in self._char_spans_offset(
                    para_text, para_start
                ):
                    spans.append((sub_start, sub_end))
                buf_start = para_start + para_len
                continue

            if buf_chars + para_len > self.chunk_size and buf_paras:
                # Emit current chunk
                start = buf_paras[0][1]
                end   = buf_paras[-1][1] + len(buf_paras[-1][0])
                spans.append((start, end))

                # Build overlap: keep last paragraphs that fit in overlap budget
                overlap_paras: List[tuple] = []
                overlap_len = 0
                for op in reversed(buf_paras):
                    if overlap_len + len(op[0]) <= self.chunk_overlap:
                        overlap_paras.insert(0, op)
                        overlap_len += len(op[0])
                    else:
                        break
                buf_paras = overlap_paras
                buf_chars = overlap_len

            buf_paras.append((para_text, para_start))
            buf_chars += para_len

        # Flush remaining
        if buf_paras:
            start = buf_paras[0][1]
            end   = buf_paras[-1][1] + len(buf_paras[-1][0])
            spans.append((start, end))

        return spans

    def _char_spans_offset(
        self, text: str, offset: int
    ) -> List[tuple]:
        """Char-window spans for a single paragraph, translated by offset."""
        base = self._char_spans(text)
        return [(s + offset, e + offset) for s, e in base]


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "This is a judgment of the Supreme Court of India.\n\n"
        "The petitioner filed a writ petition challenging the order of the "
        "High Court of Delhi dated 12 March 2019.\n\n"
        "The respondents filed a counter-affidavit on 5 April 2019.\n\n"
        "After hearing both sides, the Court held that the impugned order was "
        "contrary to the principles of natural justice and set it aside.\n\n"
        "Accordingly, the writ petition is allowed with costs of Rs. 10,000 "
        "imposed on the respondents.\n\n"
    ) * 20   # make it long enough to trigger multi-chunk

    chunker = LegalChunker(chunk_size=600, chunk_overlap=120, mode="para")
    chunks  = chunker.chunk_document(sample, doc_id="test_doc_001")
    print(f"Produced {len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"  [{c.chunk_id}] chars {c.char_start}–{c.char_end}: "
              f"{c.text[:80]}…")