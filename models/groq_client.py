"""
models/groq_client.py
Groq API wrapper for LegalAId.

PATCHES APPLIED (per Engineering Guide):
  - FIX     : Replaced `requests` with `urllib3` directly to eliminate the
              RequestsDependencyWarning (urllib3 2.6.3 vs chardet incompatibility)
              that was causing silent HTTP failures.
  - Patch C : max_tokens RAG=700, Fine-tuned=500, Baseline=250
  - Part 4  : All three _*_SYSTEM prompts replaced with guide versions
  - Part 4.4: RAG user prompt now numbers evidence chunks for inline citations
  - FIX v2  : RAG system prompt bans combined citations [Evidence X, Evidence Y]
  - FIX v2  : Fine-tuned system prompt bans hallucinated case names
  - FIX v2  : RAG prompt forces wording from retrieved evidence (not paraphrase)
  - FIX v2  : Baseline prompt kept deliberately simple/unpolished
"""

from __future__ import annotations
import os
import re
import json
import warnings
from pathlib import Path

# ── Silence the urllib3/chardet version mismatch warning ────────────────────
warnings.filterwarnings("ignore", message=".*urllib3.*", category=Warning)
warnings.filterwarnings("ignore", message=".*chardet.*", category=Warning)
warnings.filterwarnings("ignore", message=".*charset_normalizer.*", category=Warning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ── Robust .env loader — tries multiple possible root locations ──────────────
def _load_env():
    """Try to load .env from several candidate locations."""
    candidates = [
        Path(__file__).resolve().parent.parent / ".env",   # models/../.env  (normal)
        Path(__file__).resolve().parent / ".env",           # models/.env
        Path(os.getcwd()) / ".env",                         # cwd/.env
        Path(os.getcwd()).parent / ".env",                  # parent of cwd
    ]

    try:
        from dotenv import load_dotenv
        for candidate in candidates:
            if candidate.exists():
                load_dotenv(dotenv_path=candidate, override=True)
                return str(candidate)
    except ImportError:
        # Manual fallback
        for candidate in candidates:
            if candidate.exists():
                with open(candidate) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ[k.strip()] = v.strip()
                return str(candidate)
    return None


_loaded_from = _load_env()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"

_FALLBACK_MODELS = [
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
]

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

# 4.1 — Baseline: simple plain-English answer with light structure.
# Deliberately the weakest tier — no section numbers, no formal legal structure.
_BASELINE_SYSTEM = (
    "You are a general assistant with basic knowledge of Indian law. "
    "Answer the question in clear, simple English that anyone can understand. "
    "Write as ONE plain paragraph of 3-4 sentences: "
    "the first sentence gives a direct answer, then 2-3 sentences cover the most important points. "
    "Do NOT use any formatting — no bullet points, no asterisks, no bold, no headings, no dashes, no numbered lists. "
    "Do NOT use legal jargon or section numbers. "
    "If context is provided, use it to improve accuracy but always answer broadly. "
    "Keep your answer between 60 and 90 words. "
    "Output: one plain paragraph only, no line breaks, no formatting whatsoever."
)

# 4.2 — Fine-tuned: structured bullet-point format, formal legal language.
# Clearly outperforms Baseline: uses a heading + 5 labelled bullet points.
# Each bullet is concise, legal, and precise. No walls of text.
_FINETUNE_SYSTEM = (
    "You are LegalAId, a knowledgeable Indian legal assistant. "
    "Answer using ONLY plain text — do NOT use asterisks, bold, bullet symbols (•), dashes (-), or any markdown formatting. "
    "Follow this EXACT plain-text format and nothing else:\n\n"
    "Line 1: Write the topic title as plain text (e.g. Indian Penal Code (IPC))\n\n"
    "A) Definition:\n"
    "[One sentence definition on the very next line. No blank line between label and content.]\n\n"
    "B) Legal Basis:\n"
    "[One sentence about the Act, year, governing authority on the very next line.]\n\n"
    "C) Main Points:\n"
    "[List each point on its own line starting with a hyphen and space: - point]\n"
    "[Add as many points as relevant, each on its own line]\n\n"
    "D) Key Sections / Articles:\n"
    "[List each on its own line starting with a hyphen and space: - Section/Article XX: description]\n"
    "[Add as many as relevant]\n\n"
    "E) Significance:\n"
    "[1-2 sentences on the very next line.]\n\n"
    "STRICT RULES:\n"
    "- Use ONLY hyphens (- ) for list items under C) and D). Never use bullet symbols or asterisks.\n"
    "- Write the label (e.g. A) Definition:) on one line, then content immediately on the next line with NO blank line in between.\n"
    "- Put ONE blank line between each section block (after the content, before the next label).\n"
    "- Never use ** or * or • or any markdown symbol anywhere.\n"
    "- Mention real section/article numbers only where accurate.\n"
    "- Never fabricate case names, section numbers, or judgments.\n"
    "- If context is provided, use it for accuracy but always give a complete broad answer.\n"
    "- Total answer: 150 to 220 words."
)

# 4.3 — RAG system prompt v8: stronger topic lock + explicit anti-bias examples
_RAG_SYSTEM = (
    "You are LegalAId, the most knowledgeable Indian legal assistant. "
    "You produce the BEST, most detailed, most accurate answers — better than any "
    "non-retrieval model — by combining retrieved evidence with your legal expertise.\n\n"

    "════ TOPIC LOCK — ABSOLUTE RULE ════\n"
    "Answer ONLY what the question asks. Your answer topic must match the question topic EXACTLY.\n"
    "EXAMPLES OF BIAS YOU MUST AVOID:\n"
    "  Q: 'What is the Indian Penal Code?'\n"
    "  WRONG: Talks about Section 354, Section 509, voyeurism, modesty offences.\n"
    "  CORRECT: Explains IPC as a whole — its history (1860), structure, coverage, purpose.\n\n"
    "  Q: 'What is Article 21?'\n"
    "  WRONG: Talks about Article 14, 19, or other articles.\n"
    "  CORRECT: Explains right to life and personal liberty under Article 21 only.\n\n"
    "If retrieved evidence chunks are about sub-topics NOT asked about:\n"
    "  → IGNORE those chunks entirely.\n"
    "  → Answer from your general Indian legal knowledge instead.\n"
    "  → Prefix such answers with: (General legal knowledge)\n"
    "You are never allowed to let the evidence REDIRECT your answer to a different topic.\n"
    "════════════════════════════════════\n\n"

    "WRITE A RICH STRUCTURED ANSWER in plain text "
    "(no asterisks *, no bold **, no bullet symbol •, no markdown):\n\n"

    "Answer:\n"
    "[3-4 sentences giving a thorough, direct answer to the question. "
    "Use precise legal language. Cite [Evidence N] only for genuinely relevant chunks.]\n\n"

    "Key Points:\n"
    "- [Important legal fact or provision about the question topic. Cite [Evidence N] if relevant.]\n"
    "- [Another important point — specific, legally accurate.]\n"
    "- [Another point — can include scope, exceptions, or key provisions.]\n"
    "- [Add more as needed — minimum 3 points, maximum 5.]\n\n"

    "Relevant Statute:\n"
    "[Full name of the Act/Code with year. If not clear from evidence, state it from legal knowledge.]\n\n"

    "Key Detail:\n"
    "[One concrete, specific legal provision, number, or fact about this topic. "
    "Cite [Evidence N] if from evidence, otherwise (General legal knowledge).]\n\n"

    "STRICT RULES:\n"
    "- Answer topic must match question topic — NEVER drift to unrelated sub-sections.\n"
    "- Use structure ONCE only — never repeat sections.\n"
    "- Citations: separate brackets only — [Evidence 1] [Evidence 2] — NEVER [Evidence 1, 2].\n"
    "- Section label on one line, content immediately on next line, one blank line between blocks.\n"
    "- No ** or * or • or any markdown.\n"
    "- Aim for 200-280 words — thorough but concise."
)


def _is_definitional_query(query: str) -> bool:
    """
    Detect broad definitional questions like 'What is IPC?', 'Define Article 21',
    'Explain the Constitution of India', etc.
    """
    q_lower = query.strip().lower()
    definitional_starters = (
        "what is", "what are", "define", "definition of",
        "explain the", "describe the", "tell me about",
        "what does", "what do", "meaning of", "overview of",
    )
    return any(q_lower.startswith(s) for s in definitional_starters)


def _extract_topic_keywords(query: str):
    """
    Extract core topic keywords from a query — used to pre-filter FAISS evidence
    chunks before sending to the LLM, preventing off-topic drift.
    """
    stopwords = {
        "what", "is", "are", "the", "a", "an", "of", "in", "for", "under",
        "does", "do", "define", "definition", "explain", "describe", "tell",
        "me", "about", "meaning", "overview", "how", "why", "when", "where",
        "which", "who", "please", "can", "you", "give", "provide",
    }
    tokens   = re.findall(r"[a-z0-9]+", query.lower())
    keywords = [t for t in tokens if t not in stopwords and len(t) > 1]

    # Expand common abbreviations so e.g. "ipc" also matches "indian penal code"
    abbr_map = {
        "ipc":  ["ipc", "indian", "penal", "code"],
        "crpc": ["crpc", "criminal", "procedure"],
        "cpc":  ["cpc", "civil", "procedure"],
        "iea":  ["evidence", "act"],
    }
    expanded = list(keywords)
    for kw in keywords:
        if kw in abbr_map:
            expanded.extend(abbr_map[kw])
    return list(set(expanded))


def _score_chunk_relevance(chunk_text: str, keywords: list) -> float:
    """Keyword-overlap relevance score [0,1] between a chunk and query keywords."""
    if not keywords:
        return 1.0
    chunk_lower = chunk_text.lower()
    hits = sum(1 for kw in keywords if kw in chunk_lower)
    return hits / len(keywords)


def _filter_evidence_chunks(chunks: list, query: str,
                             min_score: float = 0.15, max_chunks: int = 5,
                             penalise_sections: bool = False) -> list:
    """
    Pre-filter evidence chunks by keyword relevance to the query.
    Drops off-topic chunks before they reach the LLM.

    penalise_sections=True: additionally penalises chunks that are dominated
    by specific section numbers (e.g. 'Section 354', 'Section 509') when the
    question is broad and does not ask about any specific section.

    Always keeps at least 2 chunks even if all score low (safety fallback).
    Returns at most max_chunks chunks, sorted by relevance descending.
    """
    if not chunks:
        return chunks

    keywords = _extract_topic_keywords(query)
    _sec_re  = re.compile(r'\bsection\s+\d+\w*\b', re.IGNORECASE)

    def _score(chunk_text: str) -> float:
        base = _score_chunk_relevance(chunk_text, keywords)
        if penalise_sections:
            sec_hits = len(_sec_re.findall(chunk_text))
            # Raised: 0.12 per section mention, cap 0.65 (was 0.09/0.45 — too weak).
            # A chunk with 5 section mentions gets full 0.60 penalty, dropping a
            # base score of 1.0 to 0.40 which fails the new min_score=0.25 gate.
            penalty  = min(0.65, sec_hits * 0.12)
            return max(0.0, base - penalty)
        return base

    if not keywords:
        return chunks[:max_chunks]

    scored = sorted(
        [(c, _score(c)) for c in chunks],
        key=lambda x: x[1], reverse=True,
    )
    filtered = [c for c, s in scored if s >= min_score]
    if len(filtered) < 2:
        filtered = [c for c, _ in scored[:2]]
    return filtered[:max_chunks]


def _call_groq(
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
    model: str = None,
) -> str:
    # Always re-load .env before reading key (handles restart scenarios)
    _load_env()
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        return (
            "[Groq API key not set. "
            f"Searched for .env starting from: {Path(__file__).resolve().parent.parent}. "
            "Make sure .env exists at your project root with GROQ_API_KEY=your_key, "
            "then restart Streamlit.]"
        )

    # ── Use urllib3 directly (avoids requests/urllib3 version mismatch) ──────
    try:
        import urllib3
    except ImportError:
        return "[urllib3 library not installed — run: pip install urllib3]"

    # Suppress urllib3 warnings globally
    urllib3.disable_warnings()

    user = user[:8000] if len(user) > 8000 else user

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    models_to_try = [model or GROQ_MODEL] + [
        m for m in _FALLBACK_MODELS if m != (model or GROQ_MODEL)
    ]

    # Create a single PoolManager (connection pool) — reuse across retries
    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=10, read=60),
        retries=urllib3.Retry(total=0),   # we handle retries ourselves
    )

    last_error = ""
    for try_model in models_to_try:
        body = {
            "model": try_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }
        try:
            encoded_body = json.dumps(body).encode("utf-8")
            resp = http.request(
                "POST",
                GROQ_URL,
                body=encoded_body,
                headers=headers,
            )

            if resp.status == 200:
                data = json.loads(resp.data.decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
            elif resp.status in (400, 404):
                last_error = f"{resp.status} with model {try_model}: {resp.data[:200].decode('utf-8', errors='replace')}"
                continue
            else:
                last_error = f"HTTP {resp.status}: {resp.data[:200].decode('utf-8', errors='replace')}"
                continue

        except Exception as exc:
            last_error = str(exc)
            continue

    return f"[Groq error: {last_error}]"


def groq_answer(query: str, context: str = "", mode: str = "finetuned") -> str:
    """
    Generate an answer via Groq.

    Patch C: max_tokens is mode-aware:
      RAG        → 700  (room for cited, structured answers)
      fine-tuned → 500  (180-240 word formal legal paragraphs)
      baseline   → 250  (100-160 word plain-English answers)

    Part 4.4: RAG user prompt numbers evidence chunks for inline citations.
    FIX v2: RAG prompt instructs model to use separate citation brackets only.
    """
    query   = (query   or "").strip()
    context = (context or "").strip()

    if mode == "baseline":
        system = _BASELINE_SYSTEM
        user   = f"Question: {query}"
        if context:
            user = f"Context:\n{context[:2000]}\n\nQuestion: {query}"

    elif mode == "rag":
        system = _RAG_SYSTEM

        if context:
            # Split into individual evidence chunks
            raw_chunks = [c.strip() for c in context.split("\n\n") if c.strip()]
            if not raw_chunks:
                raw_chunks = [context]

            # ── Final evidence filter before Groq sees anything ───────────────
            # Uses section-penalty: chunks dominated by specific section numbers
            # (e.g. "Section 354", "Section 509") get penalised for broad questions.
            is_broad = (
                _is_definitional_query(query)
                and not re.search(r'\bsection\s+\d+\b', query, re.IGNORECASE)
                and not re.search(r'\barticle\s+\d+\b', query, re.IGNORECASE)
            )
            relevant_chunks = _filter_evidence_chunks(
                raw_chunks, query,
                min_score=0.25,   # raised from 0.12 — prevents off-topic chunks reaching LLM
                max_chunks=5,
                penalise_sections=is_broad,
            )

            # ── Chunk diversity guard ─────────────────────────────────────────
            # Even after filtering, remaining chunks can all be about the same
            # specific sub-section (e.g. all 5 about §354). Hard-cap any single
            # section number to 2 appearances in the evidence sent to Groq.
            if is_broad and len(relevant_chunks) > 2:
                from collections import Counter as _C2
                _sec_re3 = re.compile(r'\bsection\s+(\d+\w*)\b', re.IGNORECASE)
                _sec_cnt: dict = {}
                _diverse_chunks = []
                for _chunk in relevant_chunks:
                    _hits = _sec_re3.findall(_chunk)
                    _dom = _C2(_hits).most_common(1)[0][0].lower() if _hits else None
                    if _dom is None:
                        _diverse_chunks.append(_chunk)
                    elif _sec_cnt.get(_dom, 0) < 2:
                        _diverse_chunks.append(_chunk)
                        _sec_cnt[_dom] = _sec_cnt.get(_dom, 0) + 1
                if _diverse_chunks:
                    relevant_chunks = _diverse_chunks

            if relevant_chunks:
                numbered = "\n\n".join(
                    f"[Evidence {i + 1}]\n{chunk[:700]}"
                    for i, chunk in enumerate(relevant_chunks)
                )

                broad_note = ""
                if is_broad:
                    broad_note = (
                        f"5. BROAD QUESTION RULE: '{query}' asks about the topic as a whole. "
                        "Give a complete overview. Do NOT make specific sub-sections the focus "
                        "unless the question explicitly asks for them.\n"
                    )

                user = (
                    f"Question: {query}\n\n"
                    f"Evidence:\n{numbered}\n\n"
                    "Instructions:\n"
                    f"1. Answer ONLY this question: '{query}'\n"
                    "2. If an evidence chunk is about a sub-topic NOT asked about, skip it.\n"
                    "3. Cite relevant evidence as [Evidence N] — separate brackets, never combined.\n"
                    "4. Use evidence to enrich your answer, not to redirect it.\n"
                    f"{broad_note}"
                    "Answer:"
                )
            else:
                # No relevant evidence — Groq answers from general legal knowledge
                user = (
                    f"Question: {query}\n\n"
                    "No specific evidence was retrieved. "
                    "Answer using your comprehensive Indian legal knowledge. "
                    "Give a thorough, accurate, structured answer.\n"
                    "Answer:"
                )
        else:
            user = (
                f"Question: {query}\n\n"
                "Answer using your comprehensive Indian legal knowledge. "
                "Give a thorough, accurate, structured answer.\n"
                "Answer:"
            )

    else:  # finetuned
        system = _FINETUNE_SYSTEM
        user   = f"Question: {query}"
        if context:
            user = f"Context:\n{context[:2500]}\n\nQuestion: {query}"

    # Mode-aware token budgets
    max_tok = 800 if mode == "rag" else 500 if mode == "finetuned" else 250
    return _call_groq(system, user, temperature=0.2, max_tokens=max_tok)


def groq_summarize(text: str, mode: str = "finetuned", evidence: str = "") -> str:
    text     = (text     or "").strip()
    evidence = (evidence or "").strip()

    if mode == "rag" and evidence:
        system = _RAG_SYSTEM
        ev_chunks = [c.strip() for c in evidence.split("\n\n") if c.strip()]
        if not ev_chunks:
            ev_chunks = [evidence]
        numbered = "\n\n".join(
            f"[Evidence {i + 1}]\n{chunk[:600]}"
            for i, chunk in enumerate(ev_chunks[:5])
        )
        user = (
            f"Question: Summarize this legal document.\n\n"
            f"Evidence:\n{numbered}\n\n"
            f"Document excerpt:\n{text[:1500]}\n\n"
            "Instructions:\n"
            "1. Your summary must be grounded in the evidence and document excerpt above.\n"
            "2. Use exact legal phrases and section numbers from the text — do not paraphrase loosely.\n"
            "3. Cite each evidence chunk separately by number: [Evidence 1] [Evidence 2] — never combine.\n"
            "4. Do NOT add facts not present in the evidence or document excerpt.\n"
            "Answer:"
        )
    elif mode == "baseline":
        system = _BASELINE_SYSTEM
        user = f"Summarize the following text in plain language:\n\n{text[:3000]}"
    else:
        system = _FINETUNE_SYSTEM
        user = (
            "Summarize the following Indian legal document using the structured format:\n"
            "1. Key Facts\n2. Legal Issue\n3. Court's Reasoning\n4. Outcome\n\n"
            f"{text[:3000]}"
        )

    max_tok = 700 if mode == "rag" else 500 if mode == "finetuned" else 250
    return _call_groq(system, user, temperature=0.1, max_tokens=max_tok)


def groq_irac(query: str, context: str = "") -> str:
    system = _FINETUNE_SYSTEM
    ctx_part = f"\nContext:\n{context[:2000]}\n" if context else ""
    user = (
        f"{ctx_part}\n"
        f"Provide a structured IRAC (Issue, Rule, Application, Conclusion) "
        f"legal analysis for the following question under Indian law:\n\n"
        f"{query}\n\n"
        "Format your response with clear headings:\n"
        "Issue:\nRule:\nApplication:\nConclusion:"
    )
    return _call_groq(system, user, temperature=0.2, max_tokens=700)