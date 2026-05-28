"""
ui/templates.py
───────────────────────────────────────────────────────────────────────────────
LegalAId — HTML / CSS rendering helpers for Streamlit.

All public functions return raw HTML strings.
Inject them with:  st.markdown(html, unsafe_allow_html=True)

Public API (what app.py should call)
─────────────────────────────────────
  page_styles()                       → inject once in st.markdown at app start
  page_header()                       → top banner with app title + subtitle
  mode_header(mode)                   → column header card (Baseline / RAG / Fine-tuned)
  answer_card(mode, answer,           → full answer card with optional score badge
              score, threshold,
              source_label)
  no_doc_badge()                      → amber warning shown in RAG col when NO PDF loaded
  doc_uploaded_badge(filename)        → green notice shown when uploaded PDF IS being used
  groundedness_badge(score,threshold) → coloured pill for groundedness score
  highlight_citations(text)           → wraps [Evidence N] in styled <span> tags
                                        also splits combined [Evidence X, Y] into separate spans
  evidence_panel(chunks)              → collapsible evidence source panel for RAG tab
  irac_card(irac_text)                → structured IRAC analysis card
  summary_card(mode, summary_text)    → summary result card
  error_card(message)                 → red error / warning card
  info_card(message)                  → blue informational card
  spinner_html(label)                 → inline CSS spinner with label

FIX v2 changes:
  - Added doc_uploaded_badge(filename) for when PDF IS loaded
  - no_doc_badge() is now corpus-only (unchanged text, clearer usage)
  - highlight_citations() now splits combined [Evidence X, Y] before styling
"""

from __future__ import annotations
import re

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  (mirrors config.py ACCENT_COLOR)
# ─────────────────────────────────────────────────────────────────────────────

_ACCENT      = "#1a3a5c"   # deep navy  (brand / Fine-tuned)
_RAG_COLOR   = "#1b5e38"   # forest green (RAG / grounded)
_BASE_COLOR  = "#4a4a4a"   # charcoal    (Baseline / neutral)
_WARN_BG     = "#fff8e1"   # pale amber  (citations / warnings)
_WARN_BORDER = "#f0a500"   # amber border
_ERR_BG      = "#fdecea"
_ERR_BORDER  = "#d32f2f"
_INFO_BG     = "#e8f4fd"
_INFO_BORDER = "#1565c0"
_OK_BG       = "#e8f5e9"   # FIX v2: green bg for doc-uploaded badge
_OK_BORDER   = "#388e3c"   # FIX v2: green border
_CARD_BG     = "#ffffff"
_SUBTLE_BG   = "#f8f9fa"
_BORDER      = "#dee2e6"
_TEXT        = "#212529"
_MUTED       = "#6c757d"


# ─────────────────────────────────────────────────────────────────────────────
# Mode metadata
# ─────────────────────────────────────────────────────────────────────────────

_MODE_META = {
    "baseline": {
        "label":       "Baseline",
        "icon":        "💬",
        "color":       _BASE_COLOR,
        "light_bg":    "#f4f4f4",
        "border":      "#cccccc",
        "description": "General LLM · No retrieval · No legal specialisation",
        "badge_text":  "BASELINE",
    },
    "finetuned": {
        "label":       "Fine-tuned",
        "icon":        "⚖️",
        "color":       _ACCENT,
        "light_bg":    "#eef3f8",
        "border":      "#a8c0d6",
        "description": "Domain-adapted · Indian law specialist · Structured output",
        "badge_text":  "FINE-TUNED",
    },
    "rag": {
        "label":       "RAG",
        "icon":        "📚",
        "color":       _RAG_COLOR,
        "light_bg":    "#edf7f1",
        "border":      "#a3c9b4",
        "description": "Retrieval-Augmented · Evidence-grounded · Inline citations",
        "badge_text":  "RAG",
    },
}

_DEFAULT_META = {
    "label":       "Answer",
    "icon":        "🤖",
    "color":       _ACCENT,
    "light_bg":    "#f8f9fa",
    "border":      _BORDER,
    "description": "",
    "badge_text":  "AI",
}


def _meta(mode: str) -> dict:
    return _MODE_META.get((mode or "").lower().strip(), _DEFAULT_META)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Global page styles — inject ONCE at app startup
# ─────────────────────────────────────────────────────────────────────────────

def page_styles() -> str:
    """
    Returns a <style> block to be injected via st.markdown at app startup.
    Covers answer cards, citations, badges, evidence panels, spinners.
    """
    return f"""
<style>
/* ── Reset & base ─────────────────────────────────────────────────────── */
.legalaid-root * {{
    box-sizing: border-box;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}}

/* ── Page header ──────────────────────────────────────────────────────── */
.la-page-header {{
    background: linear-gradient(135deg, {_ACCENT} 0%, #2a5480 100%);
    color: #fff;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 16px rgba(26,58,92,.25);
}}
.la-page-header h1 {{
    margin: 0 0 0.25rem 0;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.3px;
}}
.la-page-header p {{
    margin: 0;
    opacity: .85;
    font-size: .95rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}}

/* ── Mode column header ────────────────────────────────────────────────── */
.la-mode-header {{
    padding: .75rem 1rem;
    border-radius: 10px 10px 0 0;
    border-bottom: 3px solid currentColor;
    margin-bottom: 0;
}}
.la-mode-header .la-mode-badge {{
    display: inline-block;
    font-size: .65rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: .15rem .5rem;
    border-radius: 4px;
    background: rgba(255,255,255,.25);
    margin-right: .5rem;
    vertical-align: middle;
}}
.la-mode-header .la-mode-title {{
    font-size: 1.05rem;
    font-weight: 700;
    vertical-align: middle;
}}
.la-mode-header .la-mode-desc {{
    font-size: .75rem;
    margin-top: .2rem;
    opacity: .8;
}}

/* ── Answer card ──────────────────────────────────────────────────────── */
.la-answer-card {{
    background: {_CARD_BG};
    border: 1px solid {_BORDER};
    border-radius: 0 0 10px 10px;
    padding: 1.1rem 1.2rem 1rem 1.2rem;
    min-height: 140px;
    line-height: 1.65;
    font-size: .92rem;
    color: {_TEXT};
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
    position: relative;
}}
.la-answer-card.la-rag {{
    border-top: none;
    border-color: {_MODE_META['rag']['border']};
}}
.la-answer-card.la-finetuned {{
    border-top: none;
    border-color: {_MODE_META['finetuned']['border']};
}}
.la-answer-card.la-baseline {{
    border-top: none;
    border-color: #cccccc;
}}

/* ── Inline citation chips ─────────────────────────────────────────────── */
.la-citation {{
    display: inline-block;
    background: {_WARN_BG};
    color: #7a4800;
    border: 1px solid {_WARN_BORDER};
    border-radius: 4px;
    padding: 0 .35em;
    font-size: .8em;
    font-weight: 600;
    vertical-align: baseline;
    cursor: default;
    white-space: nowrap;
    transition: background .15s;
}}
.la-citation:hover {{
    background: #ffe082;
}}

/* ── "General legal knowledge" label ──────────────────────────────────── */
.la-general-label {{
    color: {_MUTED};
    font-style: italic;
    font-size: .85em;
}}

/* ── Groundedness badge ────────────────────────────────────────────────── */
.la-score-badge {{
    display: inline-flex;
    align-items: center;
    gap: .3rem;
    font-size: .72rem;
    font-weight: 600;
    padding: .2rem .55rem;
    border-radius: 20px;
    margin-top: .65rem;
    border: 1px solid transparent;
}}
.la-score-badge.la-score-good {{
    background: #e8f5e9;
    color: #2e7d32;
    border-color: #a5d6a7;
}}
.la-score-badge.la-score-warn {{
    background: {_WARN_BG};
    color: #7a4800;
    border-color: {_WARN_BORDER};
}}
.la-score-badge.la-score-low {{
    background: {_ERR_BG};
    color: #b71c1c;
    border-color: #ef9a9a;
}}

/* ── No-document warning badge ─────────────────────────────────────────── */
.la-no-doc-badge {{
    display: flex;
    align-items: flex-start;
    gap: .5rem;
    background: {_WARN_BG};
    border: 1px solid {_WARN_BORDER};
    border-radius: 8px;
    padding: .6rem .8rem;
    font-size: .8rem;
    color: #6d4300;
    margin-bottom: .75rem;
    line-height: 1.4;
}}
.la-no-doc-badge .la-no-doc-icon {{
    font-size: 1rem;
    flex-shrink: 0;
    margin-top: .05rem;
}}

/* ── FIX v2: Doc-uploaded green badge ──────────────────────────────────── */
.la-doc-uploaded-badge {{
    display: flex;
    align-items: flex-start;
    gap: .5rem;
    background: {_OK_BG};
    border: 1px solid {_OK_BORDER};
    border-radius: 8px;
    padding: .6rem .8rem;
    font-size: .8rem;
    color: #1b5e20;
    margin-bottom: .75rem;
    line-height: 1.4;
}}
.la-doc-uploaded-badge .la-doc-icon {{
    font-size: 1rem;
    flex-shrink: 0;
    margin-top: .05rem;
}}

/* ── Evidence panel ────────────────────────────────────────────────────── */
.la-evidence-panel {{
    background: {_SUBTLE_BG};
    border: 1px solid {_BORDER};
    border-radius: 8px;
    margin-top: .8rem;
    overflow: hidden;
}}
.la-evidence-panel summary {{
    padding: .55rem .9rem;
    cursor: pointer;
    font-size: .8rem;
    font-weight: 600;
    color: {_ACCENT};
    user-select: none;
    list-style: none;
}}
.la-evidence-panel summary::-webkit-details-marker {{ display: none; }}
.la-evidence-panel summary::before {{
    content: "▶ ";
    font-size: .65rem;
    transition: transform .2s;
}}
.la-evidence-panel[open] summary::before {{ content: "▼ "; }}
.la-evidence-chunk {{
    padding: .5rem .9rem .7rem .9rem;
    border-top: 1px solid {_BORDER};
    font-size: .78rem;
    line-height: 1.55;
    color: {_TEXT};
}}
.la-evidence-chunk:last-child {{ padding-bottom: .9rem; }}
.la-chunk-label {{
    font-weight: 700;
    color: {_RAG_COLOR};
    font-size: .72rem;
    letter-spacing: .5px;
    text-transform: uppercase;
    margin-bottom: .2rem;
}}
.la-chunk-source {{
    font-size: .7rem;
    color: {_MUTED};
    font-style: italic;
    margin-top: .2rem;
}}

/* ── IRAC card ─────────────────────────────────────────────────────────── */
.la-irac-card {{
    background: {_CARD_BG};
    border-left: 4px solid {_ACCENT};
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    font-size: .9rem;
    line-height: 1.7;
    color: {_TEXT};
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
}}
.la-irac-label {{
    font-weight: 700;
    color: {_ACCENT};
    font-size: .75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: .7rem 0 .15rem 0;
}}
.la-irac-label:first-child {{ margin-top: 0; }}

/* ── Summary card ──────────────────────────────────────────────────────── */
.la-summary-card {{
    background: {_CARD_BG};
    border: 1px solid {_BORDER};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: .92rem;
    line-height: 1.65;
    color: {_TEXT};
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
}}

/* ── Error / info cards ────────────────────────────────────────────────── */
.la-error-card {{
    background: {_ERR_BG};
    border: 1px solid {_ERR_BORDER};
    border-radius: 8px;
    padding: .75rem 1rem;
    font-size: .88rem;
    color: #7f0000;
    display: flex;
    gap: .5rem;
    align-items: flex-start;
}}
.la-info-card {{
    background: {_INFO_BG};
    border: 1px solid {_INFO_BORDER};
    border-radius: 8px;
    padding: .75rem 1rem;
    font-size: .88rem;
    color: #0d3b6e;
    display: flex;
    gap: .5rem;
    align-items: flex-start;
}}

/* ── Inline spinner ────────────────────────────────────────────────────── */
@keyframes la-spin {{ to {{ transform: rotate(360deg); }} }}
.la-spinner-wrap {{
    display: flex;
    align-items: center;
    gap: .6rem;
    padding: .6rem 0;
    color: {_MUTED};
    font-size: .88rem;
}}
.la-spinner {{
    width: 18px;
    height: 18px;
    border: 2.5px solid #d0d0d0;
    border-top-color: {_ACCENT};
    border-radius: 50%;
    animation: la-spin .8s linear infinite;
    flex-shrink: 0;
}}

/* ── Utility ───────────────────────────────────────────────────────────── */
.la-muted  {{ color: {_MUTED}; font-size: .8rem; }}
.la-spacer {{ height: .5rem; }}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Page header
# ─────────────────────────────────────────────────────────────────────────────

def page_header(
    title: str = "LegalAId",
    subtitle: str = "Summarise · Answer · Argue · Cite",
) -> str:
    """Top-of-page gradient banner with title and subtitle."""
    return f"""
<div class="legalaid-root">
  <div class="la-page-header">
    <h1>⚖️ {title}</h1>
    <p>{subtitle}</p>
  </div>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Mode column header
# ─────────────────────────────────────────────────────────────────────────────

def mode_header(mode: str) -> str:
    """
    Returns a coloured header card for a comparison column.

    Usage:
        st.markdown(mode_header("rag"), unsafe_allow_html=True)
    """
    m = _meta(mode)
    color = m["color"]
    return f"""
<div class="legalaid-root">
  <div class="la-mode-header"
       style="background:{m['light_bg']};color:{color};border-color:{color};">
    <span class="la-mode-badge"
          style="background:{color};color:#fff;">{m['badge_text']}</span>
    <span class="la-mode-title">{m['icon']} {m['label']}</span>
    <div class="la-mode-desc">{m['description']}</div>
  </div>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Citation highlighting
# ─────────────────────────────────────────────────────────────────────────────

# Matches individual citations: [Evidence 3]
_CITATION_RE = re.compile(r"\[Evidence\s+(\d+)\]", re.IGNORECASE)

# FIX v2: Matches combined citations: [Evidence 3, Evidence 4] or [Evidence 3, 4]
_COMBINED_CITATION_RE = re.compile(
    r"\[Evidence\s+(\d+)(?:\s*,\s*(?:Evidence\s+)?(\d+))+\]",
    re.IGNORECASE,
)

_GENERAL_RE  = re.compile(r"\(General legal knowledge\)", re.IGNORECASE)


def _expand_combined_citations(text: str) -> str:
    """
    FIX v2: Split combined citations into separate ones before highlighting.
    [Evidence 3, Evidence 4]  →  [Evidence 3] [Evidence 4]
    [Evidence 3, 4]           →  [Evidence 3] [Evidence 4]
    """
    def replacer(m):
        full = m.group(0)
        # Extract all numbers from the combined bracket
        numbers = re.findall(r"\d+", full)
        return " ".join(f"[Evidence {n}]" for n in numbers)

    return _COMBINED_CITATION_RE.sub(replacer, text)


def highlight_citations(text: str) -> str:
    """
    Convert [Evidence N] markers into styled amber chips and
    '(General legal knowledge)' into muted italic spans.

    FIX v2: Also splits combined citations like [Evidence 3, Evidence 4]
    into separate chips before styling.

    Safe to call on plain text even if no citations are present.
    """
    if not text:
        return ""
    # Escape HTML entities first (prevent XSS from LLM output)
    text = (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    # FIX v2: Split combined citations first
    text = _expand_combined_citations(text)
    # Highlight individual citations
    text = _CITATION_RE.sub(
        lambda m: f'<span class="la-citation">[Evidence {m.group(1)}]</span>',
        text,
    )
    # Style general knowledge labels
    text = _GENERAL_RE.sub(
        '<span class="la-general-label">(General legal knowledge)</span>',
        text,
    )
    # Preserve line breaks
    text = text.replace("\n", "<br>")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Groundedness badge
# ─────────────────────────────────────────────────────────────────────────────

def groundedness_badge(score: float | None, threshold: float = 0.35) -> str:
    """
    Returns a coloured pill showing the groundedness score.

    score >= threshold+0.2  → green  (well-grounded)
    score >= threshold       → amber  (marginal)
    score <  threshold       → red    (low groundedness)
    score is None            → grey   (not computed)
    """
    if score is None:
        return (
            '<span class="la-score-badge" '
            'style="background:#f0f0f0;color:#888;border-color:#ccc;">'
            '📊 Score: N/A</span>'
        )
    pct = round(score * 100)
    if score >= threshold + 0.20:
        cls, icon = "la-score-good", "✅"
        label = f"Grounded {pct}%"
    elif score >= threshold:
        cls, icon = "la-score-warn", "⚠️"
        label = f"Marginal {pct}%"
    else:
        cls, icon = "la-score-low", "🔴"
        label = f"Low score {pct}%"
    return f'<span class="la-score-badge {cls}">{icon} {label}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# 6a.  "No document" warning badge  (corpus-only fallback)
# ─────────────────────────────────────────────────────────────────────────────

def no_doc_badge() -> str:
    """
    Amber notice shown in the RAG column when NO PDF has been uploaded.
    Use this when RAG is searching the static legal corpus only.
    Do NOT show this when a PDF is uploaded — use doc_uploaded_badge() instead.
    """
    return """
<div class="legalaid-root">
  <div class="la-no-doc-badge">
    <span class="la-no-doc-icon">📂</span>
    <span>
      <strong>Using corpus knowledge</strong> — upload a PDF for
      document-specific answers. RAG is currently searching the
      static legal corpus only.
    </span>
  </div>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# 6b.  FIX v2: "Document uploaded" green badge
# ─────────────────────────────────────────────────────────────────────────────

def doc_uploaded_badge(filename: str = "") -> str:
    """
    FIX v2: Green notice shown in the RAG column when a PDF IS uploaded
    and its chunks are being used as primary evidence.

    Usage in app.py:
        if doc_chunks:
            st.markdown(doc_uploaded_badge(uploaded_file.name), unsafe_allow_html=True)
        else:
            st.markdown(no_doc_badge(), unsafe_allow_html=True)
    """
    name_part = f" <strong>{_esc(filename)}</strong>" if filename else ""
    return f"""
<div class="legalaid-root">
  <div class="la-doc-uploaded-badge">
    <span class="la-doc-icon">📄</span>
    <span>
      Answering from uploaded document{name_part} — evidence is grounded
      in your PDF, not the general corpus.
    </span>
  </div>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Full answer card
# ─────────────────────────────────────────────────────────────────────────────

def answer_card(
    mode: str,
    answer: str,
    score: float | None = None,
    threshold: float = 0.35,
    source_label: str | None = None,
) -> str:
    """
    Renders a complete answer card for one comparison column.

    Parameters
    ----------
    mode         : "baseline" | "finetuned" | "rag"
    answer       : raw answer text from the LLM
    score        : groundedness score (float 0-1) — shown for RAG only
    threshold    : GROUNDEDNESS_THRESHOLD from config (default 0.35)
    source_label : optional small label, e.g. "Source: Uploaded PDF"

    Usage
    -----
        st.markdown(
            answer_card("rag", answer_text, score=0.72),
            unsafe_allow_html=True,
        )
    """
    m        = _meta(mode)
    mode_cls = mode.lower() if mode.lower() in ("baseline", "finetuned", "rag") else ""

    # For RAG: highlight citations in answer text
    if mode.lower() == "rag":
        rendered_answer = highlight_citations(answer or "*(No answer generated)*")
    else:
        rendered_answer = (
            (answer or "*(No answer generated)*")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )

    # Footer row: score badge + optional source label
    footer_parts = []
    if mode.lower() == "rag":
        footer_parts.append(groundedness_badge(score, threshold))
    if source_label:
        footer_parts.append(
            f'<span class="la-muted" style="margin-left:.5rem;">📄 {source_label}</span>'
        )
    footer_html = (
        f'<div style="margin-top:.5rem;">{" ".join(footer_parts)}</div>'
        if footer_parts else ""
    )

    return f"""
<div class="legalaid-root">
  <div class="la-answer-card la-{mode_cls}"
       style="border-top:3px solid {m['color']};">
    {rendered_answer}
    {footer_html}
  </div>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Evidence panel (for Tab 2 / Sources tab)
# ─────────────────────────────────────────────────────────────────────────────

def evidence_panel(chunks: list[dict], open_by_default: bool = False) -> str:
    """
    Renders a collapsible list of retrieved evidence chunks.

    Each chunk dict should have:
        text   : str  — the chunk body
        source : str  — e.g. "Indian Penal Code §299" or "Uploaded PDF p.3"

    Usage
    -----
        st.markdown(evidence_panel(retrieved_chunks), unsafe_allow_html=True)
    """
    if not chunks:
        return info_card("No evidence chunks were retrieved for this query.")

    open_attr = "open" if open_by_default else ""
    items_html = ""
    for i, chunk in enumerate(chunks, 1):
        body   = (chunk.get("text") or "").strip()
        source = (chunk.get("source") or "").strip()

        body_esc = (
            body
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        source_html = (
            f'<div class="la-chunk-source">📄 {source}</div>' if source else ""
        )
        items_html += f"""
    <div class="la-evidence-chunk">
      <div class="la-chunk-label">Evidence {i}</div>
      {body_esc}
      {source_html}
    </div>"""

    count_label = f"{len(chunks)} chunk{'s' if len(chunks) != 1 else ''} retrieved"
    return f"""
<div class="legalaid-root">
  <details class="la-evidence-panel" {open_attr}>
    <summary>📚 Retrieved Evidence — {count_label}</summary>
    {items_html}
  </details>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# 9.  IRAC analysis card
# ─────────────────────────────────────────────────────────────────────────────

_IRAC_SECTIONS = ["Issue", "Rule", "Application", "Conclusion"]
_IRAC_RE = re.compile(
    r"(Issue|Rule|Application|Conclusion)\s*:\s*",
    re.IGNORECASE,
)


def irac_card(irac_text: str) -> str:
    """
    Parses and renders an IRAC analysis with coloured section headings.
    Works whether the model used 'Issue:' headers or plain prose.
    """
    if not irac_text:
        return error_card("No IRAC analysis was generated.")

    # Try to split on section headers
    parts = _IRAC_RE.split(irac_text)

    if len(parts) > 1:
        # parts = ['preamble', 'Issue', 'body', 'Rule', 'body', ...]
        html_inner = ""
        it = iter(parts)
        preamble = next(it, "").strip()
        if preamble:
            html_inner += f'<p style="margin-top:0">{_esc(preamble)}</p>'
        while True:
            label = next(it, None)
            body  = next(it, None)
            if label is None:
                break
            body_html = _esc(body or "").replace("\n", "<br>")
            html_inner += (
                f'<div class="la-irac-label">▸ {label.title()}</div>'
                f'<div style="margin-bottom:.5rem;">{body_html}</div>'
            )
    else:
        # No headers found — render as plain text
        html_inner = _esc(irac_text).replace("\n", "<br>")

    return f"""
<div class="legalaid-root">
  <div class="la-irac-card">
    {html_inner}
  </div>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Summary card
# ─────────────────────────────────────────────────────────────────────────────

def summary_card(mode: str, summary_text: str) -> str:
    """
    Renders a document summary result card.
    For RAG mode, citations in the summary text are highlighted.
    """
    if not summary_text:
        return error_card("No summary was generated.")

    m = _meta(mode)

    if mode.lower() == "rag":
        body = highlight_citations(summary_text)
    else:
        body = _esc(summary_text).replace("\n", "<br>")

    mode_label = m["label"]
    icon       = m["icon"]
    color      = m["color"]

    return f"""
<div class="legalaid-root">
  <div class="la-summary-card" style="border-top:3px solid {color};">
    <div style="font-size:.72rem;font-weight:700;color:{color};
                text-transform:uppercase;letter-spacing:1px;margin-bottom:.5rem;">
      {icon} {mode_label} Summary
    </div>
    {body}
  </div>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Error / info utility cards
# ─────────────────────────────────────────────────────────────────────────────

def error_card(message: str) -> str:
    """Red inline error / warning card."""
    return f"""
<div class="legalaid-root">
  <div class="la-error-card">
    <span>⚠️</span><span>{_esc(message)}</span>
  </div>
</div>
"""


def info_card(message: str) -> str:
    """Blue informational notice card."""
    return f"""
<div class="legalaid-root">
  <div class="la-info-card">
    <span>ℹ️</span><span>{_esc(message)}</span>
  </div>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# 12.  Inline loading spinner
# ─────────────────────────────────────────────────────────────────────────────

def spinner_html(label: str = "Generating answer…") -> str:
    """CSS-only spinner — inject while waiting for LLM response."""
    return f"""
<div class="legalaid-root">
  <div class="la-spinner-wrap">
    <div class="la-spinner"></div>
    <span>{_esc(label)}</span>
  </div>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _esc(text: str) -> str:
    """Minimal HTML escaping for LLM output."""
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test — run this file directly to verify output
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_answer = (
        "Under Section 300 IPC, culpable homicide amounts to murder when the act "
        "is done with the intention of causing death. [Evidence 1] "
        "The Supreme Court clarified that the accused must have premeditated intent. "
        "[Evidence 2] [Evidence 3] "
        "(General legal knowledge) Grave provocation can be a partial defence "
        "reducing the charge to culpable homicide not amounting to murder."
    )

    # FIX v2: test combined citation splitting
    combined_test = "This is established law. [Evidence 3, Evidence 4] It is also confirmed. [Evidence 1, 2]"
    print("Combined citation splitting test:")
    print(highlight_citations(combined_test))
    print()

    sample_chunks = [
        {
            "text": "Section 300 IPC defines murder as culpable homicide with "
                    "intention to cause death or bodily injury likely to cause death.",
            "source": "Indian Penal Code §300",
        },
        {
            "text": "The fundamental rights are protected under Part III of the Indian Constitution.",
            "source": "Uploaded PDF — p.14",
        },
    ]

    print(page_styles()[:120], "...\n[CSS truncated]\n")
    print(mode_header("rag"))
    print("--- no_doc_badge (corpus only) ---")
    print(no_doc_badge())
    print("--- doc_uploaded_badge (PDF loaded) ---")
    print(doc_uploaded_badge("Constitution.pdf"))
    print(answer_card("rag", sample_answer, score=0.71, threshold=0.35))
    print(evidence_panel(sample_chunks, open_by_default=True))
    print(groundedness_badge(0.71, 0.35))
    print(groundedness_badge(0.30, 0.35))
    print(irac_card(
        "Issue: Whether the accused had premeditated intent.\n"
        "Rule: Section 300 IPC requires intention or knowledge.\n"
        "Application: The accused acted in the heat of the moment.\n"
        "Conclusion: Charge reduced to Section 304 IPC."
    ))
    print("✅  templates.py smoke-test passed.")