"""
normalize.py
------------
Text and table normalization utilities for OCR benchmark evaluation.

Functions:
  normalize_ocr_text()           — full pipeline for CER/WER (from UET notebooks)
  normalize_cell()               — light normalization for table cells
  normalize()                    — table cell with Markdown→canonical conversion
  normalize_ws()                 — whitespace collapse, optional lowercase
  normalize_latex()              — LaTeX formula normalization
  flatten_markdown_tables_for_text() — flatten MD pipe tables to plain text
  normalize_for_text_benchmark() — legacy alias for normalize_ocr_text()
  normalize_for_nwer()           — legacy: lowercase + strip punctuation
"""

from __future__ import annotations

import re
import html as _html_module
import math
import unicodedata
from typing import Any

# ── Config ────────────────────────────────────────────────────
COLLAPSE_WHITESPACE   = True
LOWERCASE_FOR_METRICS = False


# ─────────────────────────────────────────────────────────────
# flatten_markdown_tables_for_text
# ─────────────────────────────────────────────────────────────

def flatten_markdown_tables_for_text(s: str) -> str:
    """
    Flatten Markdown pipe tables into plain OCR text.
    Used by normalize_ocr_text() for CER/WER — NOT for table structure eval.

    Examples:
        | A | B |        -> A B
        |---|---|        -> removed
        | 1. X | 7. Y | -> 1. X 7. Y
    """
    table_separator_re = re.compile(
        r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$"
    )
    out = []
    for line in str(s).splitlines():
        raw = line.strip()
        if not raw:
            continue
        raw = re.sub(r"^\s*[-+*]\s+(?=\|)", "", raw)
        if table_separator_re.fullmatch(raw):
            continue
        if "|" in raw:
            cells = raw.strip("|").split("|")
            cleaned = []
            for cell in cells:
                cell = cell.strip()
                if not cell:
                    continue
                cell = re.sub(r"<\s*br\s*/?\s*>", " ", cell, flags=re.IGNORECASE)
                if re.fullmatch(r":?-{3,}:?", cell):
                    continue
                cleaned.append(cell)
            if cleaned:
                out.append(" ".join(cleaned))
            continue
        out.append(raw)
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────
# normalize_ocr_text  (primary normalizer, from UET notebooks)
# ─────────────────────────────────────────────────────────────

def normalize_ocr_text(
    value: Any,
    *,
    lowercase: bool = False,
    ignore_soft_punctuation: bool = True,
) -> str:
    """
    Full pipeline normalizer for Markdown/HTML OCR outputs.
    Used for CER/WER on readable text content (not table structure).

    Key behaviour:
    - Removes Markdown/HTML/layout syntax
    - Flattens Markdown pipe tables to text
    - Removes image placeholders
    - Preserves Vietnamese diacritics
    - Preserves decimal commas (0,5) and thousands dots (2.214.394)
    - Optionally ignores soft punctuation (: and ;)
    """
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass

    s = str(value)

    # 1. Basic cleanup
    s = _html_module.unescape(s)
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)
    s = s.replace("\u00a0", " ").replace("\u202f", " ").replace("\u00ad", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Protect URLs
    s = re.sub(r"<(https?://[^>\s]+)>", r"\1", s)
    s = re.sub(r"<([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})>", r"\1", s)
    protected: dict[str, str] = {}

    def protect(pattern: str, text: str, flags: int = 0) -> str:
        def repl(m: re.Match) -> str:
            key = f" PROTECTEDTOKEN{len(protected)} "
            protected[key.strip()] = m.group(0)
            return key
        return re.sub(pattern, repl, text, flags=flags)

    s = protect(r"https?://[^\s)>\]]+", s)

    # 3. Remove images
    s = re.sub(r"<img\b[^>]*>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", s)

    # 4. HTML to text
    s = re.sub(r"<\s*br\s*/?\s*>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"</\s*(p|div|tr|li|h[1-6]|table|section)\s*>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<\s*(p|div|tr|li|h[1-6]|table|section)\b[^>]*>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"</?\s*(b|strong|i|em|u|span|font|code|sup|sub)\b[^>]*>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", " ", s)

    # 5. Unescape Markdown escapes
    s = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|])", r"\1", s)
    s = protect(r"\(\s*\*+\s*\)", s)

    # 6. Markdown links, headings, blockquotes, bullets
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"```[A-Za-z0-9_-]*", " ", s)
    s = s.replace("```", " ")
    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", s)
    s = re.sub(r"(?m)^\s*>\s*", "", s)
    s = re.sub(r"(?m)^\s*[-*+]\s+(?=\S)", "", s)
    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", s)
    s = re.sub(r"(?<!\S)#{1,6}(?!\S)", " ", s)
    s = re.sub(r"(?<!\S)[*_]{1,3}(?!\S)", " ", s)

    # 7. Flatten Markdown tables
    s = flatten_markdown_tables_for_text(s)

    # 8. Markdown emphasis stripping
    s = re.sub(r"(\*\*|__)(.*?)\1", r"\2", s)
    s = re.sub(r"(?<!\S)\*([^\s*][^*\n]{0,120}?[^\s*])\*(?!\S)", r"\1", s)
    s = re.sub(r"(?<!\S)_([^\s_][^_\n]{0,120}?[^\s_])_(?!\S)", r"\1", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"[*_]{1,3}", " ", s)
    s = re.sub(r"(?<!\S)#{1,6}(?!\S)", " ", s)
    s = re.sub(r"[■□▪▫●○◦•∙·]", "", s)

    # 9. Normalize layout separators
    for old, new in [("•"," "),("–","-"),("—","-"),("／","/"),(
            "·"," "),("∙"," "),("●"," "),("•"," ")]:
        s = s.replace(old, new)

    # 10. Number-aware punctuation
    s = re.sub(r"(?<=\d),\s+(?=\d)", ",", s)
    s = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", s)
    s = re.sub(r"\s+%", "%", s)
    if ignore_soft_punctuation:
        s = re.sub(r"(?<!\d)\s*[:;]\s*(?!\d)", " ", s)
    else:
        s = re.sub(r"\s*;\s*", " ; ", s)
        s = re.sub(r"\s*:\s*", ": ", s)
        s = re.sub(r"(?<=\d):\s+(?=\d)", ":", s)
    s = re.sub(r"(?<!\d),\s*(?!\d)", ", ", s)
    s = re.sub(r"\s+([,.;:%])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)

    # 11. Restore protected tokens
    for key, original in protected.items():
        s = s.replace(key, original)

    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(?:\.\s*){3,}", " ", s)

    if lowercase:
        s = s.lower()

    return s


# ─────────────────────────────────────────────────────────────
# Table-cell normalizers
# ─────────────────────────────────────────────────────────────

def normalize_ws(s: str) -> str:
    """Whitespace collapse + optional lowercase."""
    s = _html_module.unescape(str(s or ""))
    if COLLAPSE_WHITESPACE:
        s = re.sub(r"\s+", " ", s).strip()
    if LOWERCASE_FOR_METRICS:
        s = s.lower()
    return s


def normalize(s: str) -> str:
    """
    Normalize table-cell content while preserving meaningful inline Markdown.
    Converts HTML italic/bold/code tags to Markdown equivalents.
    """
    s = _html_module.unescape(str(s or ""))
    s = re.sub(r"<((?:https?://|www\.)[^>\s]+)>", r"\1", s)
    s = re.sub(r"<([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})>", r"\1", s)
    s = re.sub(r"<\s*(i|em)\s*>(.*?)<\s*/\s*\1\s*>", r"*\2*", s, flags=re.IGNORECASE|re.DOTALL)
    s = re.sub(r"<\s*(b|strong)\s*>(.*?)<\s*/\s*\1\s*>", r"**\2**", s, flags=re.IGNORECASE|re.DOTALL)
    s = re.sub(r"<\s*code\s*>(.*?)<\s*/\s*code\s*>", r"`\1`", s, flags=re.IGNORECASE|re.DOTALL)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"__(.*?)__", r"**\1**", s, flags=re.DOTALL)
    s = re.sub(r"(?<!\w)_(?!_)(.*?)(?<!_)_(?!\w)", r"*\1*", s, flags=re.DOTALL)
    s = re.sub(r"\\([|])", r"\1", s)
    s = re.sub(r"(?<=\|)\s*:?-{3,}:?\s*(?=\|)", "---", s)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
    return normalize_ws(s)


def normalize_cell(s: str) -> str:
    """Light normalization for table cell content (strips markup, keeps text)."""
    s = _html_module.unescape(str(s or ""))
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\*\*|__|\*|_|`", "", s)
    s = re.sub(r"\\([|])", r"\1", s)
    return normalize_ws(s)


def normalize_latex(s: str) -> str:
    """Normalize LaTeX formula string for comparison."""
    s = _html_module.unescape(str(s or ""))
    s = s.strip()
    s = re.sub(r"\\(left|right)\s*", "", s)
    s = re.sub(r"\s+", "", s)
    return s


# ─────────────────────────────────────────────────────────────
# Legacy aliases (backward compatibility)
# ─────────────────────────────────────────────────────────────

def normalize_for_text_benchmark(text: str) -> str:
    """Legacy alias for normalize_ocr_text()."""
    return normalize_ocr_text(text)


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_for_nwer(text: str) -> str:
    """
    Legacy nWER normalizer: normalize_ocr_text + lowercase + strip punctuation.
    """
    text = normalize_ocr_text(text)
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
