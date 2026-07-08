"""
Text normalization utilities used before computing metrics.

normalize_for_text_benchmark() — general normalization for CER/WER evaluation.
normalize_for_nwer()           — strip punctuation + lowercase for nWER.
"""

from __future__ import annotations
import re
import unicodedata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_unicode(text: str) -> str:
    """NFC normalize to ensure consistent Unicode representation."""
    return unicodedata.normalize("NFC", text)


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Main normalizers
# ---------------------------------------------------------------------------

def normalize_for_text_benchmark(text: str) -> str:
    """
    General normalization applied to both GT and hypothesis before scoring.
    - NFC unicode normalization
    - Strip leading/trailing whitespace
    - Collapse internal whitespace to single space
    - Normalize line endings to space
    """
    if not text:
        return ""
    text = _normalize_unicode(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", " ")
    text = _collapse_whitespace(text)
    return text


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_for_nwer(text: str) -> str:
    """
    Normalization for nWER:
    - Apply text_benchmark normalization
    - Lowercase
    - Remove all punctuation
    - Collapse whitespace
    """
    text = normalize_for_text_benchmark(text)
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = _collapse_whitespace(text)
    return text
