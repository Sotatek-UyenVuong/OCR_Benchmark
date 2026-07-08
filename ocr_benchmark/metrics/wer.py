"""
WER  — Word Error Rate          (secondary metric, all UC)
nWER — Normalized WER           (secondary metric, all UC)

WER  = (S + D + I) / N   where N = number of words in ground truth
nWER = WER after lowercasing + stripping punctuation
"""

from __future__ import annotations
import re
import jiwer


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize_for_nwer(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

def compute_wer(
    ground_truth: str,
    prediction: str,
    doc_id: str = "",
    page_num: int = 1,
) -> dict:
    """
    Compute Word Error Rate.

    Returns:
        {
            "doc_id": str,
            "page_num": int,
            "wer": float,
            "wer_detail": {
                "substitutions": int,
                "deletions": int,
                "insertions": int,
                "total_words_gt": int,
            },
            "ground_truth": str,
            "prediction": str,
        }
    """
    gt_words = ground_truth.split()
    n = len(gt_words)

    if n == 0:
        return {
            "doc_id": doc_id,
            "page_num": page_num,
            "wer": 0.0,
            "wer_detail": {"substitutions": 0, "deletions": 0, "insertions": 0, "total_words_gt": 0},
            "ground_truth": ground_truth,
            "prediction": prediction,
        }

    measures = jiwer.process_words(ground_truth, prediction)
    wer_value = measures.wer

    return {
        "doc_id": doc_id,
        "page_num": page_num,
        "wer": round(wer_value, 6),
        "wer_detail": {
            "substitutions": measures.substitutions,
            "deletions": measures.deletions,
            "insertions": measures.insertions,
            "total_words_gt": n,
        },
        "ground_truth": ground_truth,
        "prediction": prediction,
    }


# ---------------------------------------------------------------------------
# nWER
# ---------------------------------------------------------------------------

def compute_nwer(
    ground_truth: str,
    prediction: str,
    doc_id: str = "",
    page_num: int = 1,
) -> dict:
    """
    Compute Normalized WER (lowercase + strip punctuation before scoring).

    Returns:
        {
            "doc_id": str,
            "page_num": int,
            "nwer": float,
            "normalized_ground_truth": str,
            "normalized_prediction": str,
        }
    """
    ref = _normalize_for_nwer(ground_truth)
    hyp = _normalize_for_nwer(prediction)

    if not ref.split():
        return {
            "doc_id": doc_id,
            "page_num": page_num,
            "nwer": 0.0,
            "normalized_ground_truth": ref,
            "normalized_prediction": hyp,
        }

    nwer_value = jiwer.wer(ref, hyp)

    return {
        "doc_id": doc_id,
        "page_num": page_num,
        "nwer": round(nwer_value, 6),
        "normalized_ground_truth": ref,
        "normalized_prediction": hyp,
    }
