"""
Evaluator for UC01-03: Scan documents.
Primary metric: CER
Secondary: WER, nWER, Precision/Recall
"""

from __future__ import annotations
import re
from ..metrics.uet_metrics import normalize_ocr_text


def _char_alignment(ref: str, hyp: str) -> list[dict]:
    """
    Build a simple character-level alignment using dynamic programming.
    Returns list of {"gt": char, "pred": char, "type": "match|substitution|deletion|insertion"}.
    """
    n, m = len(ref), len(hyp)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            alignment.append({"gt": ref[i - 1], "pred": hyp[j - 1], "type": "match"})
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append({"gt": ref[i - 1], "pred": hyp[j - 1], "type": "substitution"})
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append({"gt": ref[i - 1], "pred": "-", "type": "deletion"})
            i -= 1
        else:
            alignment.append({"gt": "-", "pred": hyp[j - 1], "type": "insertion"})
            j -= 1

    alignment.reverse()
    return alignment


def _compute_cer_detail(ground_truth: str, prediction: str, include_alignment: bool = False) -> dict:
    """
    Compute CER with full detail dict, using the same alignment logic as the old cer.py.
    Returns dict compatible with the old compute_cer() return schema.
    """
    n = len(ground_truth)

    if n == 0:
        return {
            "cer": 0.0,
            "cer_detail": {"substitutions": 0, "deletions": 0, "insertions": 0, "total_chars_gt": 0},
            "char_alignment": [] if include_alignment else None,
        }

    alignment = _char_alignment(ground_truth, prediction)

    substitutions = sum(1 for a in alignment if a["type"] == "substitution")
    deletions = sum(1 for a in alignment if a["type"] == "deletion")
    insertions = sum(1 for a in alignment if a["type"] == "insertion")

    cer = (substitutions + deletions + insertions) / n

    return {
        "cer": round(cer, 6),
        "cer_detail": {
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "total_chars_gt": n,
        },
        "char_alignment": (
            [a for a in alignment if a["type"] != "match"] if include_alignment else None
        ),
    }


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize_for_nwer(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _compute_wer_detail(ground_truth: str, prediction: str) -> dict:
    """
    Compute WER with detail dict.
    Uses jiwer for the WER value, matching old wer.py behavior.
    Returns dict compatible with old compute_wer() return schema.
    """
    import jiwer

    gt_words = ground_truth.split()
    n = len(gt_words)

    if n == 0:
        return {
            "wer": 0.0,
            "wer_detail": {"substitutions": 0, "deletions": 0, "insertions": 0, "total_words_gt": 0},
        }

    measures = jiwer.process_words(ground_truth, prediction)
    wer_value = measures.wer

    return {
        "wer": round(wer_value, 6),
        "wer_detail": {
            "substitutions": measures.substitutions,
            "deletions": measures.deletions,
            "insertions": measures.insertions,
            "total_words_gt": n,
        },
    }


def _compute_nwer(ground_truth: str, prediction: str) -> float:
    """Compute normalized WER (lowercase + strip punctuation)."""
    import jiwer

    ref = _normalize_for_nwer(ground_truth)
    hyp = _normalize_for_nwer(prediction)

    if not ref.split():
        return 0.0

    return round(jiwer.wer(ref, hyp), 6)


def _char_precision_recall(gt: str, pred: str) -> dict:
    """Character-level precision and recall."""
    from collections import Counter

    gt_chars = list(gt)
    pred_chars = list(pred)

    gt_counter = Counter(gt_chars)
    pred_counter = Counter(pred_chars)
    common = sum((gt_counter & pred_counter).values())

    precision = common / len(pred_chars) if pred_chars else 0.0
    recall = common / len(gt_chars) if gt_chars else 0.0

    return {
        "char_precision": round(precision, 6),
        "char_recall": round(recall, 6),
    }


def eval_scan(
    gt_page: dict,
    pred_text: str,
    doc_id: str = "",
    include_alignment: bool = False,
) -> dict:
    """
    Evaluate a single page for scan UC (UC01-03).

    Args:
        gt_page:   One page from the GT JSON:
                   {"page_num": int, "full_text": str, ...}
        pred_text: Raw text output from the OCR model for this page.
        doc_id:    Document identifier.
        include_alignment: Pass through to CER for char-level alignment.

    Returns flat dict with all metrics for this page.
    """
    page_num = gt_page.get("page_num", 1)
    gt_text_raw = gt_page.get("full_text", "")

    # Normalize both sides using uet_metrics normalizer
    gt_text = normalize_ocr_text(gt_text_raw)
    pred_text_norm = normalize_ocr_text(pred_text)

    cer_info = _compute_cer_detail(gt_text, pred_text_norm, include_alignment)
    wer_info = _compute_wer_detail(gt_text, pred_text_norm)
    nwer_val = _compute_nwer(gt_text, pred_text_norm)
    pr = _char_precision_recall(gt_text, pred_text_norm)

    return {
        "doc_id": doc_id,
        "page_num": page_num,
        # Primary
        "cer": cer_info["cer"],
        "cer_detail": cer_info["cer_detail"],
        # Secondary
        "wer": wer_info["wer"],
        "wer_detail": wer_info["wer_detail"],
        "nwer": nwer_val,
        "char_precision": pr["char_precision"],
        "char_recall": pr["char_recall"],
        # Debug
        "ground_truth": gt_text,
        "prediction": pred_text_norm,
        "char_alignment": cer_info.get("char_alignment"),
    }
