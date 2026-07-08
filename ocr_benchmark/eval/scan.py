"""
Evaluator for UC01-03: Scan documents.
Primary metric: CER
Secondary: WER, nWER, Precision/Recall
"""

from __future__ import annotations
from ..metrics.cer import compute_cer
from ..metrics.wer import compute_wer, compute_nwer
from ..normalize import normalize_for_text_benchmark


def _char_precision_recall(gt: str, pred: str) -> dict:
    """Character-level precision and recall."""
    gt_chars = list(gt)
    pred_chars = list(pred)

    # Count matching characters (multiset intersection)
    from collections import Counter
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

    # Normalize both sides
    gt_text = normalize_for_text_benchmark(gt_text_raw)
    pred_text_norm = normalize_for_text_benchmark(pred_text)

    cer_result = compute_cer(gt_text, pred_text_norm, doc_id, page_num, include_alignment)
    wer_result = compute_wer(gt_text, pred_text_norm, doc_id, page_num)
    nwer_result = compute_nwer(gt_text, pred_text_norm, doc_id, page_num)
    pr = _char_precision_recall(gt_text, pred_text_norm)

    return {
        "doc_id": doc_id,
        "page_num": page_num,
        # Primary
        "cer": cer_result["cer"],
        "cer_detail": cer_result["cer_detail"],
        # Secondary
        "wer": wer_result["wer"],
        "wer_detail": wer_result["wer_detail"],
        "nwer": nwer_result["nwer"],
        "char_precision": pr["char_precision"],
        "char_recall": pr["char_recall"],
        # Debug
        "ground_truth": gt_text,
        "prediction": pred_text_norm,
        "char_alignment": cer_result.get("char_alignment"),
    }
