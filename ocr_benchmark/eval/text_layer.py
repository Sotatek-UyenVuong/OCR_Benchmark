"""
Evaluator for UC07-09: Text Layer PDF.
Primary metrics: CER + Layout IoU
"""

from __future__ import annotations
from ..metrics.cer import compute_cer
from ..metrics.iou import compute_layout_iou
from ..normalize import normalize_for_text_benchmark


def eval_text_layer(
    gt_page: dict,
    pred_blocks: list[dict],
    pred_full_text: str,
    doc_id: str = "",
    iou_threshold: float = 0.5,
    include_alignment: bool = False,
) -> dict:
    """
    Evaluate a single page for text-layer PDF UC (UC07-09).

    Args:
        gt_page:        One page from GT JSON:
                        {"page_num": int, "full_text": str, "blocks": [...]}
        pred_blocks:    List of predicted blocks:
                        [{"block_id": int, "bbox": [x1,y1,x2,y2], "text": str}]
        pred_full_text: Concatenated full text from model (reading order).
        doc_id:         Document identifier.
        iou_threshold:  Threshold to classify a block as "matched".
        include_alignment: Pass through to CER for char-level alignment.

    Returns flat dict with CER + IoU metrics plus debug fields.
    """
    page_num = gt_page.get("page_num", 1)
    gt_text_raw = gt_page.get("full_text", "")
    gt_blocks = gt_page.get("blocks", [])

    gt_text = normalize_for_text_benchmark(gt_text_raw)
    pred_text_norm = normalize_for_text_benchmark(pred_full_text)

    cer_result = compute_cer(gt_text, pred_text_norm, doc_id, page_num, include_alignment)
    iou_result = compute_layout_iou(gt_blocks, pred_blocks, doc_id, page_num, iou_threshold)

    return {
        "doc_id": doc_id,
        "page_num": page_num,
        # Primary
        "cer": cer_result["cer"],
        "cer_detail": cer_result["cer_detail"],
        "mean_iou": iou_result["mean_iou"],
        # Debug
        "ground_truth": gt_text,
        "prediction": pred_text_norm,
        "char_alignment": cer_result.get("char_alignment"),
        "blocks": iou_result["blocks"],
    }
