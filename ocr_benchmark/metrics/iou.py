"""
Layout IoU — Intersection over Union for text block bounding boxes.
Primary metric for UC07-09 (Text Layer PDF).

bbox format: [x1, y1, x2, y2] normalized 0-1 relative to page size.
"""

from __future__ import annotations


IOU_THRESHOLD = 0.5  # default match threshold


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _bbox_area(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h


def _bbox_iou(a: list[float], b: list[float]) -> float:
    """Compute IoU between two bboxes [x1, y1, x2, y2]."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h

    union = _bbox_area(a) + _bbox_area(b) - intersection
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Match status classification
# ---------------------------------------------------------------------------

def _classify_match(
    gt_block: dict,
    pred_block: dict | None,
    iou: float,
    threshold: float,
) -> str:
    """
    Classify the relationship between a GT block and its best matching pred block.
    """
    if pred_block is None:
        return "missed"
    if iou >= threshold:
        return "matched"

    gt_area = _bbox_area(gt_block["bbox"])
    pred_area = _bbox_area(pred_block["bbox"]) if pred_block else 0

    # Heuristic: if pred covers much more area → over_merged
    if gt_area > 0 and pred_area / gt_area > 1.8:
        return "over_merged"

    # If pred is much smaller → split (likely one of many pred blocks for this GT)
    if pred_area > 0 and gt_area / pred_area > 1.8:
        return "split"

    return "low_iou"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_layout_iou(
    gt_blocks: list[dict],
    pred_blocks: list[dict],
    doc_id: str = "",
    page_num: int = 1,
    threshold: float = IOU_THRESHOLD,
) -> dict:
    """
    Compute layout IoU between GT and predicted text blocks.

    Args:
        gt_blocks:   List of GT blocks, each: {"block_id": int, "bbox": [x1,y1,x2,y2], "text": str}
        pred_blocks: List of pred blocks, same schema.
        doc_id:      Document identifier.
        page_num:    Page number.
        threshold:   IoU threshold to consider a block "matched".

    Returns:
        {
            "doc_id": str,
            "page_num": int,
            "mean_iou": float,
            "blocks": [
                {
                    "block_id": int,
                    "iou": float,
                    "gt_bbox": list,
                    "pred_bbox": list | None,
                    "gt_text": str,
                    "pred_text": str | None,
                    "match_status": str,   # matched | over_merged | split | missed | extra
                    "issue": str | None,
                }
            ],
        }
    """
    results = []
    matched_pred_ids: set[int] = set()

    # For each GT block, find best matching pred block
    for gt in gt_blocks:
        best_iou = 0.0
        best_pred = None

        for pred in pred_blocks:
            iou = _bbox_iou(gt["bbox"], pred["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_pred = pred

        status = _classify_match(gt, best_pred, best_iou, threshold)

        entry = {
            "block_id": gt.get("block_id"),
            "iou": round(best_iou, 6),
            "gt_bbox": gt["bbox"],
            "pred_bbox": best_pred["bbox"] if best_pred else None,
            "gt_text": gt.get("text", ""),
            "pred_text": best_pred.get("text", "") if best_pred else None,
            "match_status": status,
            "issue": None,
        }

        # Add issue detail for debugging
        if status == "over_merged":
            entry["issue"] = "pred_bbox_covers_multiple_gt_blocks"
        elif status == "split":
            entry["issue"] = "gt_block_split_into_multiple_pred_blocks"
        elif status == "missed":
            entry["issue"] = "no_pred_block_found_for_gt"

        if best_pred:
            matched_pred_ids.add(id(best_pred))

        results.append(entry)

    # Extra pred blocks (no GT matched them)
    for pred in pred_blocks:
        if id(pred) not in matched_pred_ids:
            results.append({
                "block_id": pred.get("block_id"),
                "iou": 0.0,
                "gt_bbox": None,
                "pred_bbox": pred["bbox"],
                "gt_text": None,
                "pred_text": pred.get("text", ""),
                "match_status": "extra",
                "issue": "pred_block_has_no_matching_gt",
            })

    # Mean IoU over GT blocks only (extras don't count toward mean)
    gt_ious = [r["iou"] for r in results if r["match_status"] != "extra"]
    mean_iou = sum(gt_ious) / len(gt_ious) if gt_ious else 0.0

    return {
        "doc_id": doc_id,
        "page_num": page_num,
        "mean_iou": round(mean_iou, 6),
        "blocks": results,
    }
