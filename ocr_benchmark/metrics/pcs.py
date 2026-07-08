"""
punct_mIoU — Punctuation mean IoU
cap_mIoU   — Capitalization mean IoU
PCS        — Punctuation & Capitalization Score = (punct_mIoU + cap_mIoU) / 2
"""

from __future__ import annotations
import re
import jiwer

# Punctuation set to track
PUNCT_SET = {",", ".", "?", "!", ";", ":"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word_list(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _iou(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Punct mIoU
# ---------------------------------------------------------------------------

def compute_punct_miou(
    ground_truth: str,
    prediction: str,
) -> dict:
    """
    Compute punctuation mean IoU across all punctuation types that appear
    in either ref or hyp.

    Steps:
    1. Split into word lists.
    2. Align with jiwer.process_words().
    3. For each punct type, build binary vectors over aligned positions.
    4. Compute IoU per punct type → average.

    Returns:
        {
            "punct_miou": float,
            "per_punct": { ",": float, ".": float, ... },
        }
    """
    ref_words = _word_list(ground_truth)
    hyp_words = _word_list(prediction)

    if not ref_words and not hyp_words:
        return {"punct_miou": 1.0, "per_punct": {}}

    # Use jiwer word alignment
    output = jiwer.process_words(ground_truth, prediction)
    # alignment_chunk: list of AlignmentChunk with type in {"equal","substitute","delete","insert"}
    chunks = output.alignments[0]

    # Build aligned pairs: (ref_word_or_None, hyp_word_or_None)
    pairs: list[tuple[str | None, str | None]] = []
    for chunk in chunks:
        if chunk.type == "equal" or chunk.type == "substitute":
            ref_slice = ref_words[chunk.ref_start_idx : chunk.ref_end_idx]
            hyp_slice = hyp_words[chunk.hyp_start_idx : chunk.hyp_end_idx]
            for r, h in zip(ref_slice, hyp_slice):
                pairs.append((r, h))
        elif chunk.type == "delete":
            for r in ref_words[chunk.ref_start_idx : chunk.ref_end_idx]:
                pairs.append((r, None))
        elif chunk.type == "insert":
            for h in hyp_words[chunk.hyp_start_idx : chunk.hyp_end_idx]:
                pairs.append((None, h))

    # Collect per-punct TP/FP/FN
    punct_stats: dict[str, dict[str, int]] = {p: {"tp": 0, "fp": 0, "fn": 0} for p in PUNCT_SET}

    for ref_w, hyp_w in pairs:
        for p in PUNCT_SET:
            ref_has = p in (ref_w or "")
            hyp_has = p in (hyp_w or "")
            if ref_has and hyp_has:
                punct_stats[p]["tp"] += 1
            elif not ref_has and hyp_has:
                punct_stats[p]["fp"] += 1
            elif ref_has and not hyp_has:
                punct_stats[p]["fn"] += 1

    # Only average over punct types that appear in ref or hyp
    active = [p for p in PUNCT_SET if punct_stats[p]["tp"] + punct_stats[p]["fp"] + punct_stats[p]["fn"] > 0]

    per_punct = {
        p: round(_iou(punct_stats[p]["tp"], punct_stats[p]["fp"], punct_stats[p]["fn"]), 6)
        for p in active
    }

    miou = sum(per_punct.values()) / len(per_punct) if per_punct else 1.0

    return {
        "punct_miou": round(miou, 6),
        "per_punct": per_punct,
    }


# ---------------------------------------------------------------------------
# Cap mIoU
# ---------------------------------------------------------------------------

def compute_cap_miou(
    ground_truth: str,
    prediction: str,
) -> dict:
    """
    Compute capitalization mean IoU:
    flag = 1 if first char of word is uppercase.
    Only counts aligned pairs where word_core is non-empty.

    Returns:
        {
            "cap_miou": float,
            "tp": int, "fp": int, "fn": int,
        }
    """
    ref_words = _word_list(ground_truth)
    hyp_words = _word_list(prediction)

    if not ref_words and not hyp_words:
        return {"cap_miou": 1.0, "tp": 0, "fp": 0, "fn": 0}

    output = jiwer.process_words(ground_truth, prediction)
    chunks = output.alignments[0]

    pairs: list[tuple[str | None, str | None]] = []
    for chunk in chunks:
        if chunk.type in ("equal", "substitute"):
            ref_slice = ref_words[chunk.ref_start_idx : chunk.ref_end_idx]
            hyp_slice = hyp_words[chunk.hyp_start_idx : chunk.hyp_end_idx]
            for r, h in zip(ref_slice, hyp_slice):
                pairs.append((r, h))
        elif chunk.type == "delete":
            for r in ref_words[chunk.ref_start_idx : chunk.ref_end_idx]:
                pairs.append((r, None))
        elif chunk.type == "insert":
            for h in hyp_words[chunk.hyp_start_idx : chunk.hyp_end_idx]:
                pairs.append((None, h))

    tp = fp = fn = 0
    for ref_w, hyp_w in pairs:
        # skip empty cores
        ref_core = re.sub(r"[^\w]", "", ref_w or "")
        hyp_core = re.sub(r"[^\w]", "", hyp_w or "")
        if not ref_core and not hyp_core:
            continue

        ref_cap = bool(ref_w and ref_w[0].isupper())
        hyp_cap = bool(hyp_w and hyp_w[0].isupper())

        if ref_cap and hyp_cap:
            tp += 1
        elif not ref_cap and hyp_cap:
            fp += 1
        elif ref_cap and not hyp_cap:
            fn += 1

    cap_miou = _iou(tp, fp, fn)

    return {
        "cap_miou": round(cap_miou, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ---------------------------------------------------------------------------
# PCS
# ---------------------------------------------------------------------------

def compute_pcs(
    ground_truth: str,
    prediction: str,
    doc_id: str = "",
    page_num: int = 1,
) -> dict:
    """
    Compute PCS = (punct_mIoU + cap_mIoU) / 2, plus breakdown.

    Returns:
        {
            "doc_id": str,
            "page_num": int,
            "pcs": float,
            "punct_miou": float,
            "cap_miou": float,
            "per_punct": dict,
            "cap_detail": { "tp": int, "fp": int, "fn": int },
        }
    """
    punct_result = compute_punct_miou(ground_truth, prediction)
    cap_result = compute_cap_miou(ground_truth, prediction)

    pcs = (punct_result["punct_miou"] + cap_result["cap_miou"]) / 2

    return {
        "doc_id": doc_id,
        "page_num": page_num,
        "pcs": round(pcs, 6),
        "punct_miou": punct_result["punct_miou"],
        "cap_miou": cap_result["cap_miou"],
        "per_punct": punct_result["per_punct"],
        "cap_detail": {
            "tp": cap_result["tp"],
            "fp": cap_result["fp"],
            "fn": cap_result["fn"],
        },
    }
