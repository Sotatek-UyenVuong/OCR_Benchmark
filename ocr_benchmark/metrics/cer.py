"""
CER — Character Error Rate
CER = (S + D + I) / N
Primary metric for UC01-03 (Scan) and UC07-09 (Text Layer).
"""

from __future__ import annotations
import editdistance


def _char_alignment(ref: str, hyp: str) -> list[dict]:
    """
    Build a simple character-level alignment using dynamic programming.
    Returns list of {"gt": char, "pred": char, "type": "match|substitution|deletion|insertion"}.
    """
    n, m = len(ref), len(hyp)

    # dp[i][j] = edit distance between ref[:i] and hyp[:j]
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

    # Traceback
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


def compute_cer(
    ground_truth: str,
    prediction: str,
    doc_id: str = "",
    page_num: int = 1,
    include_alignment: bool = False,
) -> dict:
    """
    Compute Character Error Rate between ground_truth and prediction.

    Args:
        ground_truth: Reference text (GT).
        prediction:   Model output text.
        doc_id:       Document identifier for traceability.
        page_num:     Page number.
        include_alignment: Whether to include per-character alignment (slow for long text).

    Returns:
        {
            "doc_id": str,
            "page_num": int,
            "cer": float,           # 0 = perfect, can be > 1
            "cer_detail": {
                "substitutions": int,
                "deletions": int,
                "insertions": int,
                "total_chars_gt": int,
            },
            "ground_truth": str,
            "prediction": str,
            "char_alignment": list | None,   # only if include_alignment=True
        }
    """
    n = len(ground_truth)

    if n == 0:
        return {
            "doc_id": doc_id,
            "page_num": page_num,
            "cer": 0.0,
            "cer_detail": {"substitutions": 0, "deletions": 0, "insertions": 0, "total_chars_gt": 0},
            "ground_truth": ground_truth,
            "prediction": prediction,
            "char_alignment": [] if include_alignment else None,
        }

    alignment = _char_alignment(ground_truth, prediction)

    substitutions = sum(1 for a in alignment if a["type"] == "substitution")
    deletions = sum(1 for a in alignment if a["type"] == "deletion")
    insertions = sum(1 for a in alignment if a["type"] == "insertion")

    cer = (substitutions + deletions + insertions) / n

    result = {
        "doc_id": doc_id,
        "page_num": page_num,
        "cer": round(cer, 6),
        "cer_detail": {
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "total_chars_gt": n,
        },
        "ground_truth": ground_truth,
        "prediction": prediction,
        "char_alignment": (
            [a for a in alignment if a["type"] != "match"] if include_alignment else None
        ),
    }
    return result
