"""
Evaluator for UC04-06: Table documents.
Primary metric: TEDS
Secondary: cell accuracy, structure F1
"""

from __future__ import annotations
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Local TEDS helpers (formerly in teds.py)
# These preserve the full return schema: teds_detail + cell_diff
# ---------------------------------------------------------------------------

class _TreeNode:
    def __init__(self, tag: str, text: str = "", attrs: dict | None = None):
        self.tag = tag
        self.text = (text or "").strip()
        self.attrs = attrs or {}
        self.children: list["_TreeNode"] = []


def _parse_html_to_tree(html: str) -> _TreeNode | None:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return None
    return _tag_to_node(table)


def _tag_to_node(tag) -> _TreeNode:
    from bs4 import Tag
    attrs = {}
    if tag.get("rowspan") and tag["rowspan"] != "1":
        attrs["rowspan"] = tag["rowspan"]
    if tag.get("colspan") and tag["colspan"] != "1":
        attrs["colspan"] = tag["colspan"]

    direct_text = tag.get_text(separator=" ", strip=True) if tag.name in ("td", "th") else ""
    node = _TreeNode(tag=tag.name, text=direct_text, attrs=attrs)

    for child in tag.children:
        if isinstance(child, Tag):
            node.children.append(_tag_to_node(child))

    return node


def _tree_size(node: _TreeNode) -> int:
    return 1 + sum(_tree_size(c) for c in node.children)


def _ted(n1: _TreeNode | None, n2: _TreeNode | None) -> int:
    if n1 is None and n2 is None:
        return 0
    if n1 is None:
        return 1 + sum(_ted(None, c) for c in n2.children)
    if n2 is None:
        return 1 + sum(_ted(c, None) for c in n1.children)

    relabel = 0 if (n1.tag == n2.tag and n1.text == n2.text and n1.attrs == n2.attrs) else 1

    c1, c2 = n1.children, n2.children
    len1, len2 = len(c1), len(c2)

    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = sum(_tree_size(c1[k]) for k in range(i))
    for j in range(len2 + 1):
        dp[0][j] = sum(_tree_size(c2[k]) for k in range(j))

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            dp[i][j] = min(
                dp[i - 1][j] + _tree_size(c1[i - 1]),
                dp[i][j - 1] + _tree_size(c2[j - 1]),
                dp[i - 1][j - 1] + _ted(c1[i - 1], c2[j - 1]),
            )

    return relabel + dp[len1][len2]


def _extract_cells(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    cells = []
    for r_idx, row in enumerate(soup.find_all("tr")):
        for c_idx, cell in enumerate(row.find_all(["td", "th"])):
            cells.append({
                "row": r_idx,
                "col": c_idx,
                "is_header": cell.name == "th",
                "rowspan": int(cell.get("rowspan", 1)),
                "colspan": int(cell.get("colspan", 1)),
                "text": cell.get_text(separator=" ", strip=True),
            })
    return cells


def _build_cell_diff(gt_cells: list[dict], pred_cells: list[dict]) -> list[dict]:
    gt_map = {(c["row"], c["col"]): c for c in gt_cells}
    pred_map = {(c["row"], c["col"]): c for c in pred_cells}
    all_keys = sorted(set(gt_map) | set(pred_map))

    diff = []
    for key in all_keys:
        gt_c = gt_map.get(key)
        pr_c = pred_map.get(key)

        if gt_c is None:
            diff.append({**pr_c, "gt_text": None, "pred_text": pr_c["text"],
                         "match": False, "issue": "extra_cell"})
            continue
        if pr_c is None:
            diff.append({**gt_c, "gt_text": gt_c["text"], "pred_text": None,
                         "match": False, "issue": "missing_cell"})
            continue

        issues = []
        if gt_c["is_header"] != pr_c["is_header"]:
            issues.append("header_mismatch")
        if gt_c["rowspan"] != pr_c["rowspan"] or gt_c["colspan"] != pr_c["colspan"]:
            issues.append("span_mismatch")
        if gt_c["text"] != pr_c["text"]:
            issues.append("text_mismatch")

        diff.append({
            "row": key[0], "col": key[1],
            "gt_text": gt_c["text"], "pred_text": pr_c["text"],
            "gt_is_header": gt_c["is_header"], "pred_is_header": pr_c["is_header"],
            "gt_rowspan": gt_c["rowspan"], "pred_rowspan": pr_c["rowspan"],
            "gt_colspan": gt_c["colspan"], "pred_colspan": pr_c["colspan"],
            "match": len(issues) == 0,
            "issue": issues[0] if len(issues) == 1 else (", ".join(issues) if issues else None),
        })

    return [d for d in diff if not d.get("match", True)]


def _compute_teds(
    gt_html: str,
    pred_html: str,
    doc_id: str = "",
    page_num: int = 1,
    table_id: int = 1,
) -> dict:
    """
    Compute TEDS between two HTML table strings, with full detail schema.
    Uses UET's grid normalization (teds_similarity_table) which:
    1. Converts HTML → cell grid (normalizing rowspan/colspan)
    2. Computes TEDS on the flattened grid representation
    This avoids edit_distance > tree_size issue with complex span structures.
    """
    # Use UET's teds_similarity_table which normalizes via grid conversion
    try:
        from ocr_benchmark.metrics.uet_metrics import (
            teds_similarity_table, extract_html_tables
        )
        gt_grids  = extract_html_tables(gt_html)
        pred_grids = extract_html_tables(pred_html)
        if gt_grids and pred_grids:
            teds = teds_similarity_table(gt_grids[0], pred_grids[0])
        elif not gt_grids and not pred_grids:
            teds = 1.0
        else:
            teds = 0.0
    except Exception:
        # Fallback to raw tree edit distance
        gt_tree  = _parse_html_to_tree(gt_html)
        pred_tree = _parse_html_to_tree(pred_html)
        if gt_tree is None and pred_tree is None:
            teds = 1.0
        else:
            gt_size   = _tree_size(gt_tree)   if gt_tree   else 0
            pred_size = _tree_size(pred_tree) if pred_tree else 0
            edit_dist = _ted(gt_tree, pred_tree)
            denom = max(gt_size, pred_size)
            teds = max(0.0, min(1.0, 1.0 - edit_dist / denom)) if denom > 0 else 1.0

    # Keep tree sizes for teds_detail (informational)
    gt_tree   = _parse_html_to_tree(gt_html)
    pred_tree = _parse_html_to_tree(pred_html)
    gt_size   = _tree_size(gt_tree)   if gt_tree   else 0
    pred_size = _tree_size(pred_tree) if pred_tree else 0
    edit_dist = _ted(gt_tree, pred_tree) if (gt_tree or pred_tree) else 0

    gt_cells   = _extract_cells(gt_html)
    pred_cells = _extract_cells(pred_html)
    cell_diff  = _build_cell_diff(gt_cells, pred_cells)

    return {
        "doc_id": doc_id,
        "page_num": page_num,
        "table_id": table_id,
        "teds": round(teds, 6),
        "teds_detail": {
            "edit_distance": edit_dist,
            "gt_tree_size": gt_size,
            "pred_tree_size": pred_size,
        },
        "ground_truth_html": gt_html,
        "prediction_html": pred_html,
        "cell_diff": cell_diff,
    }


# ---------------------------------------------------------------------------
# Secondary metrics helpers
# ---------------------------------------------------------------------------

def _cell_accuracy(gt_cells: list[dict], pred_cells: list[dict]) -> float:
    """
    Fraction of (row, col) positions where text matches exactly.
    """
    gt_map = {(c["row"], c["col"]): c["text"] for c in gt_cells}
    pred_map = {(c["row"], c["col"]): c["text"] for c in pred_cells}

    if not gt_map:
        return 1.0

    correct = sum(1 for k, v in gt_map.items() if pred_map.get(k) == v)
    return round(correct / len(gt_map), 6)


def _structure_f1(gt_cells: list[dict], pred_cells: list[dict]) -> dict:
    """
    F1 on (row, col) cell positions, ignoring text content.
    Measures whether the model got the right number of rows/cols.
    """
    gt_positions = {(c["row"], c["col"]) for c in gt_cells}
    pred_positions = {(c["row"], c["col"]) for c in pred_cells}

    tp = len(gt_positions & pred_positions)
    fp = len(pred_positions - gt_positions)
    fn = len(gt_positions - pred_positions)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "structure_precision": round(precision, 6),
        "structure_recall": round(recall, 6),
        "structure_f1": round(f1, 6),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def eval_table(
    gt_page: dict,
    pred_tables: list[dict],
    doc_id: str = "",
) -> dict:
    """
    Evaluate a single page for table UC (UC04-06).

    Args:
        gt_page:     One page from GT JSON:
                     {"page_num": int, "tables": [{"table_id": int, "html": str, "cells": [...]}]}
        pred_tables: List of predicted tables for this page:
                     [{"table_id": int, "html": str, "cells": [...]}]
        doc_id:      Document identifier.

    Returns:
        {
            "doc_id": str,
            "page_num": int,
            "avg_teds": float,          # average across all tables on page
            "tables": [per-table result],
        }
    """
    page_num = gt_page.get("page_num", 1)
    gt_tables = gt_page.get("tables", [])

    pred_map = {t.get("table_id", i): t for i, t in enumerate(pred_tables)}

    table_results = []
    for gt_table in gt_tables:
        tid = gt_table.get("table_id", 1)
        pred_table = pred_map.get(tid, {})

        gt_html = gt_table.get("html", "")
        pred_html = pred_table.get("html", "")

        teds_result = _compute_teds(gt_html, pred_html, doc_id, page_num, tid)

        gt_cells = gt_table.get("cells", [])
        pred_cells = pred_table.get("cells", [])
        cell_acc = _cell_accuracy(gt_cells, pred_cells)
        struct_f1 = _structure_f1(gt_cells, pred_cells)

        table_results.append({
            "table_id": tid,
            "teds": teds_result["teds"],
            "teds_detail": teds_result["teds_detail"],
            "cell_accuracy": cell_acc,
            **struct_f1,
            "ground_truth_html": gt_html,
            "prediction_html": pred_html,
            "cell_diff": teds_result["cell_diff"],
        })

    avg_teds = (
        sum(t["teds"] for t in table_results) / len(table_results)
        if table_results else 0.0
    )

    return {
        "doc_id": doc_id,
        "page_num": page_num,
        "avg_teds": round(avg_teds, 6),
        "tables": table_results,
    }
