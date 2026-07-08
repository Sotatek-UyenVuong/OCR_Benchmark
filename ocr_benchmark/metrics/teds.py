"""
TEDS — Tree Edit Distance Similarity
TEDS = 1 - (edit_distance(pred_tree, gt_tree) / max(|pred_tree|, |gt_tree|))

Primary metric for UC04-06 (Table).
"""

from __future__ import annotations
from collections import deque
import re
from bs4 import BeautifulSoup, Tag


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class TreeNode:
    def __init__(self, tag: str, text: str = "", attrs: dict | None = None):
        self.tag = tag
        self.text = (text or "").strip()
        self.attrs = attrs or {}
        self.children: list["TreeNode"] = []

    def __repr__(self):
        return f"TreeNode({self.tag!r}, text={self.text!r})"


def _parse_html_to_tree(html: str) -> TreeNode | None:
    """Parse an HTML table string into a TreeNode tree."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return None
    return _tag_to_node(table)


def _tag_to_node(tag: Tag) -> TreeNode:
    attrs = {}
    if tag.get("rowspan") and tag["rowspan"] != "1":
        attrs["rowspan"] = tag["rowspan"]
    if tag.get("colspan") and tag["colspan"] != "1":
        attrs["colspan"] = tag["colspan"]

    direct_text = tag.get_text(separator=" ", strip=True) if tag.name in ("td", "th") else ""
    node = TreeNode(tag=tag.name, text=direct_text, attrs=attrs)

    for child in tag.children:
        if isinstance(child, Tag):
            node.children.append(_tag_to_node(child))

    return node


def _tree_size(node: TreeNode) -> int:
    """Count total nodes in tree."""
    return 1 + sum(_tree_size(c) for c in node.children)


# ---------------------------------------------------------------------------
# Tree edit distance (Zhang-Shasha simplified via DP on ordered trees)
# We use a simple recursive approach suitable for HTML table trees.
# ---------------------------------------------------------------------------

def _ted(n1: TreeNode | None, n2: TreeNode | None) -> int:
    """
    Compute tree edit distance between n1 and n2.
    Cost: insert=1, delete=1, replace=1 (if tag or text differs).
    """
    if n1 is None and n2 is None:
        return 0
    if n1 is None:
        return 1 + sum(_ted(None, c) for c in n2.children)
    if n2 is None:
        return 1 + sum(_ted(c, None) for c in n1.children)

    # Relabel cost
    relabel = 0 if (n1.tag == n2.tag and n1.text == n2.text and n1.attrs == n2.attrs) else 1

    # Align children greedily (ordered)
    c1, c2 = n1.children, n2.children
    len1, len2 = len(c1), len(c2)

    # DP over children lists
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = sum(_tree_size(c1[k]) for k in range(i))
    for j in range(len2 + 1):
        dp[0][j] = sum(_tree_size(c2[k]) for k in range(j))

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            dp[i][j] = min(
                dp[i - 1][j] + _tree_size(c1[i - 1]),      # delete subtree c1[i-1]
                dp[i][j - 1] + _tree_size(c2[j - 1]),      # insert subtree c2[j-1]
                dp[i - 1][j - 1] + _ted(c1[i - 1], c2[j - 1]),  # match subtrees
            )

    return relabel + dp[len1][len2]


# ---------------------------------------------------------------------------
# Cell diff helper
# ---------------------------------------------------------------------------

def _extract_cells(html: str) -> list[dict]:
    """Extract cell-level data from an HTML table."""
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
    """Compare GT and pred cells by (row, col) position."""
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

    return [d for d in diff if not d.get("match", True)]  # only mismatches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_teds(
    gt_html: str,
    pred_html: str,
    doc_id: str = "",
    page_num: int = 1,
    table_id: int = 1,
) -> dict:
    """
    Compute TEDS between two HTML table strings.

    Args:
        gt_html:   Ground truth HTML table.
        pred_html: Model prediction HTML table.
        doc_id:    Document identifier.
        page_num:  Page number.
        table_id:  Table index on the page.

    Returns:
        {
            "doc_id": str,
            "page_num": int,
            "table_id": int,
            "teds": float,
            "teds_detail": {
                "edit_distance": int,
                "gt_tree_size": int,
                "pred_tree_size": int,
            },
            "ground_truth_html": str,
            "prediction_html": str,
            "cell_diff": list[dict],
        }
    """
    gt_tree = _parse_html_to_tree(gt_html)
    pred_tree = _parse_html_to_tree(pred_html)

    if gt_tree is None and pred_tree is None:
        return {
            "doc_id": doc_id, "page_num": page_num, "table_id": table_id,
            "teds": 1.0,
            "teds_detail": {"edit_distance": 0, "gt_tree_size": 0, "pred_tree_size": 0},
            "ground_truth_html": gt_html,
            "prediction_html": pred_html,
            "cell_diff": [],
        }

    gt_size = _tree_size(gt_tree) if gt_tree else 0
    pred_size = _tree_size(pred_tree) if pred_tree else 0
    edit_dist = _ted(gt_tree, pred_tree)

    denom = max(gt_size, pred_size)
    teds = 1.0 - (edit_dist / denom) if denom > 0 else 1.0
    teds = max(0.0, min(1.0, teds))

    gt_cells = _extract_cells(gt_html)
    pred_cells = _extract_cells(pred_html)
    cell_diff = _build_cell_diff(gt_cells, pred_cells)

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
