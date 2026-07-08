"""
Evaluator for UC04-06: Table documents.
Primary metric: TEDS
Secondary: cell accuracy, structure F1
"""

from __future__ import annotations
from ..metrics.teds import compute_teds


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

        teds_result = compute_teds(gt_html, pred_html, doc_id, page_num, tid)

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
