"""
Quick smoke tests for all metrics.
Run: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ocr_benchmark.metrics.cer import compute_cer
from ocr_benchmark.metrics.wer import compute_wer, compute_nwer
from ocr_benchmark.metrics.pcs import compute_pcs, compute_punct_miou, compute_cap_miou
from ocr_benchmark.metrics.teds import compute_teds
from ocr_benchmark.metrics.iou import compute_layout_iou
from ocr_benchmark.normalize import normalize_for_text_benchmark, normalize_for_nwer


# ---------------------------------------------------------------------------
# CER
# ---------------------------------------------------------------------------

def test_cer_perfect():
    r = compute_cer("hello", "hello")
    assert r["cer"] == 0.0
    assert r["cer_detail"]["substitutions"] == 0

def test_cer_example_from_docs():
    # "Hợp đồng lao động" vs "Hợp đông lao đông" → 2 substitutions / 18 chars
    gt = "Hợp đồng lao động"
    pred = "Hợp đông lao đông"
    r = compute_cer(gt, pred)
    assert r["cer_detail"]["substitutions"] == 2
    assert r["cer_detail"]["total_chars_gt"] == len(gt)
    assert abs(r["cer"] - 2 / len(gt)) < 1e-4

def test_cer_empty_gt():
    r = compute_cer("", "abc")
    assert r["cer"] == 0.0

def test_cer_with_alignment():
    r = compute_cer("abc", "axc", include_alignment=True)
    assert r["char_alignment"] is not None
    assert any(a["type"] == "substitution" for a in r["char_alignment"])


# ---------------------------------------------------------------------------
# WER / nWER
# ---------------------------------------------------------------------------

def test_wer_perfect():
    r = compute_wer("hello world", "hello world")
    assert r["wer"] == 0.0

def test_wer_one_wrong():
    r = compute_wer("Hợp đồng lao động số 001", "Hợp đồng lao đông số 001")
    # "Hợp đồng lao động số 001" → 6 tokens when split by whitespace
    assert r["wer_detail"]["total_words_gt"] == 6
    assert r["wer"] > 0

def test_nwer_ignores_case_and_punct():
    r = compute_nwer("Hello, World!", "hello world")
    assert r["nwer"] == 0.0


# ---------------------------------------------------------------------------
# PCS
# ---------------------------------------------------------------------------

def test_punct_miou_perfect():
    text = "Hello, world. How are you?"
    r = compute_punct_miou(text, text)
    assert r["punct_miou"] == 1.0

def test_cap_miou_perfect():
    text = "Hello World"
    r = compute_cap_miou(text, text)
    assert r["cap_miou"] == 1.0

def test_pcs_returns_all_fields():
    r = compute_pcs("Hello, World.", "hello world")
    assert "pcs" in r
    assert "punct_miou" in r
    assert "cap_miou" in r
    assert "per_punct" in r
    assert "cap_detail" in r
    assert 0.0 <= r["pcs"] <= 1.0


# ---------------------------------------------------------------------------
# TEDS
# ---------------------------------------------------------------------------

GT_HTML = "<table><tr><th>Tên</th><th>Tuổi</th></tr><tr><td>Nguyễn A</td><td>30</td></tr></table>"
PRED_HTML_PERFECT = GT_HTML
PRED_HTML_NO_HEADER = "<table><tr><td>Tên</td><td>Tuổi</td></tr><tr><td>Nguyễn A</td><td>30</td></tr></table>"

def test_teds_perfect():
    r = compute_teds(GT_HTML, PRED_HTML_PERFECT)
    assert r["teds"] == 1.0
    assert r["cell_diff"] == []

def test_teds_header_mismatch():
    r = compute_teds(GT_HTML, PRED_HTML_NO_HEADER)
    assert r["teds"] < 1.0
    assert any(d["issue"] == "header_mismatch" for d in r["cell_diff"])

def test_teds_returns_html():
    r = compute_teds(GT_HTML, PRED_HTML_NO_HEADER)
    assert "<table>" in r["ground_truth_html"]
    assert "<table>" in r["prediction_html"]


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

GT_BLOCKS = [
    {"block_id": 1, "bbox": [0.1, 0.1, 0.9, 0.3], "text": "Block A"},
    {"block_id": 2, "bbox": [0.1, 0.4, 0.9, 0.6], "text": "Block B"},
]
PRED_BLOCKS_PERFECT = [
    {"block_id": 1, "bbox": [0.1, 0.1, 0.9, 0.3], "text": "Block A"},
    {"block_id": 2, "bbox": [0.1, 0.4, 0.9, 0.6], "text": "Block B"},
]
PRED_BLOCKS_MERGED = [
    {"block_id": 1, "bbox": [0.1, 0.1, 0.9, 0.6], "text": "Block A Block B"},
]

def test_iou_perfect():
    r = compute_layout_iou(GT_BLOCKS, PRED_BLOCKS_PERFECT)
    assert r["mean_iou"] == 1.0
    assert all(b["match_status"] == "matched" for b in r["blocks"] if b["gt_bbox"] is not None)

def test_iou_missed():
    r = compute_layout_iou(GT_BLOCKS, [])
    assert r["mean_iou"] == 0.0
    assert all(b["match_status"] == "missed" for b in r["blocks"])

def test_iou_extra_block():
    extra = [{"block_id": 99, "bbox": [0.0, 0.0, 0.1, 0.1], "text": "Extra"}]
    r = compute_layout_iou([], extra)
    assert any(b["match_status"] == "extra" for b in r["blocks"])


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

def test_normalize_collapses_whitespace():
    assert normalize_for_text_benchmark("hello  \n  world") == "hello world"

def test_normalize_nwer_strips_punct_and_case():
    assert normalize_for_nwer("Hello, World!") == "hello world"
