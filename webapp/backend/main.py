"""
OCR Benchmark Web API
FastAPI backend: upload PDF + GT JSON, run OCR model, return scores + char-level diff
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path so ocr_benchmark package resolves
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ocr_benchmark.eval.scan import eval_scan
from ocr_benchmark.eval.table import eval_table
from ocr_benchmark.eval.text_layer import eval_text_layer
from ocr_benchmark.metrics.pcs import compute_pcs
from ocr_benchmark.metrics.wer import compute_wer, compute_nwer
from ocr_benchmark.normalize import normalize_for_text_benchmark
from .gt_review import router as gt_router
from .ocr_runner import router as ocr_router

app = FastAPI(title="OCR Benchmark API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(gt_router)
app.include_router(ocr_router)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# ---------------------------------------------------------------------------
# Available mock models (real models plugged in later)
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = [
    {"id": "gpt4o",                  "name": "GPT-4o",                    "tier": "Cloud frontier"},
    {"id": "gemini_25_pro",          "name": "Gemini 2.5 Pro",            "tier": "Cloud frontier"},
    {"id": "azure_doc_intelligence", "name": "Azure Document Intelligence","tier": "Cloud OCR"},
    {"id": "google_doc_ai",          "name": "Google Document AI",        "tier": "Cloud OCR"},
    {"id": "aws_textract",           "name": "AWS Textract",              "tier": "Cloud OCR"},
    {"id": "mistral_ocr",            "name": "Mistral OCR",               "tier": "Cloud emerging"},
    {"id": "paddleocr",              "name": "PaddleOCR v4",              "tier": "Open source"},
    {"id": "surya",                  "name": "Surya",                     "tier": "Open source"},
    {"id": "tesseract",              "name": "Tesseract 5",               "tier": "Open source"},
    {"id": "marker",                 "name": "Marker OCR",                "tier": "Open source"},
]

UC_OPTIONS = [
    {"id": "scan_vi",       "label": "Scan — Tiếng Việt",    "type": "scan"},
    {"id": "scan_en",       "label": "Scan — English",       "type": "scan"},
    {"id": "scan_ja",       "label": "Scan — 日本語",         "type": "scan"},
    {"id": "table_vi",      "label": "Table — Tiếng Việt",   "type": "table"},
    {"id": "table_en",      "label": "Table — English",      "type": "table"},
    {"id": "table_ja",      "label": "Table — 日本語",        "type": "table"},
    {"id": "text_layer_vi", "label": "Text Layer — Tiếng Việt","type": "text_layer"},
    {"id": "text_layer_en", "label": "Text Layer — English",  "type": "text_layer"},
    {"id": "text_layer_ja", "label": "Text Layer — 日本語",   "type": "text_layer"},
]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/models")
def list_models():
    return AVAILABLE_MODELS


@app.get("/api/uc-options")
def list_uc_options():
    return UC_OPTIONS


@app.post("/api/evaluate")
async def evaluate(
    gt_json: UploadFile = File(..., description="Ground truth JSON file"),
    prediction_text: UploadFile = File(..., description="OCR model output text file (.txt)"),
    model_id: str = Form(...),
    uc_type: Literal["scan", "table", "text_layer"] = Form(...),
    doc_id: str = Form(default="doc_001"),
    include_alignment: bool = Form(default=True),
):
    """
    Run evaluation for a single document.
    Returns scores + per-character diff for PDF annotation.
    """
    # Read uploads
    gt_bytes = await gt_json.read()
    pred_bytes = await prediction_text.read()

    try:
        gt_data = json.loads(gt_bytes.decode("utf-8"))
    except Exception:
        raise HTTPException(400, "GT JSON is invalid")

    pred_text_raw = pred_bytes.decode("utf-8")

    # Time the evaluation
    t_start = time.perf_counter()

    results_per_page = []

    for gt_page in gt_data.get("pages", []):
        page_num = gt_page.get("page_num", 1)

        if uc_type == "scan":
            page_result = eval_scan(
                gt_page,
                pred_text_raw,
                doc_id=doc_id,
                include_alignment=include_alignment,
            )
        elif uc_type == "table":
            # For table UC, pred must also be JSON with "tables" key
            try:
                pred_data = json.loads(pred_text_raw)
                pred_tables = pred_data.get("pages", [{}])[page_num - 1].get("tables", [])
            except Exception:
                pred_tables = []
            page_result = eval_table(gt_page, pred_tables, doc_id=doc_id)
        else:  # text_layer
            page_result = eval_text_layer(
                gt_page,
                pred_page_blocks=[],
                pred_full_text=pred_text_raw,
                doc_id=doc_id,
                include_alignment=include_alignment,
            )

        # Augment with WER / PCS for scan & text_layer
        if uc_type in ("scan", "text_layer"):
            gt_text = normalize_for_text_benchmark(gt_page.get("full_text", ""))
            pred_text_norm = normalize_for_text_benchmark(pred_text_raw)
            wer_r = compute_wer(gt_text, pred_text_norm, doc_id, page_num)
            nwer_r = compute_nwer(gt_text, pred_text_norm, doc_id, page_num)
            pcs_r = compute_pcs(gt_text, pred_text_norm, doc_id, page_num)

            page_result["wer"] = wer_r["wer"]
            page_result["wer_detail"] = wer_r["wer_detail"]
            page_result["nwer"] = nwer_r["nwer"]
            page_result["pcs"] = pcs_r["pcs"]
            page_result["punct_miou"] = pcs_r["punct_miou"]
            page_result["cap_miou"] = pcs_r["cap_miou"]

        results_per_page.append(page_result)

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    # Aggregate summary
    def _mean(vals):
        return round(sum(vals) / len(vals), 6) if vals else 0.0

    summary = {
        "model_id": model_id,
        "doc_id": doc_id,
        "uc_type": uc_type,
        "n_pages": len(results_per_page),
        "processing_time_ms": round(elapsed_ms, 2),
    }

    if uc_type in ("scan", "text_layer"):
        summary["avg_cer"] = _mean([r["cer"] for r in results_per_page])
        summary["avg_wer"] = _mean([r.get("wer", 0) for r in results_per_page])
        summary["avg_nwer"] = _mean([r.get("nwer", 0) for r in results_per_page])
        summary["avg_pcs"] = _mean([r.get("pcs", 0) for r in results_per_page])
        summary["avg_punct_miou"] = _mean([r.get("punct_miou", 0) for r in results_per_page])
        summary["avg_cap_miou"] = _mean([r.get("cap_miou", 0) for r in results_per_page])
    elif uc_type == "table":
        summary["avg_teds"] = _mean([r.get("avg_teds", 0) for r in results_per_page])

    return JSONResponse({
        "summary": summary,
        "pages": results_per_page,
    })


@app.get("/api/health")
def health():
    return {"status": "ok"}
