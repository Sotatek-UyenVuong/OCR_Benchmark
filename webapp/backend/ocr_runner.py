"""
ocr_runner.py
-------------
API router để trigger OCR và evaluate.

POST /api/ocr/run      — chạy OCR (idempotent: skip nếu prediction đã có)
POST /api/ocr/eval     — evaluate 1 doc với GT đã lưu
GET  /api/ocr/pdfs     — list tất cả PDF trong raw/
GET  /api/ocr/result   — load kết quả eval đã lưu (nếu có)
"""

from __future__ import annotations
import json
import time
import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ocr_benchmark.eval.scan import eval_scan, _compute_wer_detail as _wer_detail_fn, _compute_nwer as _nwer_fn
from ocr_benchmark.eval.table import eval_table
from ocr_benchmark.eval.text_layer import eval_text_layer
from ocr_benchmark.normalize import normalize_for_text_benchmark


def compute_wer(gt_text: str, pred_text: str, doc_id: str = "", page_num: int = 1) -> dict:
    """Compatibility wrapper returning old compute_wer() dict schema."""
    info = _wer_detail_fn(gt_text, pred_text)
    return {
        "doc_id": doc_id,
        "page_num": page_num,
        "wer": info["wer"],
        "wer_detail": info["wer_detail"],
        "ground_truth": gt_text,
        "prediction": pred_text,
    }


def compute_nwer(gt_text: str, pred_text: str, doc_id: str = "", page_num: int = 1) -> dict:
    """Compatibility wrapper returning old compute_nwer() dict schema."""
    return {
        "doc_id": doc_id,
        "page_num": page_num,
        "nwer": _nwer_fn(gt_text, pred_text),
    }
from .config import PROJECT_ROOT, RAW_ROOT, GT_ROOT, PRED_ROOT, RESULT_ROOT, MARKER_SUBDIR

router = APIRouter(prefix="/api/ocr", tags=["OCR"])

# In-memory job status: job_id → { status, message, progress }
_jobs: dict[str, dict] = {}


# ── helpers ──────────────────────────────────────────────────────────────────

def _uc_type_from_path(uc_dir: str) -> str:
    """'scan' | 'table' | 'text_layer' from directory name"""
    return uc_dir  # directory name IS the uc_type

def _lang_label(lang: str) -> str:
    return {"vi": "Tiếng Việt", "en": "English", "ja": "日本語"}.get(lang, lang)

def _pred_path(model: str, uc_type: str, lang: str, doc_id: str) -> Path:
    """predictions/<model>/<uc_type>/<lang>/<doc_id>.json"""
    return PRED_ROOT / model / uc_type / lang / f"{doc_id}.json"

def _gt_path(uc_type: str, lang: str, doc_id: str) -> Path:
    return GT_ROOT / uc_type / lang / f"{doc_id}.json"

def _result_path(model: str, uc_type: str, lang: str, doc_id: str) -> Path:
    return RESULT_ROOT / model / uc_type / lang / f"{doc_id}_eval.json"


def _safe_mean(vals):
    return round(sum(vals) / len(vals), 6) if vals else 0.0


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.get("/pdfs")
def list_pdfs():
    """
    List all PDF files in raw/ with their OCR and GT status per model.
    """
    pdfs = []
    for pdf in sorted(RAW_ROOT.rglob("*.pdf")):
        rel = pdf.relative_to(RAW_ROOT)
        parts = rel.parts
        if len(parts) != 3:   # <uc_type>/<lang>/<doc_id>.pdf
            continue
        uc_type, lang, filename = parts
        doc_id = pdf.stem

        # Check marker prediction exists
        marker_text_pred = RAW_ROOT / uc_type / lang / MARKER_SUBDIR / f"{doc_id}_text_prediction.json"
        marker_table_pred = RAW_ROOT / uc_type / lang / MARKER_SUBDIR / f"{doc_id}_table_prediction.json"
        marker_unified_pred = RAW_ROOT / uc_type / lang / MARKER_SUBDIR / f"{doc_id}_prediction.json"
        marker_done = marker_text_pred.exists() or marker_unified_pred.exists()

        # Check GT saved
        gt_file = _gt_path(uc_type, lang, doc_id)

        # Check eval result
        eval_file = _result_path("marker", uc_type, lang, doc_id)

        pdfs.append({
            "doc_id": doc_id,
            "uc_type": uc_type,
            "lang": lang,
            "lang_label": _lang_label(lang),
            "pdf_url": f"/api/gt/pdf/{uc_type}/{lang}/{filename}",
            "ocr_done": {"marker": marker_done},
            "gt_saved": gt_file.exists(),
            "eval_done": {"marker": eval_file.exists()},
            # split info
            "has_text_pred": marker_text_pred.exists(),
            "has_table_pred": marker_table_pred.exists(),
        })
    return pdfs


class RunOCRRequest(BaseModel):
    doc_id: str
    uc_type: str
    lang: str
    model: str = "marker"
    force: bool = False          # re-run even if already done
    ocr_mode: str = "balanced"   # fast | balanced | accurate


@router.post("/run")
async def run_ocr(req: RunOCRRequest, background_tasks: BackgroundTasks):
    """
    Trigger OCR for a document.
    Idempotent: if prediction already exists, skip (unless force=True).
    Returns job_id to poll status.
    """
    pdf_path = RAW_ROOT / req.uc_type / req.lang / f"{req.doc_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, f"PDF not found: {pdf_path.relative_to(PROJECT_ROOT)}")

    # Check if already done
    text_pred = RAW_ROOT / req.uc_type / req.lang / MARKER_SUBDIR / f"{req.doc_id}_text_prediction.json"
    unified_pred = RAW_ROOT / req.uc_type / req.lang / MARKER_SUBDIR / f"{req.doc_id}_prediction.json"
    already_done = text_pred.exists() or unified_pred.exists()

    if already_done and not req.force:
        return {
            "job_id": None,
            "status": "skipped",
            "message": "OCR already done. Use force=true to re-run.",
            "text_pred_path": str(text_pred.relative_to(PROJECT_ROOT)) if text_pred.exists() else None,
        }

    # Start background job
    job_id = f"{req.doc_id}_{req.model}_{int(time.time())}"
    _jobs[job_id] = {"status": "running", "message": "Starting OCR…", "progress": 0}

    background_tasks.add_task(_run_ocr_job, job_id, req, pdf_path)

    return {"job_id": job_id, "status": "running", "message": "OCR started in background"}


async def _run_ocr_job(job_id: str, req: RunOCRRequest, pdf_path: Path):
    """Background task: run Marker OCR and save prediction files."""
    try:
        _jobs[job_id]["message"] = "Uploading to Marker API…"
        _jobs[job_id]["progress"] = 10

        # Import here to avoid circular at module load
        from ocr_benchmark.ocr_model.marker_convert import convert

        lang_map = {"vi": "vi,en", "en": "en", "ja": "ja,en"}
        langs = lang_map.get(req.lang, req.lang)

        out_dir = RAW_ROOT / req.uc_type / req.lang / MARKER_SUBDIR

        _jobs[job_id]["message"] = "Processing…"
        _jobs[job_id]["progress"] = 30

        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(None, lambda: convert(
            source=str(pdf_path),
            output_dir=out_dir,
            langs=langs,
            mode=req.ocr_mode,
            uc_type="split",
            doc_id=req.doc_id,
            save_full=True,
            save_md=True,
            save_json=True,
            save_html=False,
        ))

        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["message"] = "OCR complete"
        _jobs[job_id]["progress"] = 100
        _jobs[job_id]["outputs"] = {k: str(v.relative_to(PROJECT_ROOT)) for k, v in outputs.items()}

    except Exception as e:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["message"] = str(e)
        _jobs[job_id]["progress"] = 0


@router.get("/job/{job_id}")
def get_job_status(job_id: str):
    """Poll OCR job status."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    return _jobs[job_id]


class EvalRequest(BaseModel):
    doc_id: str
    uc_type: str
    lang: str
    model: str = "marker"
    split: Literal["text", "table", "both"] = "text"


@router.post("/eval")
def run_eval(req: EvalRequest):
    """
    Evaluate a document: compare GT with model prediction.
    Saves result to benchmark_results/ and returns scores.
    """
    gt_file = _gt_path(req.uc_type, req.lang, req.doc_id)
    if not gt_file.exists():
        raise HTTPException(404, "GT not found — save GT first")

    with open(gt_file, encoding="utf-8") as f:
        gt_data = json.load(f)

    results = {}

    # ── text eval ──────────────────────────────────────────────────────────
    if req.split in ("text", "both"):
        text_pred_file = RAW_ROOT / req.uc_type / req.lang / MARKER_SUBDIR / f"{req.doc_id}_text_prediction.json"
        # Fallback to unified
        if not text_pred_file.exists():
            text_pred_file = RAW_ROOT / req.uc_type / req.lang / MARKER_SUBDIR / f"{req.doc_id}_prediction.json"
        if not text_pred_file.exists():
            raise HTTPException(404, "Text prediction not found — run OCR first")

        with open(text_pred_file, encoding="utf-8") as f:
            pred_data = json.load(f)

        pred_pages = {p["page_num"]: p for p in pred_data.get("pages", [])}
        page_results = []

        for gt_page in gt_data.get("pages", []):
            pnum = gt_page.get("page_num", 1)
            pred_page = pred_pages.get(pnum, {})
            pred_text = pred_page.get("full_text", "")

            if req.uc_type == "text_layer":
                r = eval_text_layer(
                    gt_page,
                    pred_blocks=pred_page.get("blocks", []),
                    pred_full_text=pred_text,
                    doc_id=req.doc_id,
                    include_alignment=True,
                )
            else:
                r = eval_scan(gt_page, pred_text=pred_text, doc_id=req.doc_id, include_alignment=True)
                # Add WER/nWER
                gt_norm = normalize_for_text_benchmark(gt_page.get("full_text", ""))
                pred_norm = normalize_for_text_benchmark(pred_text)
                r.update(compute_wer(gt_norm, pred_norm, req.doc_id, pnum))
                r["nwer"] = compute_nwer(gt_norm, pred_norm, req.doc_id, pnum)["nwer"]

            page_results.append(r)

        summary = {
            "avg_cer": _safe_mean([r["cer"] for r in page_results]),
            "avg_wer": _safe_mean([r.get("wer", 0) for r in page_results]),
            "avg_nwer": _safe_mean([r.get("nwer", 0) for r in page_results]),
            "n_pages": len(page_results),
        }
        if req.uc_type == "text_layer":
            summary["avg_mean_iou"] = _safe_mean([r.get("mean_iou", 0) for r in page_results])

        results["text"] = {"summary": summary, "pages": page_results}

    # ── table eval ────────────────────────────────────────────────────────
    if req.split in ("table", "both"):
        table_pred_file = RAW_ROOT / req.uc_type / req.lang / MARKER_SUBDIR / f"{req.doc_id}_table_prediction.json"
        if not table_pred_file.exists():
            if req.split == "table":
                raise HTTPException(404, "Table prediction not found — run OCR first")
        else:
            with open(table_pred_file, encoding="utf-8") as f:
                table_pred_data = json.load(f)

            pred_table_pages = {p["page_num"]: p for p in table_pred_data.get("pages", [])}
            table_results = []

            for gt_page in gt_data.get("pages", []):
                pnum = gt_page.get("page_num", 1)
                if "tables" not in gt_page:
                    continue
                pred_page = pred_table_pages.get(pnum, {})
                r = eval_table(gt_page, pred_tables=pred_page.get("tables", []), doc_id=req.doc_id)
                table_results.append(r)

            if table_results:
                results["table"] = {
                    "summary": {
                        "avg_teds": _safe_mean([r["avg_teds"] for r in table_results]),
                        "n_pages": len(table_results),
                    },
                    "pages": table_results,
                }

    # ── Save result ───────────────────────────────────────────────────────
    out_path = _result_path(req.model, req.uc_type, req.lang, req.doc_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"doc_id": req.doc_id, "model": req.model, **results}, f, ensure_ascii=False, indent=2)

    return {"doc_id": req.doc_id, "model": req.model, **results}


@router.get("/result")
def get_result(
    doc_id: str = Query(...),
    uc_type: str = Query(...),
    lang: str = Query(...),
    model: str = Query(default="marker"),
):
    """Load a previously saved eval result."""
    path = _result_path(model, uc_type, lang, doc_id)
    if not path.exists():
        raise HTTPException(404, "No eval result found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
