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
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ocr_benchmark.eval.scan import eval_scan, _compute_wer_detail as _wer_detail_fn, _compute_nwer as _nwer_fn
from ocr_benchmark.eval.table import eval_table
from ocr_benchmark.eval.text_layer import eval_text_layer
from ocr_benchmark.metrics.uet_metrics import compute_all_metrics, compute_table_metrics_from_html
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

# Model → output subfolder mapping
MODEL_SUBDIR = {
    "marker":  "marker_output",
    "mistral": "mistral_output",
}

def _model_subdir(model: str) -> str:
    return MODEL_SUBDIR.get(model, "marker_output")

router = APIRouter(prefix="/api/ocr", tags=["OCR"])

# In-memory job status: job_id → { status, message, progress }
_jobs: dict[str, dict] = {}


# ── helpers ──────────────────────────────────────────────────────────────────

def _uc_type_from_path(uc_dir: str) -> str:
    """'scan' | 'table' | 'text_layer' from directory name"""
    return uc_dir  # directory name IS the uc_type

def _lang_label(lang: str) -> str:
    return {"vi": "Tiếng Việt", "en": "English", "ja": "日本語",
            "ko": "한국어", "zh": "中文"}.get(lang, lang)

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
    List all source files in raw/ (PDF or images) with their OCR and GT status per model.
    """
    _SOURCE_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
    pdfs = []
    seen: set[str] = set()

    for src in sorted(RAW_ROOT.rglob("*")):
        if src.suffix.lower() not in _SOURCE_EXTS:
            continue
        rel = src.relative_to(RAW_ROOT)
        parts = rel.parts
        if len(parts) != 3:   # <uc_type>/<lang>/<doc_id>.<ext>
            continue
        uc_type, lang, filename = parts
        doc_id = src.stem
        key = f"{uc_type}/{lang}/{doc_id}"
        if key in seen:
            continue
        seen.add(key)

        # Check per-model prediction status
        ocr_done = {}
        has_text_pred_by_model = {}
        has_table_pred_by_model = {}
        for m, subdir in MODEL_SUBDIR.items():
            text_pred  = RAW_ROOT / uc_type / lang / subdir / f"{doc_id}_text_prediction.json"
            table_pred = RAW_ROOT / uc_type / lang / subdir / f"{doc_id}_table_prediction.json"
            unified    = RAW_ROOT / uc_type / lang / subdir / f"{doc_id}_prediction.json"
            done = text_pred.exists() or unified.exists()
            ocr_done[m] = done
            has_text_pred_by_model[m]  = text_pred.exists() or unified.exists()
            has_table_pred_by_model[m] = table_pred.exists()

        # Check GT saved
        gt_file = _gt_path(uc_type, lang, doc_id)

        # Check eval result per model
        eval_done = {m: _result_path(m, uc_type, lang, doc_id).exists() for m in MODEL_SUBDIR}

        # Build URL — PDF gets served via gt/pdf endpoint, images via ocr/image endpoint
        is_pdf = src.suffix.lower() == ".pdf"
        pdf_url   = f"/api/gt/pdf/{uc_type}/{lang}/{filename}" if is_pdf else None
        image_url = f"/api/ocr/image/{uc_type}/{lang}/{filename}" if not is_pdf else None

        pdfs.append({
            "doc_id": doc_id,
            "uc_type": uc_type,
            "lang": lang,
            "lang_label": _lang_label(lang),
            "pdf_url": pdf_url,
            "image_url": image_url,
            "source_ext": src.suffix.lower(),
            "ocr_done": ocr_done,
            "gt_saved": gt_file.exists(),
            "eval_done": eval_done,
            "has_text_pred":  has_text_pred_by_model.get("marker", False),
            "has_table_pred": has_table_pred_by_model.get("marker", False),
            "has_text_pred_by_model":  has_text_pred_by_model,
            "has_table_pred_by_model": has_table_pred_by_model,
        })
    return pdfs


@router.get("/image/{uc_type}/{lang}/{filename}")
def serve_image(uc_type: str, lang: str, filename: str):
    """Serve a raw image file (PNG/JPG) for preview."""
    img_path = RAW_ROOT / uc_type / lang / filename
    if not img_path.exists():
        raise HTTPException(404, f"Image not found: {img_path}")
    ext = img_path.suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "webp": "image/webp"}.get(ext.lstrip("."), "application/octet-stream")
    return FileResponse(str(img_path), media_type=mime)


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
    # Find source file — PDF or image
    _SOURCE_EXTS = [".pdf", ".png", ".jpg", ".jpeg", ".webp"]
    pdf_path: Path | None = None
    for ext in _SOURCE_EXTS:
        candidate = RAW_ROOT / req.uc_type / req.lang / f"{req.doc_id}{ext}"
        if candidate.exists():
            pdf_path = candidate
            break
    if pdf_path is None:
        raise HTTPException(404, f"Source file not found for {req.doc_id} in raw/{req.uc_type}/{req.lang}/")

    # Check if already done for THIS model
    model_subdir = _model_subdir(req.model)
    out_dir = RAW_ROOT / req.uc_type / req.lang / model_subdir

    text_pred   = out_dir / f"{req.doc_id}_text_prediction.json"
    unified_pred = out_dir / f"{req.doc_id}_prediction.json"
    already_done = text_pred.exists() or unified_pred.exists()

    if already_done and not req.force:
        return {
            "job_id": None,
            "status": "skipped",
            "message": f"OCR ({req.model}) already done. Use force=true to re-run.",
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
        _jobs[job_id]["message"] = f"Uploading to {req.model} API…"
        _jobs[job_id]["progress"] = 10

        # Import here to avoid circular at module load
        from ocr_benchmark.ocr_model.marker_convert import convert

        lang_map = {"vi": "vi,en", "en": "en", "ja": "ja,en"}
        langs = lang_map.get(req.lang, req.lang)

        out_dir = RAW_ROOT / req.uc_type / req.lang / _model_subdir(req.model)

        _jobs[job_id]["message"] = "Processing…"
        _jobs[job_id]["progress"] = 30

        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()

        if req.model == "mistral":
            from ocr_benchmark.ocr_model.mistral_convert import convert as mistral_convert
            outputs = await loop.run_in_executor(None, lambda: mistral_convert(
                source=str(pdf_path),
                output_dir=out_dir,
                langs=langs,
                mode=req.ocr_mode,
                uc_type="split",
                doc_id=req.doc_id,
                save_full=True,
                save_md=True,
                timeout=300,
                max_retries=3,
            ))
        else:
            # Default: Marker
            from ocr_benchmark.ocr_model.marker_convert import convert as marker_convert
            outputs = await loop.run_in_executor(None, lambda: marker_convert(
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
    """Poll OCR job status. If job not in memory (server restarted), check prediction files."""
    if job_id in _jobs:
        return _jobs[job_id]

    # Server restarted — try to recover status from prediction files
    # job_id format: {doc_id}_{model}_{timestamp}
    parts = job_id.rsplit("_", 2)
    if len(parts) == 3:
        doc_id_model, model, _ts = parts
        # Try to extract doc_id and model from job_id
        for m_name in MODEL_SUBDIR:
            if f"_{m_name}_" in job_id:
                model = m_name
                doc_id = job_id[:job_id.rfind(f"_{m_name}_")]
                subdir = _model_subdir(model)
                # Check all uc_type/lang combos
                for pred_file in RAW_ROOT.rglob(f"{subdir}/{doc_id}_*_prediction.json"):
                    return {
                        "status": "done",
                        "message": "OCR complete (recovered after server restart)",
                        "progress": 100,
                    }
                return {
                    "status": "error",
                    "message": "Server restarted during OCR — please re-run",
                    "progress": 0,
                }

    raise HTTPException(404, "Job not found")


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
        model_subdir = _model_subdir(req.model)
        text_pred_file = RAW_ROOT / req.uc_type / req.lang / model_subdir / f"{req.doc_id}_text_prediction.json"
        # Fallback to unified
        if not text_pred_file.exists():
            text_pred_file = RAW_ROOT / req.uc_type / req.lang / model_subdir / f"{req.doc_id}_prediction.json"
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

            # Use UET metrics for full metric set
            r = compute_all_metrics(gt_page, pred_page)
            r["doc_id"] = req.doc_id
            r["page_num"] = pnum

            if req.uc_type == "text_layer":
                iou_r = eval_text_layer(
                    gt_page,
                    pred_blocks=pred_page.get("blocks", []),
                    pred_full_text=pred_text,
                    doc_id=req.doc_id,
                )
                r["mean_iou"] = iou_r.get("mean_iou", 0)

            page_results.append(r)

        summary = {
            "avg_cer":                     _safe_mean([r.get("cer", 0)                        for r in page_results]),
            "avg_wer":                     _safe_mean([r.get("wer", 0)                        for r in page_results]),
            "avg_nwer":                    _safe_mean([r.get("nwer", 0)                       for r in page_results]),
            "avg_char_f1":                 _safe_mean([r.get("char_f1", 0)                    for r in page_results]),
            "avg_word_f1":                 _safe_mean([r.get("word_f1", 0)                    for r in page_results]),
            "avg_normalized_edit_similarity": _safe_mean([r.get("normalized_edit_similarity", 0) for r in page_results]),
            "n_pages":                     len(page_results),
        }
        if req.uc_type == "text_layer":
            summary["avg_mean_iou"] = _safe_mean([r.get("mean_iou", 0) for r in page_results])

        results["text"] = {"summary": summary, "pages": page_results}

    # ── table eval ────────────────────────────────────────────────────────
    if req.split in ("table", "both"):
        model_subdir = _model_subdir(req.model)
        table_pred_file = RAW_ROOT / req.uc_type / req.lang / model_subdir / f"{req.doc_id}_table_prediction.json"
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
                # UET table metrics (TEDS + cell metrics) từ HTML trực tiếp
                gt_html_list   = [t["html"] for p in gt_data.get("pages",[]) for t in (p.get("tables") or []) if t.get("html")]
                pred_html_list = [t["html"] for p in table_pred_data.get("pages",[]) for t in (p.get("tables") or []) if t.get("html")]
                uet_tbl = compute_table_metrics_from_html(gt_html_list, pred_html_list)

                results["table"] = {
                    "summary": {
                        "avg_teds":                      _safe_mean([r["avg_teds"] for r in table_results]),
                        "avg_cell_exact_f1":             uet_tbl.get("table_cell_exact_f1_mean", 0),
                        "avg_cell_text_similarity":      uet_tbl.get("table_cell_text_similarity_mean", 0),
                        "avg_row_count_similarity":      uet_tbl.get("table_row_count_similarity_mean", 0),
                        "avg_col_count_similarity":      uet_tbl.get("table_col_count_similarity_mean", 0),
                        "table_count_f1":                uet_tbl.get("table_count_f1", 0),
                        "n_pages": len(table_results),
                    },
                    "pages": table_results,
                    "uet": uet_tbl,
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


@router.get("/summary")
def get_eval_summary():
    """
    Aggregate all eval results into a dashboard summary.
    Returns per-model, per-UC-type averages for key metrics.
    """
    from statistics import mean as _mean

    # Collect all eval JSON files
    # Structure: benchmark_results/<model>/<uc_type>/<lang>/<doc_id>_eval.json
    all_results: list[dict] = []
    for eval_file in RESULT_ROOT.rglob("*_eval.json"):
        parts = eval_file.relative_to(RESULT_ROOT).parts
        if len(parts) < 4:
            continue
        model_name = parts[0]   # marker / mistral
        uc_type    = parts[1]   # scan / table / text_layer
        lang       = parts[2]   # en / vi / ja
        try:
            with open(eval_file, encoding="utf-8") as f:
                data = json.load(f)
            all_results.append({
                "model": model_name,
                "uc_type": uc_type,
                "lang": lang,
                "doc_id": data.get("doc_id", ""),
                "text": data.get("text", {}).get("summary", {}),
                "table": data.get("table", {}).get("summary", {}),
            })
        except Exception:
            continue

    if not all_results:
        return {"models": {}, "by_uc": {}, "total_docs": 0}

    # Helper: safe mean ignoring None/missing
    def smean(vals):
        v = [x for x in vals if x is not None]
        return round(_mean(v), 4) if v else None

    # Group by model
    models_data: dict[str, dict] = {}
    for r in all_results:
        m = r["model"]
        if m not in models_data:
            models_data[m] = {
                "docs": 0,
                "cer": [], "char_f1": [], "word_f1": [],
                "teds": [], "cell_exact_f1": [],
                "by_uc": {}
            }
        md = models_data[m]
        md["docs"] += 1

        uc = r["uc_type"]
        if uc not in md["by_uc"]:
            md["by_uc"][uc] = {
                "docs": 0,
                "cer": [], "char_f1": [], "word_f1": [],
                "teds": [], "cell_exact_f1": []
            }
        uc_d = md["by_uc"][uc]
        uc_d["docs"] += 1

        # Text metrics
        if r["text"]:
            t = r["text"]
            for key, lst in [
                ("avg_cer",        md["cer"]),
                ("avg_char_f1",    md["char_f1"]),
                ("avg_word_f1",    md["word_f1"]),
            ]:
                if t.get(key) is not None:
                    lst.append(t[key])
            for key, lst in [
                ("avg_cer",        uc_d["cer"]),
                ("avg_char_f1",    uc_d["char_f1"]),
                ("avg_word_f1",    uc_d["word_f1"]),
            ]:
                if t.get(key) is not None:
                    lst.append(t[key])

        # Table metrics
        if r["table"]:
            tb = r["table"]
            for key, lst in [
                ("avg_teds",             md["teds"]),
                ("avg_cell_exact_f1",    md["cell_exact_f1"]),
            ]:
                if tb.get(key) is not None:
                    lst.append(tb[key])
            for key, lst in [
                ("avg_teds",             uc_d["teds"]),
                ("avg_cell_exact_f1",    uc_d["cell_exact_f1"]),
            ]:
                if tb.get(key) is not None:
                    lst.append(tb[key])

    # Build summary output
    def _summarize(d: dict) -> dict:
        return {
            "docs":          d["docs"],
            "avg_cer":       smean(d["cer"]),
            "avg_char_f1":   smean(d["char_f1"]),
            "avg_word_f1":   smean(d["word_f1"]),
            "avg_teds":      smean(d["teds"]),
            "avg_cell_exact_f1": smean(d["cell_exact_f1"]),
        }

    summary: dict = {"models": {}, "total_docs": len(all_results)}
    for model_name, md in models_data.items():
        summary["models"][model_name] = {
            **_summarize(md),
            "by_uc": {uc: _summarize(uc_d) for uc, uc_d in md["by_uc"].items()},
        }

    return summary
