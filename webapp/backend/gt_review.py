"""
gt_review.py
------------
FastAPI router cho GT Review Tool.

Endpoints:
  GET  /api/gt/files              — liệt kê tất cả draft prediction files
  GET  /api/gt/load               — load 1 draft file + PDF + bbox annotations
  GET  /api/gt/pdf/{doc_path}     — serve PDF file để hiển thị trên web
  GET  /api/gt/bboxes             — trả bbox của tất cả blocks trong 1 doc
  POST /api/gt/save               — lưu GT đã sửa vào ground_truth/
  GET  /api/gt/status             — tổng quan review progress
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ── Paths: import từ config để dùng DATA_ROOT env var khi deploy ──
from .config import PROJECT_ROOT, RAW_ROOT, GT_ROOT, MARKER_SUBDIR

router = APIRouter(prefix="/api/gt", tags=["GT Review"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iter_draft_files():
    """
    Scan raw/ for *_text_prediction.json and *_table_prediction.json files.
    Yields dicts with metadata.
    """
    for pred_file in sorted(RAW_ROOT.rglob(f"{MARKER_SUBDIR}/*_prediction.json")):
        # Resolve UC type and language from folder path
        # raw/<uc_type>/<lang>/marker_output/<stem>_prediction.json
        parts = pred_file.relative_to(RAW_ROOT).parts
        if len(parts) < 3:
            continue
        uc_type, lang = parts[0], parts[1]   # e.g. "scan", "en"
        stem = pred_file.stem  # e.g. "scan_en_001_text_prediction"

        # Determine split type (text / table / unified)
        if stem.endswith("_text_prediction"):
            split = "text"
            doc_id = stem.replace("_text_prediction", "")
        elif stem.endswith("_table_prediction"):
            split = "table"
            doc_id = stem.replace("_table_prediction", "")
        else:
            split = "unified"
            doc_id = stem.replace("_prediction", "")

        # Check if GT already saved
        gt_path = GT_ROOT / uc_type / lang / f"{doc_id}.json"
        gt_saved = gt_path.exists()
        gt_status = ""
        gt_reviewer = ""
        gt_updated = ""
        if gt_saved:
            try:
                import json as _json
                with open(gt_path, encoding="utf-8") as _f:
                    _gt = _json.load(_f)
                gt_status   = _gt.get("status", "")
                gt_reviewer = _gt.get("reviewer", "")
                gt_updated  = _gt.get("updated_at", "")
            except Exception:
                pass

        # Corresponding PDF
        pdf_path = RAW_ROOT / uc_type / lang / f"{doc_id}.pdf"

        yield {
            "doc_id": doc_id,
            "uc_type": uc_type,
            "lang": lang,
            "split": split,
            "draft_path": str(pred_file.relative_to(PROJECT_ROOT)),
            "pdf_exists": pdf_path.exists(),
            "gt_saved": gt_saved,
            "gt_status": gt_status,
            "gt_reviewer": gt_reviewer,
            "gt_updated": gt_updated,
            "gt_path": str(gt_path.relative_to(PROJECT_ROOT)) if gt_saved else None,
        }


# ── Models ────────────────────────────────────────────────────────────────────

class SaveGTRequest(BaseModel):
    doc_id: str
    uc_type: str
    lang: str
    gt_data: dict           # full GT JSON matching benchmark schema
    reviewer: str = ""      # tên người review
    status: str = "in_progress"  # "in_progress" | "done"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/files")
def list_draft_files():
    """List all draft prediction files available for review."""
    files = list(_iter_draft_files())
    total      = len(files)
    reviewed   = sum(1 for f in files if f["gt_saved"])
    pending    = total - reviewed
    return {
        "total": total,
        "reviewed": reviewed,
        "pending": pending,
        "files": files,
    }


@router.get("/load")
def load_draft(
    doc_id:  str = Query(...),
    uc_type: str = Query(...),
    lang:    str = Query(...),
    split:   str = Query(default="text"),
    model:   str = Query(default="marker"),
):
    """Load a draft prediction file and its corresponding GT (if exists)."""
    # Resolve subdir based on model
    from webapp.backend.ocr_runner import _model_subdir
    subdir = _model_subdir(model)

    if split == "text":
        draft_name = f"{doc_id}_text_prediction.json"
    elif split == "table":
        draft_name = f"{doc_id}_table_prediction.json"
    else:
        draft_name = f"{doc_id}_prediction.json"

    draft_path = RAW_ROOT / uc_type / lang / subdir / draft_name

    if not draft_path.exists():
        raise HTTPException(404, f"Draft not found: {draft_path.relative_to(PROJECT_ROOT)}")

    with open(draft_path, encoding="utf-8") as f_:
        draft_data = json.load(f_)

    # Load existing GT if already saved (unified file containing both text + tables)
    gt_path = GT_ROOT / uc_type / lang / f"{doc_id}.json"
    existing_gt = None
    if gt_path.exists():
        with open(gt_path, encoding="utf-8") as f_:
            existing_gt = json.load(f_)

    # Check PDF
    pdf_path = RAW_ROOT / uc_type / lang / f"{doc_id}.pdf"
    pdf_url = f"/api/gt/pdf/{uc_type}/{lang}/{doc_id}.pdf" if pdf_path.exists() else None

    return {
        "doc_id": doc_id,
        "uc_type": uc_type,
        "lang": lang,
        "split": split,
        "model": model,
        "draft": draft_data,
        "existing_gt": existing_gt,
        "pdf_url": pdf_url,
        "gt_saved": gt_path.exists(),
    }


@router.get("/pdf/{uc_type}/{lang}/{filename}")
def serve_pdf(uc_type: str, lang: str, filename: str):
    """Serve the raw PDF file for display in the browser."""
    pdf_path = RAW_ROOT / uc_type / lang / filename
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise HTTPException(404, "PDF not found")
    return FileResponse(str(pdf_path), media_type="application/pdf")


@router.post("/save")
def save_gt(req: SaveGTRequest):
    """Save reviewed GT JSON to ground_truth/<uc_type>/<lang>/<doc_id>.json."""
    if "pages" not in req.gt_data:
        raise HTTPException(400, "GT data must have 'pages' key")

    gt_dir = GT_ROOT / req.uc_type / req.lang
    gt_dir.mkdir(parents=True, exist_ok=True)

    import datetime
    req.gt_data["doc_id"]   = req.doc_id
    req.gt_data["reviewer"] = req.reviewer
    req.gt_data["status"]   = req.status   # "in_progress" | "done"
    req.gt_data["updated_at"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    gt_path = gt_dir / f"{req.doc_id}.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(req.gt_data, f, ensure_ascii=False, indent=2)

    return {
        "success": True,
        "saved_to": str(gt_path.relative_to(PROJECT_ROOT)),
        "doc_id": req.doc_id,
        "status": req.status,
        "reviewer": req.reviewer,
    }


@router.get("/md/{uc_type}/{lang}/{doc_id}")
def get_markdown(uc_type: str, lang: str, doc_id: str, model: str = "marker"):
    """Serve raw Markdown output from OCR for display."""
    subdir = {"marker": "marker_output", "mistral": "mistral_output"}.get(model, "marker_output")
    md_path = RAW_ROOT / uc_type / lang / subdir / f"{doc_id}.md"
    if not md_path.exists():
        raise HTTPException(404, f"Markdown file not found for model={model} — run OCR first")
    content = md_path.read_text(encoding="utf-8")
    return {"doc_id": doc_id, "model": model, "markdown": content, "path": str(md_path.relative_to(PROJECT_ROOT))}


@router.get("/full_response")
def get_full_response(
    doc_id:  str = Query(...),
    uc_type: str = Query(...),
    lang:    str = Query(...),
    model:   str = Query(default="marker"),
):
    """Serve full OCR response JSON — used by frontend to get authoritative tbl-N.html IDs."""
    subdir = {"marker": "marker_output", "mistral": "mistral_output"}.get(model, "marker_output")
    full_path = RAW_ROOT / uc_type / lang / subdir / f"{doc_id}_full_response.json"
    if not full_path.exists():
        raise HTTPException(404, f"full_response not found for model={model}")
    with open(full_path, encoding="utf-8") as f:
        return json.load(f)


@router.get("/bboxes")
def get_bboxes(
    doc_id:  str = Query(...),
    uc_type: str = Query(...),
    lang:    str = Query(...),
    model:   str = Query(default="marker"),
):
    """Return bbox annotations from OCR full_response.json (Marker and Mistral)."""
    subdir = {"marker": "marker_output", "mistral": "mistral_output"}.get(model, "marker_output")
    full_path = RAW_ROOT / uc_type / lang / subdir / f"{doc_id}_full_response.json"
    if not full_path.exists():
        return {"pages": {}}

    with open(full_path, encoding="utf-8") as f:
        full = json.load(f)

    import re as _re

    # ── Mistral format ────────────────────────────────────────────────────────
    if model == "mistral":
        SKIP_MISTRAL = {"image", "figure"}
        pages_out: dict[str, list] = {}
        for page in (full.get("pages") or []):
            page_num = int(page.get("index", 0)) + 1
            dims = page.get("dimensions") or {}
            pw = dims.get("width") or 1
            ph = dims.get("height") or 1
            blocks = []
            for i, blk in enumerate(page.get("blocks") or []):
                bt = blk.get("type", "text")
                if bt.lower() in SKIP_MISTRAL:
                    continue
                x1 = blk.get("top_left_x", 0)
                y1 = blk.get("top_left_y", 0)
                x2 = blk.get("bottom_right_x", 0)
                y2 = blk.get("bottom_right_y", 0)
                bbox_norm = [
                    round(x1 / pw, 4), round(y1 / ph, 4),
                    round(x2 / pw, 4), round(y2 / ph, 4),
                ]
                content = blk.get("content", "")
                # Strip HTML tags for text preview
                text = _re.sub(r"<[^>]+>", " ", content)
                text = _re.sub(r"\s+", " ", text).strip()[:120]
                blocks.append({
                    "block_id": i,
                    "type": bt,
                    "bbox": bbox_norm,
                    "text": text,
                })
            if blocks:
                pages_out[str(page_num)] = blocks
        return {"pages": pages_out}

    # ── Marker format ─────────────────────────────────────────────────────────
    SKIP = {"Picture", "Figure", "Image", "FigureGroup", "PictureGroup",
            "Document", "Page", "Line", "Span"}

    marker_json = full.get("json", {})
    page_nodes = marker_json.get("children") or []
    pages_out: dict[str, list] = {}

    for pi, page_node in enumerate(page_nodes):
        page_num = pi + 1
        pw = (page_node.get("bbox") or [0, 0, 1, 1])[2] or 1
        ph = (page_node.get("bbox") or [0, 0, 1, 1])[3] or 1
        blocks = []

        for i, blk in enumerate(page_node.get("children") or []):
            bt = blk.get("block_type", "Text")
            if bt in SKIP:
                continue
            bbox_abs = blk.get("bbox") or [0, 0, 0, 0]
            bbox_norm = [
                round(bbox_abs[0] / pw, 4),
                round(bbox_abs[1] / ph, 4),
                round(bbox_abs[2] / pw, 4),
                round(bbox_abs[3] / ph, 4),
            ]
            html = blk.get("html", "")
            text = _re.sub(r"<img\b[^>]*>", "", html, flags=_re.I)
            text = _re.sub(r"<[^>]+>", " ", text)
            text = _re.sub(r"\s+", " ", text).strip()[:120]

            blocks.append({
                "block_id": i,
                "type": bt,
                "bbox": bbox_norm,
                "text": text,
            })

        if blocks:
            pages_out[str(page_num)] = blocks

    return {"pages": pages_out}


@router.delete("/gt/{uc_type}/{lang}/{doc_id}")
def delete_gt(uc_type: str, lang: str, doc_id: str):
    """Xóa GT đã lưu để reset về draft mới."""
    gt_path = GT_ROOT / uc_type / lang / f"{doc_id}.json"
    if not gt_path.exists():
        raise HTTPException(404, "GT not found")
    gt_path.unlink()
    return {"success": True, "deleted": str(gt_path.relative_to(PROJECT_ROOT))}



def review_status():
    """Overview of review progress per UC."""
    from collections import defaultdict
    files = list(_iter_draft_files())

    by_uc: dict = defaultdict(lambda: {"total": 0, "reviewed": 0, "files": []})
    for f in files:
        key = f"{f['uc_type']}/{f['lang']}"
        by_uc[key]["total"] += 1
        if f["gt_saved"]:
            by_uc[key]["reviewed"] += 1
        by_uc[key]["files"].append(f)

    return {
        "overall": {
            "total": len(files),
            "reviewed": sum(1 for f in files if f["gt_saved"]),
        },
        "by_uc": dict(by_uc),
    }
