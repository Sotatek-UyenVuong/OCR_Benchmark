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

    # Also yield image-based docs (PNG/JPG) that don't have a prediction JSON yet
    # raw/<uc_type>/<lang>/<doc_id>.png or .jpg  (not inside any subfolder)
    _IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
    seen_doc_ids: set[str] = set()
    for img_file in sorted(RAW_ROOT.rglob("*")):
        if img_file.suffix.lower() not in _IMG_EXTS:
            continue
        parts = img_file.relative_to(RAW_ROOT).parts
        # Must be at depth 2: raw/<uc_type>/<lang>/<file.png>
        if len(parts) != 3:
            continue
        uc_type, lang, fname = parts
        doc_id = img_file.stem  # e.g. "scan_ko_001"
        key = f"{uc_type}/{lang}/{doc_id}"
        if key in seen_doc_ids:
            continue
        seen_doc_ids.add(key)

        gt_path = GT_ROOT / uc_type / lang / f"{doc_id}.json"
        gt_saved = gt_path.exists()
        gt_status = gt_reviewer = gt_updated = ""
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

        yield {
            "doc_id":     doc_id,
            "uc_type":    uc_type,
            "lang":       lang,
            "split":      "image",
            "draft_path": str(img_file.relative_to(PROJECT_ROOT)),
            "pdf_exists": False,
            "gt_saved":   gt_saved,
            "gt_status":  gt_status,
            "gt_reviewer": gt_reviewer,
            "gt_updated": gt_updated,
            "gt_path":    str(gt_path.relative_to(PROJECT_ROOT)) if gt_saved else None,
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


import subprocess
import tempfile
import glob as _glob
from pathlib import Path as _Path
from fastapi.responses import Response

# ── PDF PNG Cache (disk-backed) ───────────────────────────────────────────────
# Disk cache: RAW_ROOT/../.pdf_cache/<uc_type>/<lang>/<doc_id>/dpi<N>/page-NNN.png
# Survives server restarts; only re-renders if PDF mtime changes.

def _disk_cache_dir(uc_type: str, lang: str, doc_id: str, dpi: int) -> _Path:
    cache_root = PROJECT_ROOT / ".pdf_cache" / uc_type / lang / doc_id / f"dpi{dpi}"
    return cache_root

def _render_pdf_pages(pdf_path, dpi: int, cache_key: str) -> dict[int, bytes]:
    """
    Render all pages of a PDF to PNG using pdftoppm.
    Disk-cached: writes PNGs to .pdf_cache/; survives server restarts.
    Only re-renders when PDF mtime changes (stored in .mtime sentinel file).
    """
    uc_type, lang, doc_id = cache_key.split("/")[:3]   # "uc/lang/doc/dpi144"
    cache_dir = _disk_cache_dir(uc_type, lang, doc_id, dpi)

    # Check mtime sentinel — invalidate if PDF changed
    mtime_file = cache_dir / ".mtime"
    pdf_mtime  = str(pdf_path.stat().st_mtime) if pdf_path.exists() else ""
    cache_valid = (
        cache_dir.exists()
        and mtime_file.exists()
        and mtime_file.read_text().strip() == pdf_mtime
    )

    if not cache_valid:
        # Re-render
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_prefix = f"{tmpdir}/page"
            result = subprocess.run(
                ["pdftoppm", "-r", str(dpi), "-png", str(pdf_path), out_prefix],
                capture_output=True,
                timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(f"pdftoppm failed: {result.stderr.decode()[:200]}")

            import re as _re
            for fpath in sorted(_glob.glob(f"{tmpdir}/page*.png")):
                m = _re.search(r'page-?(\d+)\.png$', fpath)
                pn = int(m.group(1)) if m else 0
                dest = cache_dir / f"page-{pn:03d}.png"
                import shutil as _shutil
                _shutil.copy2(fpath, dest)

        # Write mtime sentinel
        mtime_file.write_text(pdf_mtime)

    # Read from disk cache
    pages: dict[int, bytes] = {}
    import re as _re2
    for fpath in sorted(cache_dir.glob("page-*.png")):
        m = _re2.search(r'page-(\d+)\.png$', fpath.name)
        if m:
            pages[int(m.group(1))] = fpath.read_bytes()
    return pages


def _get_page_count_fast(pdf_path) -> int:
    """Get page count cheaply via pdfinfo (no rendering)."""
    try:
        r = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True, timeout=10
        )
        if r.returncode == 0:
            for line in r.stdout.decode().splitlines():
                if line.startswith("Pages:"):
                    return int(line.split(":")[1].strip())
    except Exception:
        pass
    # Fallback: count cached pages if available
    return 0


@router.get("/pdf/{uc_type}/{lang}/{filename}")
def serve_pdf(uc_type: str, lang: str, filename: str):
    """Serve the raw PDF file for display in the browser."""
    pdf_path = RAW_ROOT / uc_type / lang / filename
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise HTTPException(404, "PDF not found")
    return FileResponse(str(pdf_path), media_type="application/pdf")


@router.get("/pdf_page/{uc_type}/{lang}/{doc_id}/{page_num}")
def get_pdf_page_image(
    uc_type: str, lang: str, doc_id: str, page_num: int,
    dpi: int = 144,
):
    """Serve a PDF page as PNG. Disk-cached; fast after first render."""
    pdf_path = RAW_ROOT / uc_type / lang / f"{doc_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "PDF not found")

    cache_key = f"{uc_type}/{lang}/{doc_id}/dpi{dpi}"

    # Check disk cache first (no subprocess needed)
    cache_dir = _disk_cache_dir(uc_type, lang, doc_id, dpi)
    cached_file = cache_dir / f"page-{page_num:03d}.png"
    mtime_file = cache_dir / ".mtime"
    pdf_mtime = str(pdf_path.stat().st_mtime)
    if (cached_file.exists()
            and mtime_file.exists()
            and mtime_file.read_text().strip() == pdf_mtime):
        return Response(
            content=cached_file.read_bytes(),
            media_type="image/png",
            headers={"Cache-Control": "max-age=86400", "ETag": f'"{pdf_mtime}-p{page_num}"'},
        )

    # Not cached — render all pages
    try:
        pages = _render_pdf_pages(pdf_path, dpi, cache_key)
    except Exception as e:
        raise HTTPException(500, str(e))

    if page_num not in pages:
        raise HTTPException(404, f"Page {page_num} not found (total: {len(pages)})")

    return Response(
        content=pages[page_num],
        media_type="image/png",
        headers={"Cache-Control": "max-age=86400", "ETag": f'"{pdf_mtime}-p{page_num}"'},
    )


@router.get("/pdf_info/{uc_type}/{lang}/{doc_id}")
def get_pdf_info(uc_type: str, lang: str, doc_id: str, dpi: int = 144):
    """Return page count. Uses pdfinfo for fast count; triggers background render."""
    pdf_path = RAW_ROOT / uc_type / lang / f"{doc_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "PDF not found")

    # Fast path: pdfinfo (no rendering)
    page_count = _get_page_count_fast(pdf_path)

    # Also check disk cache count (if pdfinfo unavailable)
    if not page_count:
        cache_dir = _disk_cache_dir(uc_type, lang, doc_id, dpi)
        if cache_dir.exists():
            page_count = len(list(cache_dir.glob("page-*.png")))

    # Trigger full render in background if not cached yet
    if not page_count or not (_disk_cache_dir(uc_type, lang, doc_id, dpi) / ".mtime").exists():
        import threading
        cache_key = f"{uc_type}/{lang}/{doc_id}/dpi{dpi}"
        def _bg_render():
            try: _render_pdf_pages(pdf_path, dpi, cache_key)
            except Exception: pass
        threading.Thread(target=_bg_render, daemon=True).start()

        # If we have page count from pdfinfo, return it immediately
        if page_count:
            return {"page_count": page_count, "rendering": True}

        # No pdfinfo available — need to wait for render
        cache_key = f"{uc_type}/{lang}/{doc_id}/dpi{dpi}"
        try:
            pages = _render_pdf_pages(pdf_path, dpi, cache_key)
            return {"page_count": len(pages)}
        except Exception as e:
            raise HTTPException(500, str(e))

    return {"page_count": page_count}


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


def _get_image_exif_orientation(img_path: Path) -> int:
    """Return EXIF orientation tag value (1-8), or 1 if absent/unreadable.

    Uses a pure-stdlib JPEG EXIF parser so it works even without Pillow.
    Falls back to Pillow if available (for other formats like TIFF/WEBP).
    """
    try:
        # --- Fast path: try Pillow first (supports all formats) ---
        from PIL import Image as _PILImage
        with _PILImage.open(img_path) as im:
            exif = im.getexif()
            if exif:
                return exif.get(274, 1)
    except Exception:
        pass

    # --- Fallback: pure-stdlib JPEG EXIF reader ---
    try:
        with open(img_path, "rb") as fh:
            return _read_jpeg_exif_orientation(fh)
    except Exception:
        pass

    return 1


def _read_jpeg_exif_orientation(fh) -> int:
    """Minimal JPEG/EXIF parser — reads Orientation tag (0x0112) without Pillow."""
    import struct

    sig = fh.read(2)
    if sig != b"\xff\xd8":          # Not JPEG
        return 1

    while True:
        marker = fh.read(2)
        if len(marker) < 2:
            break
        if marker[0] != 0xff:
            break

        # APP1 marker = 0xff 0xe1
        if marker == b"\xff\xe1":
            seg_len = struct.unpack(">H", fh.read(2))[0]
            app1_data = fh.read(seg_len - 2)

            # Check Exif header
            if app1_data[:6] != b"Exif\x00\x00":
                continue

            tiff = app1_data[6:]
            byte_order = tiff[:2]
            if byte_order == b"II":
                endian = "<"
            elif byte_order == b"MM":
                endian = ">"
            else:
                break

            # IFD0 offset
            ifd_offset = struct.unpack(endian + "I", tiff[4:8])[0]
            num_entries = struct.unpack(endian + "H",
                                        tiff[ifd_offset: ifd_offset + 2])[0]
            for i in range(num_entries):
                entry_offset = ifd_offset + 2 + i * 12
                tag = struct.unpack(endian + "H", tiff[entry_offset: entry_offset + 2])[0]
                if tag == 0x0112:  # Orientation
                    value = struct.unpack(endian + "H",
                                          tiff[entry_offset + 8: entry_offset + 10])[0]
                    return value
            return 1
        else:
            # Skip segment
            seg_len_bytes = fh.read(2)
            if len(seg_len_bytes) < 2:
                break
            seg_len = struct.unpack(">H", seg_len_bytes)[0]
            fh.seek(seg_len - 2, 1)

    return 1


def _rotate_bbox_norm(bbox: list[float], orientation: int) -> list[float]:
    """
    Transform a normalized bbox [x1,y1,x2,y2] from Marker/OCR coordinate space
    to browser-display coordinate space, compensating for EXIF orientation.

    Marker processes the raw pixel data (ignoring EXIF), so its coords are in
    the raw-image coordinate system.  The browser (and img.clientWidth/Height)
    uses the EXIF-corrected display orientation.

    EXIF orientation values that involve rotation:
      6 = 90° CW  (most common "portrait photo taken landscape")
      8 = 90° CCW
      3 = 180°

    For EXIF 6 (90° CW): rotate raw landscape → display portrait
      raw point (rx_norm, ry_norm) → display (1-ry_norm, rx_norm)
      bbox [x1,y1,x2,y2] → [1-y2, x1, 1-y1, x2]
    """
    x1, y1, x2, y2 = bbox
    if orientation == 6:
        # 90° CW: new_x = 1-y_old, new_y = x_old
        return [1 - y2, x1, 1 - y1, x2]
    elif orientation == 8:
        # 90° CCW: new_x = y_old, new_y = 1-x_old
        return [y1, 1 - x2, y2, 1 - x1]
    elif orientation == 3:
        # 180°
        return [1 - x2, 1 - y2, 1 - x1, 1 - y1]
    # 1, 2, 4, 5, 7 — no rotation needed (or flip only)
    return bbox


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

    # Detect EXIF orientation for image-based docs (PNG/JPG, no PDF)
    # Only needed when the source is a raw image (not a PDF rendered to image)
    _IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
    exif_orientation = 1
    for ext in _IMG_EXTS:
        img_path = RAW_ROOT / uc_type / lang / f"{doc_id}{ext}"
        if img_path.exists():
            exif_orientation = _get_image_exif_orientation(img_path)
            break

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
                bbox_norm = _rotate_bbox_norm(bbox_norm, exif_orientation)
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
            # Compensate for EXIF orientation: Marker ignores EXIF, browser applies it
            bbox_norm = _rotate_bbox_norm(bbox_norm, exif_orientation)

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
