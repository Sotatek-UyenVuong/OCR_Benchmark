"""
upload_scorer.py
----------------
FastAPI router cho Upload & Score feature.

GET  /api/upload/gt_docs  — list tất cả GT docs có sẵn (để populate dropdown)
POST /api/upload/score    — upload folder .md files + model_name + doc_id → trả score
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .config import GT_ROOT

# Import scoring utilities (best-effort — graceful degradation if missing deps)
try:
    from ocr_benchmark.metrics.uet_metrics import compute_all_metrics
    _METRICS_AVAILABLE = True
except Exception:
    _METRICS_AVAILABLE = False

try:
    from ocr_benchmark.normalize import normalize_ocr_text
    _NORMALIZE_AVAILABLE = True
except Exception:
    _NORMALIZE_AVAILABLE = False

router = APIRouter(prefix="/api/upload", tags=["Upload & Score"])

# ── Regex for content filtering (pre-normalize_ocr_text) ──────────────────────
_FIGURE_RE = re.compile(r'<figure\b[^>]*>.*?</figure>', re.IGNORECASE | re.DOTALL)
_FIGCAPTION_RE = re.compile(r'<figcaption\b[^>]*>.*?</figcaption>', re.IGNORECASE | re.DOTALL)
_DIV_IMG_RE = re.compile(r'<div\b[^>]*>\s*<img\b[^>]*/?\s*>\s*</div>', re.IGNORECASE | re.DOTALL)

# Regex for parsing page filenames: {doc_id}_{page_index}.md
_PAGE_FILENAME_RE = re.compile(r'^(.+?)_(\d+)\.md$')

# Metrics to average at document level
_AVERAGED_METRICS = [
    "cer", "wer", "char_f1", "word_f1",
    "normalized_edit_similarity",
    "table_teds_doc", "table_cell_exact_f1_mean",
]


# ── Pydantic models ────────────────────────────────────────────────────────────

class GTDocEntry(BaseModel):
    doc_id: str
    uc_type: str
    lang: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _filter_content(text: str) -> str:
    """Remove image wrappers/figures not handled by normalize_ocr_text(), then normalize."""
    text = _FIGURE_RE.sub("", text)
    text = _FIGCAPTION_RE.sub("", text)
    text = _DIV_IMG_RE.sub("", text)
    if _NORMALIZE_AVAILABLE:
        text = normalize_ocr_text(text)
    return text


def _parse_doc_id(doc_id: str) -> tuple[str, str]:
    """Extract (uc_type, lang) from doc_id like 'scan_en_001'. Raises 422 if malformed."""
    parts = doc_id.split("_")
    if len(parts) < 3 or not parts[0] or not parts[1]:
        raise HTTPException(
            422,
            detail=f"Invalid doc_id format '{doc_id}'. Expected: {{uc_type}}_{{lang}}_{{seq}}"
        )
    return parts[0], parts[1]


def _safe_mean(vals: list) -> Optional[float]:
    valid = [v for v in vals if v is not None and isinstance(v, (int, float))]
    return round(sum(valid) / len(valid), 6) if valid else None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/gt_docs", response_model=list[GTDocEntry])
def list_gt_docs():
    """
    Return all documents that have a Ground Truth file.
    Sorted by uc_type → lang → doc_id.
    """
    entries: list[GTDocEntry] = []
    if not GT_ROOT.exists():
        return entries

    for path in GT_ROOT.rglob("*.json"):
        parts = path.relative_to(GT_ROOT).parts
        if len(parts) != 3:          # expect (uc_type, lang, filename)
            continue
        uc_type, lang = parts[0], parts[1]
        doc_id = path.stem
        entries.append(GTDocEntry(doc_id=doc_id, uc_type=uc_type, lang=lang))

    return sorted(entries, key=lambda e: (e.uc_type, e.lang, e.doc_id))


@router.post("/score")
async def score_upload(
    files: list[UploadFile] = File(..., description="Folder of .md files (one per page)"),
    model_name: str = Form(...),
    doc_id: str = Form(...),
):
    """
    Score uploaded .md prediction files against the server-side Ground Truth.

    - files: all files from webkitdirectory folder (non-.md files are silently ignored)
    - model_name: free-text name for display
    - doc_id: e.g. 'scan_en_001' — must match an existing GT file
    """
    # ── 1. Validate inputs ────────────────────────────────────────────────────
    if not model_name or not model_name.strip():
        raise HTTPException(422, detail="model_name must be a non-empty string")
    model_name = model_name.strip()[:100]

    if not doc_id or not doc_id.strip():
        raise HTTPException(422, detail="doc_id is required")
    doc_id = doc_id.strip()

    # ── 2. Filter to .md files only ───────────────────────────────────────────
    md_files = [f for f in files if (f.filename or "").lower().endswith(".md")]
    if len(md_files) == 0:
        raise HTTPException(422, detail="No .md files found in the uploaded folder.")
    if len(md_files) > 500:
        raise HTTPException(422, detail=f"File limit exceeded: maximum 500 .md files allowed (got {len(md_files)})")

    # ── 3. Derive uc_type + lang from doc_id ─────────────────────────────────
    uc_type, lang = _parse_doc_id(doc_id)

    # ── 4. Load and validate Ground Truth ────────────────────────────────────
    gt_path = GT_ROOT / uc_type / lang / f"{doc_id}.json"
    if not gt_path.exists():
        raise HTTPException(
            404,
            detail=f"Ground Truth not found for '{doc_id}' "
                   f"(expected: ground_truth/{uc_type}/{lang}/{doc_id}.json)"
        )
    try:
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(422, detail=f"Ground Truth file for '{doc_id}' is corrupt or invalid JSON")

    gt_pages_raw = gt_data.get("pages")
    if not gt_pages_raw:
        raise HTTPException(422, detail=f"Ground Truth for '{doc_id}' has no pages")

    # ── 5. Parse prediction filenames and detect base_index ──────────────────
    parsed_pages: list[dict] = []
    for f in md_files:
        filename = Path(f.filename or "").name  # strip any folder prefix from webkitdirectory
        m = _PAGE_FILENAME_RE.match(filename)
        if not m:
            continue  # silently discard non-matching files
        prefix, idx_str = m.group(1), m.group(2)
        # Accept files with exact doc_id prefix OR any prefix (be permissive)
        page_index = int(idx_str)
        content_bytes = await f.read()
        content = content_bytes.decode("utf-8", errors="replace")
        parsed_pages.append({"page_index": page_index, "raw_content": content, "filename": filename})

    if not parsed_pages:
        raise HTTPException(
            422,
            detail=f"No valid page files found. Files must be named like: {doc_id}_0.md, {doc_id}_1.md, etc."
        )

    # Detect base_index (0-indexed or 1-indexed)
    base_index = min(p["page_index"] for p in parsed_pages)

    # Compute page_num = page_index - base_index + 1
    for p in parsed_pages:
        p["page_num"] = p["page_index"] - base_index + 1

    # Sort and deduplicate: keep first occurrence per page_num
    parsed_pages.sort(key=lambda p: p["page_index"])
    seen_page_nums: set[int] = set()
    deduped: list[dict] = []
    for p in parsed_pages:
        if p["page_num"] not in seen_page_nums:
            seen_page_nums.add(p["page_num"])
            deduped.append(p)

    # ── 6. Apply content filter ───────────────────────────────────────────────
    for p in deduped:
        p["filtered_text"] = _filter_content(p["raw_content"])

    # Build lookup by page_num
    pred_by_num: dict[int, dict] = {p["page_num"]: p for p in deduped}

    # ── 7. Score each GT page ─────────────────────────────────────────────────
    page_results: list[dict] = []
    n_matched = 0

    if not _METRICS_AVAILABLE:
        return JSONResponse({
            "model": model_name,
            "doc_id": doc_id,
            "uc_type": uc_type,
            "lang": lang,
            "error": "Scoring dependencies not available (uet_metrics import failed)",
            "results": {"text": {"summary": {"n_pages": len(gt_pages_raw), "n_matched_pages": 0}, "pages": []}},
        })

    for gt_page in gt_pages_raw:
        pnum = gt_page.get("page_num", 1)
        pred_entry = pred_by_num.get(pnum)

        if pred_entry is not None:
            n_matched += 1
            pred_page = {"full_text": pred_entry["filtered_text"], "tables": []}
        else:
            pred_page = {"full_text": "", "tables": []}

        try:
            metrics = compute_all_metrics(gt_page, pred_page)
        except Exception as exc:
            metrics = {"page_num": pnum, "error": str(exc)[:200]}

        metrics["page_num"] = pnum
        page_results.append(metrics)

    # ── 8. Compute document-level averages ────────────────────────────────────
    summary: dict = {"n_pages": len(gt_pages_raw), "n_matched_pages": n_matched}
    for metric in _AVERAGED_METRICS:
        summary[metric] = _safe_mean([r.get(metric) for r in page_results])

    return {
        "model": model_name,
        "doc_id": doc_id,
        "uc_type": uc_type,
        "lang": lang,
        "results": {
            "text": {
                "summary": summary,
                "pages": page_results,
            }
        },
    }
