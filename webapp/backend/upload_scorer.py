"""
upload_scorer.py
----------------
FastAPI router cho Upload & Score feature.

GET  /api/upload/gt_docs      — list tất cả GT docs có sẵn
GET  /api/upload/known_models — list model names đã từng upload
GET  /api/upload/doc_result   — load saved result cho 1 doc+model
GET  /api/upload/leaderboard  — bảng xếp hạng (>= min_docs)
POST /api/upload/score        — upload .md files → score + save
"""

from __future__ import annotations

import datetime
import json
import re
import sys
from pathlib import Path
from statistics import mean as _mean
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .config import GT_ROOT, RESULT_ROOT

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

# ── Regex ─────────────────────────────────────────────────────────────────────
_FIGURE_RE = re.compile(r'<figure\b[^>]*>.*?</figure>', re.IGNORECASE | re.DOTALL)
_FIGCAPTION_RE = re.compile(r'<figcaption\b[^>]*>.*?</figcaption>', re.IGNORECASE | re.DOTALL)
_DIV_IMG_RE = re.compile(r'<div\b[^>]*>\s*<img\b[^>]*/?\s*>\s*</div>', re.IGNORECASE | re.DOTALL)
_TABLE_RE = re.compile(r'<table\b[^>]*>.*?</table>', re.IGNORECASE | re.DOTALL)
_PAGE_FILENAME_RE = re.compile(r'^(.+?)_(\d+)\.md$')

_AVERAGED_METRICS = [
    "cer", "wer", "char_f1", "word_f1",
    "normalized_edit_similarity",
    "table_teds_doc", "table_cell_exact_f1_mean",
]

_MODELS_FILE = _PROJECT_ROOT / "benchmark_results" / ".upload_models.json"


# ── Pydantic ──────────────────────────────────────────────────────────────────

class GTDocEntry(BaseModel):
    doc_id: str
    uc_type: str
    lang: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_known_models() -> list[str]:
    try:
        if _MODELS_FILE.exists():
            return json.loads(_MODELS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _save_model_name(name: str) -> None:
    try:
        known = _load_known_models()
        if name not in known:
            known.append(name)
            known.sort()
            _MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
            _MODELS_FILE.write_text(json.dumps(known, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _result_path(model: str, uc_type: str, lang: str, doc_id: str) -> Path:
    return RESULT_ROOT / model / uc_type / lang / f"{doc_id}_eval.json"


def _save_result(model: str, uc_type: str, lang: str, doc_id: str, result: dict) -> None:
    path = _result_path(model, uc_type, lang, doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "doc_id": doc_id, "model": model,
        "source": "upload",
        "scored_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        **result,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_html_tables(raw_md: str) -> list[dict]:
    return [{"table_id": i, "html": m.group(0)} for i, m in enumerate(_TABLE_RE.finditer(raw_md), start=1)]


def _filter_content(text: str) -> str:
    text = _FIGURE_RE.sub("", text)
    text = _FIGCAPTION_RE.sub("", text)
    text = _DIV_IMG_RE.sub("", text)
    text = _TABLE_RE.sub("", text)
    if _NORMALIZE_AVAILABLE:
        text = normalize_ocr_text(text)
    return text


def _parse_doc_id(doc_id: str) -> tuple[str, str]:
    parts = doc_id.split("_")
    if len(parts) < 3 or not parts[0] or not parts[1]:
        raise HTTPException(422, detail=f"Invalid doc_id '{doc_id}'. Expected: {{uc_type}}_{{lang}}_{{seq}}")
    # Text-layer document IDs use the compact "textlayer_*" prefix while the
    # dataset directory is named "text_layer".
    uc_type = "text_layer" if parts[0] == "textlayer" else parts[0]
    return uc_type, parts[1]


def _safe_mean(vals: list) -> Optional[float]:
    valid = [v for v in vals if v is not None and isinstance(v, (int, float))]
    return round(sum(valid) / len(valid), 6) if valid else None


# ── Evidence Collection ────────────────────────────────────────────────────────

import difflib as _difflib
import logging as _logging

_logger = _logging.getLogger(__name__)

_VALID_METRICS = {
    "cer", "wer", "char_f1", "word_f1",
    "normalized_edit_similarity", "table_teds_doc", "table_cell_exact_f1_mean",
}


def collect_evidence(gt_page: dict, pred_page: dict, metrics: dict) -> dict:
    """
    Extract per-page evidence from GT and prediction content.
    Stored as pages[i]._evidence in eval JSON for later agent retrieval.

    Returns dict with:
      gt_text         : str (≤400 Unicode chars, normalized)
      pred_text       : str (≤400 Unicode chars, normalized)
      inserted_chunks : list[str] — top-4 word sequences pred added
      deleted_chunks  : list[str] — top-4 word sequences pred removed
      structural_flags: {has_table: bool, figure_mentions: bool}
      candidate_signals: list[str] — up to 5 diagnostic strings
    """
    # ── Normalize texts ────────────────────────────────────────────────────
    gt_raw   = gt_page.get("full_text", "") or ""
    pred_raw = pred_page.get("full_text", "") or ""

    if _NORMALIZE_AVAILABLE:
        gt_norm   = normalize_ocr_text(gt_raw)
        pred_norm = normalize_ocr_text(pred_raw)
    else:
        gt_norm   = gt_raw
        pred_norm = pred_raw

    gt_text   = gt_norm[:400]
    pred_text = pred_norm[:400]

    # ── Word-level diff ────────────────────────────────────────────────────
    gt_words   = gt_norm.split()
    pred_words = pred_norm.split()
    sm = _difflib.SequenceMatcher(None, gt_words, pred_words, autojunk=False)

    inserted_all: list[str] = []
    deleted_all:  list[str] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace") and j2 > j1:
            inserted_all.append(" ".join(pred_words[j1:j2]))
        if tag in ("delete", "replace") and i2 > i1:
            deleted_all.append(" ".join(gt_words[i1:i2]))

    # Top-4 by length (most informative chunks first)
    inserted_chunks = sorted(inserted_all, key=len, reverse=True)[:4]
    deleted_chunks  = sorted(deleted_all,  key=len, reverse=True)[:4]

    ins_words = sum(len(c.split()) for c in inserted_all)
    del_words = sum(len(c.split()) for c in deleted_all)
    gt_len    = max(len(gt_norm), 1)
    pred_len  = max(len(pred_norm), 1)
    size_ratio = pred_len / gt_len

    # ── Structural flags ────────────────────────────────────────────────────
    # Use raw pred markdown if available, else fall back to pred_text
    pred_md_raw = pred_page.get("_raw_content", pred_norm)
    has_table      = bool(re.search(r'<table\b', pred_md_raw, re.IGNORECASE))
    figure_mentions = bool(re.search(r'(?i)\bfigure\b', pred_md_raw))

    # ── Candidate signals ───────────────────────────────────────────────────
    signals: list[str] = []

    if size_ratio > 1.3 and ins_words > del_words * 1.5:
        signals.append(
            f"Pred thêm ~{ins_words} từ so GT ({size_ratio:.1f}x) — "
            f"có thể GT thiếu nội dung hoặc model đọc thừa"
        )
    if size_ratio < 0.65 and del_words > ins_words * 1.5:
        signals.append(
            f"Pred thiếu ~{del_words} từ của GT — "
            "model bỏ sót nội dung, có thể do layout phức tạp"
        )

    gt_table_count = len(gt_page.get("tables") or [])
    pred_table_count = len(pred_page.get("tables") or [])
    if gt_table_count > 0 and not has_table and pred_table_count == 0:
        signals.append(
            f"GT có {gt_table_count} bảng nhưng pred không có cấu trúc bảng — "
            "model flatten bảng thành text"
        )
    if figure_mentions and ins_words > 10:
        signals.append("Pred chứa nội dung hình minh họa (FIGURE) có thể không nằm trong GT")

    cer  = metrics.get("cer", 0) or 0
    cf1  = metrics.get("char_f1", 0) or 0
    if cf1 > 0.80 and cer > 0.25 and 0.85 < size_ratio < 1.2:
        signals.append(
            f"Char F1 cao ({cf1:.0%}) nhưng CER cao ({cer:.0%}) — "
            "nội dung giống nhưng thứ tự đọc có thể khác (reading order)"
        )

    signals = signals[:5]  # cap at 5

    return {
        "gt_text":         gt_text,
        "pred_text":       pred_text,
        "inserted_chunks": inserted_chunks,
        "deleted_chunks":  deleted_chunks,
        "structural_flags": {
            "has_table":       has_table,
            "figure_mentions": figure_mentions,
        },
        "candidate_signals": signals,
        # Raw counts for agent tools
        "_meta": {
            "gt_text_len":    len(gt_norm),
            "pred_text_len":  len(pred_norm),
            "size_ratio":     round(size_ratio, 3),
            "ins_word_count": ins_words,
            "del_word_count": del_words,
            "gt_table_count": gt_table_count,
            "pred_table_count": pred_table_count,
        },
    }


def get_page_evidence(doc_id: str, page_num: int, model: Optional[str] = None) -> dict:
    """
    Load _evidence from saved eval file for a specific page.
    Used by chat agent tools.
    """
    uc_type, lang = _parse_doc_id(doc_id)

    # Resolve model if not provided
    if not model:
        candidates = list(RESULT_ROOT.glob(f"*/{uc_type}/{lang}/{doc_id}_eval.json"))
        if not candidates:
            return {"error": "document_not_found",
                    "details": f"No eval found for doc '{doc_id}' in any model directory"}
        model = candidates[0].relative_to(RESULT_ROOT).parts[0]

    eval_path = _result_path(model, uc_type, lang, doc_id)
    if not eval_path.exists():
        all_docs = [p.stem.replace("_eval", "")
                    for p in (RESULT_ROOT / model).rglob("*_eval.json")] if (RESULT_ROOT / model).exists() else []
        return {"error": "document_not_found",
                "details": f"No eval for model='{model}' doc='{doc_id}'",
                "available_docs": all_docs[:20]}

    try:
        data = json.loads(eval_path.read_text(encoding="utf-8"))
    except Exception:
        return {"error": "read_error", "details": "Failed to read eval file"}

    pages = (data.get("text") or {}).get("pages") or []
    page_nums = [p.get("page_num") for p in pages]

    if page_num not in page_nums:
        return {"error": "page_out_of_range",
                "details": f"page_num={page_num} not found",
                "valid_range": [min(page_nums), max(page_nums)] if page_nums else []}

    page = next(p for p in pages if p.get("page_num") == page_num)
    evidence = page.get("_evidence")

    if not evidence:
        return {"error": "evidence_unavailable",
                "details": "This page was scored before evidence collection was added.",
                "suggestion": "Re-score this document to collect evidence.",
                "metrics": {k: v for k, v in page.items()
                            if k != "_evidence" and isinstance(v, (int, float))}}

    return {
        "doc_id":    doc_id,
        "page_num":  page_num,
        "model":     model,
        "evidence":  evidence,
        "metrics":   {k: v for k, v in page.items()
                      if k not in ("_evidence", "_meta") and isinstance(v, (int, float))},
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/gt_docs", response_model=list[GTDocEntry])
def list_gt_docs():
    """All documents with a Ground Truth file, sorted."""
    entries: list[GTDocEntry] = []
    if not GT_ROOT.exists():
        return entries
    for path in GT_ROOT.rglob("*.json"):
        parts = path.relative_to(GT_ROOT).parts
        if len(parts) != 3:
            continue
        uc_type, lang = parts[0], parts[1]
        entries.append(GTDocEntry(doc_id=path.stem, uc_type=uc_type, lang=lang))
    return sorted(entries, key=lambda e: (e.uc_type, e.lang, e.doc_id))


@router.get("/known_models")
def list_known_models() -> list[str]:
    """Model names previously used via Upload & Score (for autocomplete)."""
    return _load_known_models()


@router.get("/doc_result")
def get_doc_result(doc_id: str, model: str):
    """Load saved eval result for a specific doc + model. Used by compare panel."""
    uc_type, lang = _parse_doc_id(doc_id)
    path = _result_path(model, uc_type, lang, doc_id)
    if not path.exists():
        raise HTTPException(404, detail=f"No result for model='{model}' doc='{doc_id}'")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(500, detail="Failed to read result file")


def _collect_model_stats() -> dict:
    """Scan benchmark_results/ and aggregate per-model statistics."""
    def _avg(lst):
        v = [x for x in lst if x is not None]
        return round(_mean(v), 4) if v else None

    model_stats: dict[str, dict] = {}
    if not RESULT_ROOT.exists():
        return model_stats

    for eval_file in RESULT_ROOT.rglob("*_eval.json"):
        parts = eval_file.relative_to(RESULT_ROOT).parts
        if len(parts) < 4:
            continue
        model_name = parts[0]
        doc_id_stem = eval_file.stem.replace("_eval", "")
        try:
            data = json.loads(eval_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        if model_name not in model_stats:
            model_stats[model_name] = {
                "model": model_name, "docs": 0,
                "char_f1": [], "cer": [], "teds": [], "cell_f1": [],
                "source": data.get("source", "pipeline"),
                "evaluated_docs": [],
            }
        s = model_stats[model_name]
        s["docs"] += 1
        s["evaluated_docs"].append(doc_id_stem)
        if data.get("source") == "upload":
            s["source"] = "upload"

        txt = (data.get("text") or {}).get("summary") or {}
        if txt.get("avg_char_f1") is not None:
            s["char_f1"].append(txt["avg_char_f1"])
        if txt.get("avg_cer") is not None:
            s["cer"].append(txt["avg_cer"])
        tbl = (data.get("table") or {}).get("summary") or {}
        if tbl.get("avg_teds") is not None:
            s["teds"].append(tbl["avg_teds"])
        elif txt.get("table_teds_doc") is not None:
            # upload results: TEDS stored directly in text.summary
            s["teds"].append(txt["table_teds_doc"])
        else:
            # upload results: compute average from pages
            pages = (data.get("text") or {}).get("pages") or []
            page_teds = [p["table_teds_doc"] for p in pages
                         if isinstance(p.get("table_teds_doc"), (int, float))]
            if page_teds:
                s["teds"].append(sum(page_teds) / len(page_teds))

        if tbl.get("avg_cell_exact_f1") is not None:
            s["cell_f1"].append(tbl["avg_cell_exact_f1"])
        elif txt.get("table_cell_exact_f1_mean") is not None:
            s["cell_f1"].append(txt["table_cell_exact_f1_mean"])
        else:
            pages = (data.get("text") or {}).get("pages") or []
            page_cell = [p["table_cell_exact_f1_mean"] for p in pages
                         if isinstance(p.get("table_cell_exact_f1_mean"), (int, float))]
            if page_cell:
                s["cell_f1"].append(sum(page_cell) / len(page_cell))

    # compute averages in-place
    for s in model_stats.values():
        s["avg_char_f1"] = _avg(s["char_f1"])
        s["avg_cer"]     = _avg(s["cer"])
        s["avg_teds"]    = _avg(s["teds"])
        s["avg_cell_f1"] = _avg(s["cell_f1"])
    return model_stats


@router.get("/leaderboard")
def get_leaderboard(min_docs: int = 24):
    """
    Leaderboard: models with >= min_docs evaluated.
    Ranking criteria (in order):
      1. Char F1 — higher is better (primary text quality)
      2. TEDS    — higher is better (table structure quality)
    Only models with results on ALL min_docs documents appear here.
    Models still in progress (< min_docs) are NOT shown — use /progress.
    """
    stats = _collect_model_stats()
    rows = []
    for s in stats.values():
        if s["docs"] < min_docs:
            continue
        rows.append({
            "model": s["model"], "docs": s["docs"],
            "avg_char_f1": s["avg_char_f1"],
            "avg_cer": s["avg_cer"],
            "avg_teds": s["avg_teds"],
            "avg_cell_f1": s["avg_cell_f1"],
            "source": s["source"],
        })
    rows.sort(key=lambda r: (r["avg_char_f1"] or 0, r["avg_teds"] or 0), reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        r["is_best"] = (i == 0)
    return rows


@router.get("/progress")
def get_model_progress(total_docs: int = 24):
    """
    Progress for ALL models (including incomplete ones).
    Shows how many of total_docs have been evaluated so far.
    Used in Leaderboard UI to show 'X/24 — Y more needed' for in-progress models.
    Includes list of which doc_ids are still missing.
    """
    # Build the full list of 24 doc_ids from GT
    all_docs: set[str] = set()
    if GT_ROOT.exists():
        for path in GT_ROOT.rglob("*.json"):
            parts = path.relative_to(GT_ROOT).parts
            if len(parts) == 3:
                all_docs.add(path.stem)

    stats = _collect_model_stats()
    rows = []
    for s in stats.values():
        evaluated = set(s.get("evaluated_docs", []))
        missing = sorted(all_docs - evaluated)
        rows.append({
            "model": s["model"],
            "docs": s["docs"],
            "total_docs": total_docs,
            "missing": len(missing),
            "missing_docs": missing,
            "complete": s["docs"] >= total_docs,
            "avg_char_f1": s["avg_char_f1"],
            "avg_cer": s["avg_cer"],
            "avg_teds": s["avg_teds"],
            "avg_cell_f1": s["avg_cell_f1"],
            "source": s["source"],
        })
    rows.sort(key=lambda r: r["docs"], reverse=True)
    return rows


@router.get("/doc_comparison")
def get_doc_comparison(doc_id: str) -> dict:
    """
    Return all models' summary scores for one document side-by-side.
    Used by chat upload panel to show instant comparison without LLM call.
    """
    uc_type, lang = _parse_doc_id(doc_id)
    pattern = f"*/{uc_type}/{lang}/{doc_id}_eval.json"
    
    models_data = {}
    for eval_path in RESULT_ROOT.glob(pattern):
        model_name = eval_path.relative_to(RESULT_ROOT).parts[0]
        try:
            data = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        
        txt = (data.get("text") or {}).get("summary") or {}
        tbl = (data.get("table") or {}).get("summary") or {}
        pages = (data.get("text") or {}).get("pages") or []
        
        # Resolve metrics with fallback to page averages
        def _resolve(key, alt=None):
            v = txt.get(key) or txt.get(f"avg_{key}")
            if v is not None: return v
            if alt: v = tbl.get(alt)
            if v is not None: return v
            vals = [p[key] for p in pages if isinstance(p.get(key), (int, float))]
            return round(sum(vals)/len(vals), 4) if vals else None
        
        models_data[model_name] = {
            "cer":     _resolve("cer",   "avg_cer"),
            "wer":     _resolve("wer",   "avg_wer"),
            "char_f1": _resolve("char_f1", "avg_char_f1"),
            "word_f1": _resolve("word_f1", "avg_word_f1"),
            "edit_sim": _resolve("normalized_edit_similarity"),
            "teds":    _resolve("table_teds_doc", "avg_teds"),
            "cell_f1": _resolve("table_cell_exact_f1_mean", "avg_cell_exact_f1"),
            "n_pages": txt.get("n_pages") or len(pages),
            "source":  data.get("source", "pipeline"),
        }
    
    return {"doc_id": doc_id, "uc_type": uc_type, "lang": lang, "models": models_data}
async def score_upload(
    files: list[UploadFile] = File(...),
    model_name: str = Form(...),
    doc_id: str = Form(...),
):
    """
    Score uploaded .md prediction files against the server-side Ground Truth.
    Saves result to benchmark_results/ for dashboard and leaderboard.
    """
    if not model_name or not model_name.strip():
        raise HTTPException(422, detail="model_name must be a non-empty string")
    model_name = model_name.strip()[:100]
    if not doc_id or not doc_id.strip():
        raise HTTPException(422, detail="doc_id is required")
    doc_id = doc_id.strip()

    md_files = [f for f in files if (f.filename or "").lower().endswith(".md")]
    if not md_files:
        raise HTTPException(422, detail="No .md files found in the uploaded folder.")
    if len(md_files) > 500:
        raise HTTPException(422, detail=f"File limit exceeded: max 500 .md files (got {len(md_files)})")

    uc_type, lang = _parse_doc_id(doc_id)

    gt_path = GT_ROOT / uc_type / lang / f"{doc_id}.json"
    if not gt_path.exists():
        raise HTTPException(404, detail=f"Ground Truth not found for '{doc_id}' (expected: ground_truth/{uc_type}/{lang}/{doc_id}.json)")
    try:
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(422, detail=f"Ground Truth file for '{doc_id}' is corrupt or invalid JSON")
    gt_pages_raw = gt_data.get("pages")
    if not gt_pages_raw:
        raise HTTPException(422, detail=f"Ground Truth for '{doc_id}' has no pages")

    parsed_pages: list[dict] = []
    for f in md_files:
        filename = Path(f.filename or "").name
        m = _PAGE_FILENAME_RE.match(filename)
        if not m:
            continue
        page_index = int(m.group(2))
        content = (await f.read()).decode("utf-8", errors="replace")
        parsed_pages.append({"page_index": page_index, "raw_content": content})

    if not parsed_pages:
        raise HTTPException(422, detail=f"No valid page files. Name them like: {doc_id}_0.md, {doc_id}_1.md, ...")

    base_index = min(p["page_index"] for p in parsed_pages)
    for p in parsed_pages:
        p["page_num"] = p["page_index"] - base_index + 1

    parsed_pages.sort(key=lambda p: p["page_index"])
    seen: set[int] = set()
    deduped = [p for p in parsed_pages if p["page_num"] not in seen and not seen.add(p["page_num"])]  # type: ignore

    for p in deduped:
        p["filtered_text"] = _filter_content(p["raw_content"])
        p["tables"] = _extract_html_tables(p["raw_content"])

    pred_by_num: dict[int, dict] = {p["page_num"]: p for p in deduped}

    if not _METRICS_AVAILABLE:
        return {"model": model_name, "doc_id": doc_id, "uc_type": uc_type, "lang": lang,
                "error": "Scoring dependencies not available",
                "results": {"text": {"summary": {"n_pages": len(gt_pages_raw), "n_matched_pages": 0}, "pages": []}}}

    page_results: list[dict] = []
    n_matched = 0
    for gt_page in gt_pages_raw:
        pnum = gt_page.get("page_num", 1)
        pred_entry = pred_by_num.get(pnum)
        if pred_entry is not None:
            n_matched += 1
            pred_page = {"full_text": pred_entry["filtered_text"], "tables": pred_entry.get("tables", [])}
        else:
            pred_page = {"full_text": "", "tables": []}
        try:
            metrics = compute_all_metrics(gt_page, pred_page)
        except Exception as exc:
            metrics = {"error": str(exc)[:200]}
        metrics["page_num"] = pnum

        # Collect evidence for chat agent retrieval
        try:
            # Pass raw_content for structural flag detection
            pred_page_with_raw = {**pred_page, "_raw_content": (pred_entry or {}).get("raw_content", "")}
            metrics["_evidence"] = collect_evidence(gt_page, pred_page_with_raw, metrics)
        except Exception as exc:
            _logger.error("collect_evidence failed for page %d: %s", pnum, exc)
        # (no _evidence key if exception — page still saved)

        page_results.append(metrics)

    summary: dict = {"n_pages": len(gt_pages_raw), "n_matched_pages": n_matched}
    for metric in _AVERAGED_METRICS:
        summary[metric] = _safe_mean([r.get(metric) for r in page_results])

    # Save with avg_ keys so dashboard GET /api/ocr/summary reads it correctly
    result_for_save = {
        "text": {
            "summary": {
                "avg_cer":    summary.get("cer"),
                "avg_wer":    summary.get("wer"),
                "avg_char_f1": summary.get("char_f1"),
                "avg_word_f1": summary.get("word_f1"),
                "avg_normalized_edit_similarity": summary.get("normalized_edit_similarity"),
                "n_pages":    summary["n_pages"],
                "n_matched_pages": summary["n_matched_pages"],
            },
            "pages": page_results,
        }
    }
    _save_result(model_name, uc_type, lang, doc_id, result_for_save)
    _save_model_name(model_name)

    return {
        "model": model_name,
        "doc_id": doc_id,
        "uc_type": uc_type,
        "lang": lang,
        "results": {"text": {"summary": summary, "pages": page_results}},
    }


@router.post("/score")
async def score_upload(
    files: list[UploadFile] = File(...),
    model_name: str = Form(...),
    doc_id: str = Form(...),
):
    """
    Score uploaded .md prediction files against the server-side Ground Truth.
    Saves result to benchmark_results/ for dashboard and leaderboard.
    """
    if not model_name or not model_name.strip():
        raise HTTPException(422, detail="model_name must be a non-empty string")
    model_name = model_name.strip()[:100]
    if not doc_id or not doc_id.strip():
        raise HTTPException(422, detail="doc_id is required")
    doc_id = doc_id.strip()

    md_files = [f for f in files if (f.filename or "").lower().endswith(".md")]
    if not md_files:
        raise HTTPException(422, detail="No .md files found in the uploaded folder.")
    if len(md_files) > 500:
        raise HTTPException(422, detail=f"File limit exceeded: max 500 .md files (got {len(md_files)})")

    uc_type, lang = _parse_doc_id(doc_id)

    gt_path = GT_ROOT / uc_type / lang / f"{doc_id}.json"
    if not gt_path.exists():
        raise HTTPException(404, detail=f"Ground Truth not found for '{doc_id}' (expected: ground_truth/{uc_type}/{lang}/{doc_id}.json)")
    try:
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(422, detail=f"Ground Truth file for '{doc_id}' is corrupt or invalid JSON")
    gt_pages_raw = gt_data.get("pages")
    if not gt_pages_raw:
        raise HTTPException(422, detail=f"Ground Truth for '{doc_id}' has no pages")

    parsed_pages: list[dict] = []
    for f in md_files:
        filename = Path(f.filename or "").name
        m = _PAGE_FILENAME_RE.match(filename)
        if not m:
            continue
        page_index = int(m.group(2))
        content = (await f.read()).decode("utf-8", errors="replace")
        parsed_pages.append({"page_index": page_index, "raw_content": content})

    if not parsed_pages:
        raise HTTPException(422, detail=f"No valid page files. Name them like: {doc_id}_0.md, {doc_id}_1.md, ...")

    base_index = min(p["page_index"] for p in parsed_pages)
    for p in parsed_pages:
        p["page_num"] = p["page_index"] - base_index + 1

    parsed_pages.sort(key=lambda p: p["page_index"])
    seen: set[int] = set()
    deduped = [p for p in parsed_pages if p["page_num"] not in seen and not seen.add(p["page_num"])]  # type: ignore

    for p in deduped:
        p["filtered_text"] = _filter_content(p["raw_content"])
        p["tables"] = _extract_html_tables(p["raw_content"])

    pred_by_num: dict[int, dict] = {p["page_num"]: p for p in deduped}

    if not _METRICS_AVAILABLE:
        return {"model": model_name, "doc_id": doc_id, "uc_type": uc_type, "lang": lang,
                "error": "Scoring dependencies not available",
                "results": {"text": {"summary": {"n_pages": len(gt_pages_raw), "n_matched_pages": 0}, "pages": []}}}

    page_results: list[dict] = []
    n_matched = 0
    for gt_page in gt_pages_raw:
        pnum = gt_page.get("page_num", 1)
        pred_entry = pred_by_num.get(pnum)
        if pred_entry is not None:
            n_matched += 1
            pred_page = {"full_text": pred_entry["filtered_text"], "tables": pred_entry.get("tables", [])}
        else:
            pred_page = {"full_text": "", "tables": []}
        try:
            metrics = compute_all_metrics(gt_page, pred_page)
        except Exception as exc:
            metrics = {"error": str(exc)[:200]}
        metrics["page_num"] = pnum

        # Collect evidence for chat agent retrieval
        try:
            pred_page_with_raw = {**pred_page, "_raw_content": (pred_entry or {}).get("raw_content", "")}
            metrics["_evidence"] = collect_evidence(gt_page, pred_page_with_raw, metrics)
        except Exception as exc:
            _logger.error("collect_evidence failed for page %d: %s", pnum, exc)

        page_results.append(metrics)

    summary: dict = {"n_pages": len(gt_pages_raw), "n_matched_pages": n_matched}
    for metric in _AVERAGED_METRICS:
        summary[metric] = _safe_mean([r.get(metric) for r in page_results])

    # Save with avg_ keys so dashboard GET /api/ocr/summary reads it correctly
    result_for_save = {
        "text": {
            "summary": {
                "avg_cer":    summary.get("cer"),
                "avg_wer":    summary.get("wer"),
                "avg_char_f1": summary.get("char_f1"),
                "avg_word_f1": summary.get("word_f1"),
                "avg_normalized_edit_similarity": summary.get("normalized_edit_similarity"),
                "n_pages":    summary["n_pages"],
                "n_matched_pages": summary["n_matched_pages"],
            },
            "pages": page_results,
        }
    }
    _save_result(model_name, uc_type, lang, doc_id, result_for_save)
    _save_model_name(model_name)

    return {
        "model": model_name,
        "doc_id": doc_id,
        "uc_type": uc_type,
        "lang": lang,
        "results": {"text": {"summary": summary, "pages": page_results}},
    }
