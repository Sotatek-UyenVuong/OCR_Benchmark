#!/usr/bin/env python3
"""
batch_reeval.py
---------------
Re-evaluate all benchmark results using the updated scoring formula
(text metrics now include table cell content — commit 15422e2).

Supports 3 pred sources (auto-detected per model):
  A) Upload models with saved .md pages
     benchmark_results/{model}/{uc}/{lang}/{doc_id}/{doc_id}_p{N}.md
     → Chadra 2 (partial), any future uploaded model

  B) Pipeline models with raw/ prediction JSONs
     raw/{uc}/{lang}/{model}_output/{doc_id}_text_prediction.json
     raw/{uc}/{lang}/{model}_output/{doc_id}_table_prediction.json
     → marker, mistral

  C) Upload models WITHOUT saved .md pages (Paddle OCR VL 1.6)
     pred_text stored in _evidence.pred_text (truncated, 400 chars)
     → Cannot re-eval accurately; SKIP with warning.

Usage:
    python3 scripts/batch_reeval.py              # all models
    python3 scripts/batch_reeval.py --model marker
    python3 scripts/batch_reeval.py --model "Chadra 2" --doc scan_en_001
    python3 scripts/batch_reeval.py --dry-run    # show what would run
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULT_ROOT = PROJECT_ROOT / "benchmark_results"
GT_ROOT     = PROJECT_ROOT / "ground_truth"
RAW_ROOT    = PROJECT_ROOT / "raw"

# ── Import scoring helpers ────────────────────────────────────────────────────
from ocr_benchmark.normalize import normalize_ocr_text
from ocr_benchmark.metrics.uet_metrics import compute_all_metrics

# Reuse helpers from upload_scorer without importing FastAPI
_TABLE_RE     = re.compile(r'<table\b[^>]*>.*?</table>', re.IGNORECASE | re.DOTALL)
_IMAGE_RE     = re.compile(r'!\[[^\]]*\]\([^)]*\)(?:\s*\n[ \t]*\n[^\n<#!*][^\n]*(?:\n[^\n<#!*][^\n]*)*)?', re.DOTALL)
_FENCED_RE    = re.compile(r'```[\w-]*\n[\s\S]*?```', re.DOTALL)
_FIGURE_RE    = re.compile(r'<figure\b[^>]*>.*?</figure>', re.IGNORECASE | re.DOTALL)


def _flatten_html_table_text(html: str) -> str:
    cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', html, re.IGNORECASE | re.DOTALL)
    cell_texts = [re.sub(r'<[^>]+>', ' ', c).strip() for c in cells]
    return ' '.join(t for t in cell_texts if t)


def _filter_content(text: str) -> str:
    """Strip non-text elements, replace HTML tables with flat cell text."""
    text = _FIGURE_RE.sub("", text)
    text = _IMAGE_RE.sub("", text)
    text = _FENCED_RE.sub("", text)
    text = _TABLE_RE.sub(lambda m: "\n" + _flatten_html_table_text(m.group(0)) + "\n", text)
    return normalize_ocr_text(text)


def _extract_html_tables(raw_md: str) -> list[dict]:
    return [{"table_id": i, "html": m.group(0)}
            for i, m in enumerate(_TABLE_RE.finditer(raw_md), start=1)]


def _build_full_text_for_scoring(gt_page: dict) -> str:
    """GT full_text + flatten cell text from all tables."""
    parts = [gt_page.get("full_text") or ""]
    for tbl in (gt_page.get("tables") or []):
        cells = tbl.get("cells") or []
        if cells:
            cell_text = " ".join(c.get("text", "") for c in cells if c.get("text", "").strip())
        else:
            cell_text = _flatten_html_table_text(tbl.get("html") or "")
        if cell_text:
            parts.append(cell_text)
    return "\n".join(p for p in parts if p)


def _safe_mean(vals: list) -> Optional[float]:
    v = [x for x in vals if x is not None]
    return round(sum(v) / len(v), 6) if v else None


_AVERAGED_METRICS = [
    "cer", "wer", "normalized_edit_similarity", "char_f1", "word_f1",
    "table_teds_doc", "table_teds_matched_mean",
    "table_cell_exact_f1_mean", "table_cell_text_similarity_mean",
    "table_row_count_similarity_mean", "table_col_count_similarity_mean",
    "table_count_f1",
]


# ── Pred-source resolvers ─────────────────────────────────────────────────────

def _load_pred_from_md_pages(
    model: str, uc_type: str, lang: str, doc_id: str, gt_pages: list[dict]
) -> Optional[dict[int, dict]]:
    """Source A: per-page .md files in benchmark_results/{model}/{uc}/{lang}/{doc_id}/"""
    pred_dir = RESULT_ROOT / model / uc_type / lang / doc_id
    if not pred_dir.exists():
        return None
    md_files = list(pred_dir.glob(f"{doc_id}_p*.md"))
    if not md_files:
        return None

    result: dict[int, dict] = {}
    for f in md_files:
        m = re.search(r'_p(\d+)\.md$', f.name)
        if not m:
            continue
        pnum = int(m.group(1))
        raw = f.read_text(encoding="utf-8")
        result[pnum] = {
            "full_text": _filter_content(raw),
            "tables":    _extract_html_tables(raw),
        }
    return result if result else None


def _load_pred_from_raw_pipeline(
    model: str, uc_type: str, lang: str, doc_id: str
) -> Optional[dict[int, dict]]:
    """Source B: raw/{uc}/{lang}/{model}_output/{doc_id}_*_prediction.json"""
    model_slug = model.lower().replace(" ", "_")
    raw_dir = RAW_ROOT / uc_type / lang / f"{model_slug}_output"
    if not raw_dir.exists():
        return None

    text_pred_file  = raw_dir / f"{doc_id}_text_prediction.json"
    table_pred_file = raw_dir / f"{doc_id}_table_prediction.json"

    if not text_pred_file.exists():
        return None

    try:
        text_data = json.loads(text_pred_file.read_text(encoding="utf-8"))
    except Exception:
        return None

    table_by_page: dict[int, list[dict]] = {}
    if table_pred_file.exists():
        try:
            tbl_data = json.loads(table_pred_file.read_text(encoding="utf-8"))
            for p in (tbl_data.get("pages") or []):
                table_by_page[p["page_num"]] = p.get("tables", [])
        except Exception:
            pass

    result: dict[int, dict] = {}
    for p in (text_data.get("pages") or []):
        pnum = p["page_num"]
        raw_text = p.get("full_text") or ""
        tables_raw = table_by_page.get(pnum, [])
        # Convert tables to expected format with html key
        tables = [{"table_id": t.get("table_id", i+1), "html": t.get("html", "")}
                  for i, t in enumerate(tables_raw) if t.get("html")]
        result[pnum] = {
            "full_text": _filter_content(raw_text),
            "tables":    tables,
        }
    return result if result else None


# ── Core re-eval function ─────────────────────────────────────────────────────

def reeval_document(
    model: str, uc_type: str, lang: str, doc_id: str, dry_run: bool = False
) -> dict:
    """Re-evaluate one document and overwrite its _eval.json. Returns status dict."""
    eval_path = RESULT_ROOT / model / uc_type / lang / f"{doc_id}_eval.json"
    if not eval_path.exists():
        return {"status": "skip", "reason": "eval_file_not_found"}

    # Load GT
    gt_path = GT_ROOT / uc_type / lang / f"{doc_id}.json"
    if not gt_path.exists():
        return {"status": "skip", "reason": "gt_not_found"}
    try:
        gt_data  = json.loads(gt_path.read_text(encoding="utf-8"))
        gt_pages = gt_data.get("pages") or []
    except Exception as e:
        return {"status": "error", "reason": f"gt_parse_error: {e}"}

    # Load existing eval (to preserve metadata and _evidence)
    try:
        existing = json.loads(eval_path.read_text(encoding="utf-8"))
    except Exception:
        existing = {}

    # Resolve pred pages — try source A then B
    pred_by_num = (_load_pred_from_md_pages(model, uc_type, lang, doc_id, gt_pages) or
                   _load_pred_from_raw_pipeline(model, uc_type, lang, doc_id))

    if pred_by_num is None:
        return {
            "status": "skip",
            "reason": "no_pred_source (no saved .md pages, no raw/ prediction JSON)",
        }

    if dry_run:
        return {"status": "dry_run", "pred_pages": len(pred_by_num), "gt_pages": len(gt_pages)}

    # Re-score
    page_results: list[dict] = []
    n_matched = 0
    for gt_page in gt_pages:
        pnum       = gt_page.get("page_num", 1)
        pred_entry = pred_by_num.get(pnum)
        if pred_entry:
            n_matched += 1
            pred_page = pred_entry
        else:
            pred_page = {"full_text": "", "tables": []}

        gt_page_for_scoring = {**gt_page, "full_text": _build_full_text_for_scoring(gt_page)}
        try:
            metrics = compute_all_metrics(gt_page_for_scoring, pred_page)
        except Exception as exc:
            metrics = {"error": str(exc)[:200]}
        metrics["page_num"] = pnum

        # Preserve existing _evidence (don't recompute to keep it fast)
        existing_pages = (existing.get("text") or {}).get("pages") or []
        old_page = next((p for p in existing_pages if p.get("page_num") == pnum), {})
        if old_page.get("_evidence"):
            metrics["_evidence"] = old_page["_evidence"]

        page_results.append(metrics)

    summary: dict = {"n_pages": len(gt_pages), "n_matched_pages": n_matched}
    for metric in _AVERAGED_METRICS:
        summary[metric] = _safe_mean([r.get(metric) for r in page_results])

    # Build new eval payload
    source = existing.get("source", "pipeline")
    new_eval = {
        "doc_id":     doc_id,
        "model":      model,
        "source":     source,
        "scored_at":  existing.get("scored_at", ""),
        "reeval_at":  datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "text": {
            "summary": {
                "avg_cer":          summary.get("cer"),
                "avg_wer":          summary.get("wer"),
                "avg_char_f1":      summary.get("char_f1"),
                "avg_word_f1":      summary.get("word_f1"),
                "avg_normalized_edit_similarity": summary.get("normalized_edit_similarity"),
                "n_pages":          summary["n_pages"],
                "n_matched_pages":  n_matched,
            },
            "pages": page_results,
        },
    }
    # Preserve table section if original had it (pipeline evals)
    if existing.get("table"):
        new_eval["table"] = existing["table"]

    eval_path.write_text(json.dumps(new_eval, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "ok",
        "pages": len(page_results),
        "matched": n_matched,
        "avg_char_f1": summary.get("char_f1"),
        "avg_cer": summary.get("cer"),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch re-evaluate all benchmark results")
    parser.add_argument("--model",   help="Re-eval only this model (substring match)")
    parser.add_argument("--doc",     help="Re-eval only this doc_id")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without writing")
    args = parser.parse_args()

    # Discover all eval files
    eval_files = sorted(RESULT_ROOT.rglob("*_eval.json"))
    print(f"Found {len(eval_files)} eval files\n")

    results: list[dict] = []
    ok = skip = error = 0

    for eval_file in eval_files:
        parts = eval_file.relative_to(RESULT_ROOT).parts
        if len(parts) < 4:
            continue
        model, uc_type, lang = parts[0], parts[1], parts[2]
        doc_id = eval_file.stem.replace("_eval", "")

        # Filter flags
        if args.model and args.model.lower() not in model.lower():
            continue
        if args.doc and args.doc != doc_id:
            continue

        result = reeval_document(model, uc_type, lang, doc_id, dry_run=args.dry_run)
        result.update({"model": model, "uc_type": uc_type, "lang": lang, "doc_id": doc_id})
        results.append(result)

        status = result["status"]
        if status == "ok":
            ok += 1
            cf1 = result.get("avg_char_f1")
            cer = result.get("avg_cer")
            print(f"  ✅ {model}/{uc_type}/{lang}/{doc_id} — CharF1={cf1:.1%} CER={cer:.1%}")
        elif status == "dry_run":
            skip += 1
            print(f"  🔍 {model}/{uc_type}/{lang}/{doc_id} — {result['pred_pages']} pred / {result['gt_pages']} gt pages")
        elif status == "skip":
            skip += 1
            print(f"  ⏭  {model}/{uc_type}/{lang}/{doc_id} — SKIP: {result.get('reason')}")
        else:
            error += 1
            print(f"  ❌ {model}/{uc_type}/{lang}/{doc_id} — ERROR: {result.get('reason')}")

    print(f"\n{'DRY RUN ' if args.dry_run else ''}Summary: ✅ {ok}  ⏭ {skip}  ❌ {error}")


if __name__ == "__main__":
    main()
