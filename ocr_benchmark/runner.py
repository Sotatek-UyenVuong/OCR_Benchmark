"""
Benchmark runner.

Reads GT JSON + predictions, runs eval per page, aggregates by UC, writes CSV.

Usage:
    python -m ocr_benchmark.runner \
        --gt_dir ground_truth/ \
        --pred_dir predictions/gpt4o/ \
        --model gpt4o \
        --output_dir benchmark_results/
"""

from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path

from .eval.scan import eval_scan
from .eval.table import eval_table
from .eval.text_layer import eval_text_layer


# ---------------------------------------------------------------------------
# UC → type mapping
# ---------------------------------------------------------------------------

UC_TYPE = {
    "UC01": "scan", "UC02": "scan", "UC03": "scan",
    "UC04": "table", "UC05": "table", "UC06": "table",
    "UC07": "text_layer", "UC08": "text_layer", "UC09": "text_layer",
}

UC_NAME = {
    "UC01": "scan_vi", "UC02": "scan_en", "UC03": "scan_ja",
    "UC04": "table_vi", "UC05": "table_en", "UC06": "table_ja",
    "UC07": "text_layer_vi", "UC08": "text_layer_en", "UC09": "text_layer_ja",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return None


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Per-UC aggregation
# ---------------------------------------------------------------------------

def _aggregate_scan(page_results: list[dict]) -> dict:
    return {
        "avg_cer": round(_safe_mean([r["cer"] for r in page_results]), 6),
        "avg_wer": round(_safe_mean([r["wer"] for r in page_results]), 6),
        "avg_nwer": round(_safe_mean([r["nwer"] for r in page_results]), 6),
        "avg_char_precision": round(_safe_mean([r["char_precision"] for r in page_results]), 6),
        "avg_char_recall": round(_safe_mean([r["char_recall"] for r in page_results]), 6),
        "n_pages": len(page_results),
    }


def _aggregate_table(page_results: list[dict]) -> dict:
    return {
        "avg_teds": round(_safe_mean([r["avg_teds"] for r in page_results]), 6),
        "n_pages": len(page_results),
    }


def _aggregate_text_layer(page_results: list[dict]) -> dict:
    return {
        "avg_cer": round(_safe_mean([r["cer"] for r in page_results]), 6),
        "avg_mean_iou": round(_safe_mean([r["mean_iou"] for r in page_results]), 6),
        "n_pages": len(page_results),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmark(
    gt_dir: str | Path,
    pred_dir: str | Path,
    model: str,
    output_dir: str | Path,
    include_alignment: bool = False,
) -> dict:
    """
    Run full benchmark for one model.

    Directory structure expected:
        gt_dir/
            scan/vi/<doc_id>.json
            table/vi/<doc_id>.json
            text_layer/vi/<doc_id>.json
            ...

        pred_dir/
            scan/vi/<doc_id>.json     ({"pages": [{"page_num": int, "full_text": str}]})
            table/vi/<doc_id>.json    ({"pages": [{"page_num": int, "tables": [...]}]})
            text_layer/vi/<doc_id>.json

    Returns summary dict and writes CSVs to output_dir.
    """
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_page_rows: list[dict] = []   # one row per page
    uc_summaries: list[dict] = []    # one row per UC

    for uc_id, uc_type in UC_TYPE.items():
        uc_name = UC_NAME[uc_id]
        # Derive subfolder from uc_name: "scan_vi" → "scan/vi"
        subfolder = uc_name.replace("_", "/", 1)

        gt_uc_dir = gt_dir / subfolder
        pred_uc_dir = pred_dir / subfolder

        if not gt_uc_dir.exists():
            print(f"[SKIP] GT dir not found: {gt_uc_dir}")
            continue

        page_results: list[dict] = []

        for gt_path in sorted(gt_uc_dir.glob("*.json")):
            doc_id = gt_path.stem
            pred_path = pred_uc_dir / gt_path.name

            gt_doc = _load_json(gt_path)
            pred_doc = _load_json(pred_path)

            if gt_doc is None or pred_doc is None:
                continue

            for gt_page in gt_doc.get("pages", []):
                page_num = gt_page.get("page_num", 1)
                pred_pages = {p["page_num"]: p for p in pred_doc.get("pages", [])}
                pred_page = pred_pages.get(page_num, {})

                try:
                    if uc_type == "scan":
                        result = eval_scan(
                            gt_page,
                            pred_page.get("full_text", ""),
                            doc_id=doc_id,
                            include_alignment=include_alignment,
                        )
                    elif uc_type == "table":
                        result = eval_table(
                            gt_page,
                            pred_page.get("tables", []),
                            doc_id=doc_id,
                        )
                    elif uc_type == "text_layer":
                        result = eval_text_layer(
                            gt_page,
                            pred_page.get("blocks", []),
                            pred_page.get("full_text", ""),
                            doc_id=doc_id,
                            include_alignment=include_alignment,
                        )
                    else:
                        continue

                    result["model"] = model
                    result["uc"] = uc_id
                    result["uc_name"] = uc_name
                    page_results.append(result)
                    all_page_rows.append(result)

                except Exception as e:
                    print(f"[ERROR] {doc_id} page {page_num}: {e}")

        if not page_results:
            continue

        # Aggregate
        if uc_type == "scan":
            agg = _aggregate_scan(page_results)
        elif uc_type == "table":
            agg = _aggregate_table(page_results)
        else:
            agg = _aggregate_text_layer(page_results)

        uc_summaries.append({"model": model, "uc": uc_id, "uc_name": uc_name, **agg})

    # Write page-level CSV
    _write_csv(all_page_rows, output_dir / f"{model}_page_results.csv")

    # Write UC-level summary CSV
    _write_csv(uc_summaries, output_dir / f"{model}_uc_summary.csv")

    # Leaderboard row
    overall = _build_overall_row(model, uc_summaries)
    print(f"\n=== {model} Overall ===")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    return {"page_results": all_page_rows, "uc_summaries": uc_summaries, "overall": overall}


def _flatten_for_csv(row: dict) -> dict:
    """Flatten nested dicts one level deep for CSV writing."""
    flat = {}
    for k, v in row.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat[f"{k}.{sub_k}"] = sub_v
        elif isinstance(v, list):
            flat[k] = json.dumps(v, ensure_ascii=False)
        else:
            flat[k] = v
    return flat


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    flat_rows = [_flatten_for_csv(r) for r in rows]
    fieldnames = list(dict.fromkeys(k for r in flat_rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"[OK] Written: {path}")


def _build_overall_row(model: str, uc_summaries: list[dict]) -> dict:
    """Compute macro-average across all UCs."""
    row: dict = {"model": model}

    for metric in ["avg_cer", "avg_wer", "avg_nwer", "avg_teds", "avg_mean_iou"]:
        vals = [s[metric] for s in uc_summaries if metric in s]
        if vals:
            row[metric] = round(_safe_mean(vals), 6)

    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OCR Benchmark Runner")
    parser.add_argument("--gt_dir", required=True, help="Path to ground_truth/ directory")
    parser.add_argument("--pred_dir", required=True, help="Path to predictions/<model>/ directory")
    parser.add_argument("--model", required=True, help="Model name (used in output filenames)")
    parser.add_argument("--output_dir", default="benchmark_results", help="Where to write CSVs")
    parser.add_argument("--alignment", action="store_true", help="Include char alignment in output")
    args = parser.parse_args()

    run_benchmark(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        model=args.model,
        output_dir=args.output_dir,
        include_alignment=args.alignment,
    )


if __name__ == "__main__":
    main()
