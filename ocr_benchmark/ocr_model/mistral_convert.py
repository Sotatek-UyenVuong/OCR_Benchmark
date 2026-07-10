#!/usr/bin/env python3
"""
mistral_convert.py
------------------
Convert a local PDF to Markdown using Mistral OCR API.
Tables are extracted as HTML (TABLE_FORMAT=html).

Response format (Mistral OCR v4):
  {
    "pages": [{"index": 0, "markdown": "...", "images": [...]}],
    ...
  }

Unlike Marker, Mistral returns flat Markdown — no block_type.
Tables appear inline as <table>...</table> in the markdown.
We extract them using BeautifulSoup.

Prediction schema output (same as marker_convert):
  split="text":  {"pages": [{"page_num": int, "full_text": str}]}
  split="table": {"pages": [{"page_num": int, "tables": [...]}]}
  split="split": {"text": ..., "table": ...}
"""

from __future__ import annotations

import os
import re
import json
import base64
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_OCR_URL: str = "https://api.mistral.ai/v1/ocr"
MISTRAL_OCR_MODEL: str = "mistral-ocr-latest"


# ── Helpers ───────────────────────────────────────────────────

def _encode_pdf(pdf_path: Path) -> str:
    """Base64-encode a PDF file as data URL."""
    with open(pdf_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:application/pdf;base64,{encoded}"


def _extract_markdown(result: dict) -> list[dict]:
    """
    Extract per-page markdown from Mistral response.
    Returns: [{"page_num": int, "markdown": str}]
    """
    pages = result.get("pages") or []
    out = []
    for page in sorted(pages, key=lambda p: int(p.get("index", 0))):
        out.append({
            "page_num": int(page.get("index", 0)) + 1,  # 0-indexed → 1-indexed
            "markdown": str(page.get("markdown") or ""),
        })
    return out


def _strip_html_tables(md: str) -> str:
    """Remove HTML table blocks from markdown, return plain text only."""
    # Remove <table>...</table> blocks (possibly multiline)
    clean = re.sub(r"<table[\s\S]*?</table>", " ", md, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _extract_html_table_strings(md: str) -> list[str]:
    """Extract all <table>...</table> strings from markdown."""
    tables = re.findall(r"<table[\s\S]*?</table>", md, flags=re.IGNORECASE)
    return tables


def _html_table_to_cells(html: str) -> list[dict]:
    """Parse HTML table string into cells list for benchmark schema."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []
    cells = []
    soup = BeautifulSoup(html, "html.parser")
    for ri, row in enumerate(soup.find_all("tr")):
        ci = 0
        for cell in row.find_all(["td", "th"]):
            rs = int(cell.get("rowspan", 1))
            cs = int(cell.get("colspan", 1))
            cells.append({
                "row": ri, "col": ci,
                "rowspan": rs, "colspan": cs,
                "text": cell.get_text(separator=" ", strip=True),
                "is_header": cell.name == "th",
            })
            ci += cs
    return cells


# ── Core convert ──────────────────────────────────────────────

def convert(
    source: str,
    output_dir: Path,
    langs: str = "",
    mode: str = "balanced",
    timeout: int = 120,
    save_md: bool = True,
    save_full: bool = True,
    uc_type: str | None = None,
    doc_id: str = "",
    save_prediction: bool = True,
) -> dict[str, Path]:
    """
    Submit PDF to Mistral OCR API and write outputs.

    Returns dict of {output_type: Path}.
    """
    if not MISTRAL_API_KEY:
        raise EnvironmentError(
            "MISTRAL_API_KEY not set. Add it to .env or export MISTRAL_API_KEY=..."
        )

    file_path = Path(source)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    stem = file_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build request
    payload = {
        "model": MISTRAL_OCR_MODEL,
        "document": {
            "type": "document_url",
            "document_url": _encode_pdf(file_path),
        },
        "table_format": "html",   # tables as HTML inline
        "include_image_base64": False,
    }

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    print(f"⏳ Calling Mistral OCR…", end="", flush=True)
    t0 = time.time()
    response = requests.post(
        MISTRAL_OCR_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    elapsed = round(time.time() - t0, 1)
    print(f" done ({elapsed}s)")

    response.raise_for_status()
    result = response.json()
    result.setdefault("status", "complete")
    result.setdefault("success", True)

    # Parse pages
    page_data = _extract_markdown(result)
    full_md = "\n\n".join(p["markdown"] for p in page_data)
    result["markdown"] = full_md

    outputs: dict[str, Path] = {}

    if save_md and full_md:
        path = output_dir / f"{stem}.md"
        path.write_text(full_md, encoding="utf-8")
        outputs["markdown"] = path
        print(f"✅ Markdown  → {path}")

    if save_full:
        path = output_dir / f"{stem}_full_response.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs["full_response"] = path
        print(f"✅ Full JSON → {path}")

    # Build prediction files
    if uc_type and save_prediction:
        effective_doc_id = doc_id or stem
        prediction = build_prediction_metadata(page_data, uc_type, doc_id=effective_doc_id)

        if uc_type == "split":
            for part in ("text", "table"):
                path = output_dir / f"{stem}_{part}_prediction.json"
                path.write_text(
                    json.dumps(prediction[part], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                outputs[f"prediction_{part}"] = path
                print(f"✅ Prediction ({part:5s}) → {path}")
        else:
            path = output_dir / f"{stem}_prediction.json"
            path.write_text(
                json.dumps(prediction, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            outputs["prediction"] = path
            print(f"✅ Prediction → {path}")

    return outputs


# ── Prediction builders ───────────────────────────────────────

def build_scan_prediction(page_data: list[dict], doc_id: str = "") -> dict:
    """Build text prediction from Mistral page data."""
    pages = []
    for p in page_data:
        # Strip HTML tables from text (they go to table prediction)
        text = _strip_html_tables(p["markdown"])
        pages.append({"page_num": p["page_num"], "full_text": text})
    return {"doc_id": doc_id, "pages": pages}


def build_table_prediction(page_data: list[dict], doc_id: str = "") -> dict:
    """Build table prediction from Mistral page data (HTML tables inline in MD)."""
    pages = []
    for p in page_data:
        html_tables = _extract_html_table_strings(p["markdown"])
        if not html_tables:
            continue
        table_list = []
        for ti, html in enumerate(html_tables, start=1):
            cells = _html_table_to_cells(html)
            table_list.append({"table_id": ti, "html": html, "cells": cells})
        if table_list:
            pages.append({"page_num": p["page_num"], "tables": table_list})
    return {"doc_id": doc_id, "pages": pages}


def build_text_layer_prediction(page_data: list[dict], doc_id: str = "") -> dict:
    """Build text_layer prediction (same as scan for Mistral)."""
    pages = []
    for p in page_data:
        text = _strip_html_tables(p["markdown"])
        pages.append({
            "page_num": p["page_num"],
            "full_text": text,
            "blocks": [{"block_id": 1, "bbox": [0, 0, 1, 1], "text": text}],
        })
    return {"doc_id": doc_id, "pages": pages}


def build_split_prediction(page_data: list[dict], doc_id: str = "") -> dict[str, dict]:
    """Build both text and table predictions."""
    return {
        "text":  build_scan_prediction(page_data, doc_id=doc_id),
        "table": build_table_prediction(page_data, doc_id=doc_id),
    }


def build_prediction_metadata(
    page_data: list[dict],
    uc_type: str,
    doc_id: str = "",
) -> dict:
    builders = {
        "scan":       build_scan_prediction,
        "table":      build_table_prediction,
        "text_layer": build_text_layer_prediction,
        "split":      build_split_prediction,
    }
    if uc_type not in builders:
        raise ValueError(f"Unknown uc_type '{uc_type}'. Must be one of: {list(builders)}")
    return builders[uc_type](page_data, doc_id=doc_id)
