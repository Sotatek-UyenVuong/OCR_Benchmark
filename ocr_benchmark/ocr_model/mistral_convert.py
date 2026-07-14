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
    Extract per-page markdown + tables from Mistral response.

    With extract_header=True / extract_footer=True, Mistral puts headers/footers
    into separate fields (page.header, page.footer) and removes them from markdown.
    We use only page.markdown (body content) and page.tables.

    Returns: [{"page_num": int, "markdown": str, "tables": [{"id": str, "html": str}]}]
    """
    pages = result.get("pages") or []
    out = []
    for page in sorted(pages, key=lambda p: int(p.get("index", 0))):
        # Collect tables from the dedicated tables array
        tables = []
        for tbl in (page.get("tables") or []):
            html = tbl.get("content") or ""
            if html.strip().lower().startswith("<table"):
                tables.append({"id": tbl.get("id", ""), "html": html})

        # Use only body markdown — header/footer are already separated
        # (extract_header=True, extract_footer=True in request)
        markdown = str(page.get("markdown") or "")

        out.append({
            "page_num": int(page.get("index", 0)) + 1,  # 0-indexed → 1-indexed
            "markdown": markdown,
            "tables": tables,
            "header": page.get("header"),   # keep for build_scan_prediction
            "footer": page.get("footer"),   # keep for build_scan_prediction
        })
    return out


def _strip_html_tables(md: str) -> str:
    """Remove tables, image references, placeholder links from markdown text.
    Also strips heading markers (#) at start of lines — these are OCR formatting
    artifacts (Mistral adds them), not actual document content.
    Preserves newlines for readable multi-line full_text.
    """
    # Remove <table>...</table> blocks
    clean = re.sub(r"<table[\s\S]*?</table>", "\n", md, flags=re.IGNORECASE)
    # Remove Mistral table placeholder links: [tbl-0.html](tbl-0.html)
    clean = re.sub(r"\[[^\]]*\.html\]\([^\)]*\.html\)", "", clean)
    # Remove markdown image syntax: ![alt](url)
    clean = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", clean)
    # Remove HTML img tags
    clean = re.sub(r"<img\b[^>]*/?>", "", clean, flags=re.IGNORECASE)
    # Strip markdown heading markers at start of line (# ## ### etc)
    # These are OCR formatting artifacts, not document content
    clean = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]+", "", clean)
    # Collapse multiple spaces/tabs on same line (not newlines)
    clean = re.sub(r"[ \t]+", " ", clean)
    # Collapse 3+ consecutive newlines → 2
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean.strip()


def _extract_html_table_strings(md: str) -> list[str]:
    """Extract all <table>...</table> strings from markdown (fallback for old format)."""
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
    timeout: int = 300,   # increased: large PDFs take time to upload
    max_retries: int = 3,
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
        "table_format": "html",       # tables as HTML (in pages[].tables[].content)
        "include_image_base64": False, # no image blobs — saves bandwidth
        "extract_header": True,        # separate header from body markdown
        "extract_footer": True,        # separate footer from body markdown
    }

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    print(f"⏳ Calling Mistral OCR…", end="", flush=True)
    t0 = time.time()

    last_error: Exception | None = None
    response = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                MISTRAL_OCR_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"\n  ⚠️  Attempt {attempt + 1} failed ({e}), retry in {wait}s…", end="", flush=True)
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Mistral OCR failed after {max_retries} attempts: {last_error}"
                ) from last_error

    elapsed = round(time.time() - t0, 1)
    print(f" done ({elapsed}s)")
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
    """Build text prediction from Mistral page data.
    Combines header + body markdown + footer to match Marker convention
    (Marker includes all text regions: headers, footers, body).
    Table placeholders and image refs stripped.
    """
    pages = []
    for p in page_data:
        parts = []
        if p.get("header"):
            parts.append(str(p["header"]).strip())
        parts.append(p["markdown"].strip())
        if p.get("footer"):
            parts.append(str(p["footer"]).strip())
        combined = "\n".join(filter(None, parts))
        text = _strip_html_tables(combined)
        pages.append({"page_num": p["page_num"], "full_text": text})
    return {"doc_id": doc_id, "pages": pages}


def build_table_prediction(page_data: list[dict], doc_id: str = "") -> dict:
    """
    Build table prediction from Mistral page data.

    Tables come from pages[].tables (set by _extract_markdown),
    NOT from scanning the markdown text (which only has placeholder links).
    """
    pages = []
    for p in page_data:
        # Primary source: dedicated tables list from Mistral response
        html_tables = p.get("tables") or []

        # Fallback: scan markdown for inline <table> (old API behaviour)
        if not html_tables:
            for html in _extract_html_table_strings(p["markdown"]):
                html_tables.append({"id": "", "html": html})

        if not html_tables:
            continue

        table_list = []
        for ti, tbl in enumerate(html_tables, start=1):
            html = tbl["html"]
            cells = _html_table_to_cells(html)
            table_list.append({"table_id": ti, "html": html, "cells": cells})

        if table_list:
            pages.append({"page_num": p["page_num"], "tables": table_list})

    return {"doc_id": doc_id, "pages": pages}


def build_text_layer_prediction(page_data: list[dict], doc_id: str = "") -> dict:
    """Build text_layer prediction — header + body + footer, matching Marker convention."""
    pages = []
    for p in page_data:
        parts = []
        if p.get("header"): parts.append(str(p["header"]).strip())
        parts.append(p["markdown"].strip())
        if p.get("footer"): parts.append(str(p["footer"]).strip())
        text = _strip_html_tables("\n".join(filter(None, parts)))
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
