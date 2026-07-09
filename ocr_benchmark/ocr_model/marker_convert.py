#!/usr/bin/env python3
"""
marker_convert.py
-----------------
Convert a local file OR a remote URL to Markdown, JSON, and HTML
using the Datalab Convert API (https://www.datalab.to/api/v1/convert).

Note: API endpoint changed from /api/v1/marker to /api/v1/convert.

Also exports structured prediction metadata matching the benchmark schema:
  - scan     → {"pages": [{"page_num": int, "full_text": str}]}
  - table    → {"pages": [{"page_num": int, "tables": [{"table_id": int, "html": str, "cells": [...]}]}]}
  - text_layer → {"pages": [{"page_num": int, "full_text": str, "blocks": [...]}]}

Usage:
    python marker_convert.py <file_or_url> [options]

Examples:
    python marker_convert.py report.pdf
    python marker_convert.py https://example.com/doc.pdf
    python marker_convert.py invoice.png -o ./output -l vi,en -m accurate
    python marker_convert.py report.pdf --uc scan --doc-id scan_vi_001
"""

import os
import re
import sys
import json
import time
import mimetypes
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Load MARKER_API_KEY (with fallback to legacy API_KEY)
API_KEY: str = os.getenv("MARKER_API_KEY") or os.getenv("API_KEY", "")
# New endpoint: /api/v1/convert (previously /api/v1/marker)
API_URL: str = "https://www.datalab.to/api/v1/convert"

SUPPORTED_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".png", ".jpg", ".jpeg", ".webp",
}

# UC → benchmark type mapping
UC_TYPE: dict[str, str] = {
    "UC01": "scan", "UC02": "scan", "UC03": "scan",
    "UC04": "table", "UC05": "table", "UC06": "table",
    "UC07": "text_layer", "UC08": "text_layer", "UC09": "text_layer",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _stem_from_url(url: str) -> str:
    """Best-effort: derive a clean stem from a remote URL."""
    path_part = url.split("?")[0].rstrip("/")
    name = path_part.split("/")[-1] or "output"
    return Path(name).stem or "output"


def _poll(check_url: str, headers: dict, poll_timeout: int, interval: int = 3) -> dict:
    """Poll the Marker API until status == 'complete' or raise on error/timeout."""
    deadline = time.time() + poll_timeout
    while True:
        if time.time() > deadline:
            raise TimeoutError(f"Polling timed out after {poll_timeout}s")

        try:
            r = requests.get(check_url, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
        except requests.exceptions.Timeout:
            time.sleep(interval)
            continue

        status = data.get("status", "")
        if status == "complete":
            print()  # newline after dots
            return data
        if status == "error":
            raise RuntimeError(f"Marker API error: {data.get('error')}")

        print(".", end="", flush=True)
        time.sleep(interval)


# ──────────────────────────────────────────────────────────────────────────────
# Metadata builders — Marker JSON → benchmark prediction schema
# ──────────────────────────────────────────────────────────────────────────────

# Block types Marker dùng cho bảng / biểu mẫu — đều có HTML dạng <table>
TABLE_BLOCK_TYPES = {"Table", "Form"}


def _extract_blocks_from_marker(marker_json: dict) -> list[dict]:
    """
    Flatten all blocks from Marker JSON response into a list of
    {"block_id", "bbox", "text", "page_num", "block_type", "_raw_html"} dicts.

    Actual Marker API JSON structure (v2):
      {
        "children": [                   # one entry per page
          {
            "block_type": "Page",
            "bbox": [x1, y1, x2, y2],  # absolute pixels, page dimensions
            "children": [              # content blocks on this page
              {
                "block_type": "Text" | "Table" | "SectionHeader" | ...,
                "bbox": [x1, y1, x2, y2],
                "html": str,
                "markdown": str,       # plain-text representation
                "page": int,           # 0-indexed page number
                ...
              }
            ]
          }
        ],
        "metadata": {...}
      }
    """
    # Block types to skip — no useful text content for OCR evaluation
    SKIP_BLOCK_TYPES = {
        "Picture", "Figure", "Image",   # image blocks
        "Caption",                       # image captions
        "PageHeader",                    # usually just a logo/watermark
        "PageFooter",                    # page numbers, running headers
    }

    all_blocks: list[dict] = []

    # Top-level children are Page nodes
    page_nodes = marker_json.get("children") or []

    for page_idx, page_node in enumerate(page_nodes):
        # Page bbox gives us dimensions for normalisation
        page_bbox = page_node.get("bbox") or [0, 0, 1, 1]
        page_w = page_bbox[2] or 1   # x2 of page = width
        page_h = page_bbox[3] or 1   # y2 of page = height

        content_blocks = page_node.get("children") or []

        for i, blk in enumerate(content_blocks):
            block_type = blk.get("block_type", "Text")

            # Skip image/picture/figure blocks — not relevant for text eval
            if block_type in SKIP_BLOCK_TYPES:
                continue

            # page_num: prefer block's own "page" field (0-indexed) + 1
            raw_page = blk.get("page")
            if raw_page is not None:
                page_num = int(raw_page) + 1
            else:
                page_num = page_idx + 1

            bbox_abs = blk.get("bbox") or [0, 0, 0, 0]
            bbox_norm = [
                round(bbox_abs[0] / page_w, 6),
                round(bbox_abs[1] / page_h, 6),
                round(bbox_abs[2] / page_w, 6),
                round(bbox_abs[3] / page_h, 6),
            ]

            text = _block_text(blk)
            if not text:
                continue  # skip empty blocks

            all_blocks.append({
                "block_id": i + 1,
                "bbox": bbox_norm,
                "text": text,
                "page_num": page_num,
                "block_type": block_type,
                "_raw_html": blk.get("html", ""),
            })

    return all_blocks


def _block_text(blk: dict) -> str:
    """
    Extract plain text from a Marker block dict.

    Priority:
      1. "markdown" field — when non-empty (Marker v2 sometimes populates this)
      2. Strip HTML tags from "html" field — most reliable in practice
      3. "lines[].spans[].text" — Marker v1 fallback
    """
    # 1. markdown field — skip if empty (Marker v2 often leaves it blank)
    md = (blk.get("markdown") or "").strip()
    if md:
        # Remove markdown image tags  ![alt](src)
        text = re.sub(r"!\[.*?\]\([^)]*\)", "", md)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            return text

    # 2. Parse HTML (primary path for Marker v2)
    html = (blk.get("html") or "").strip()
    if html:
        # Remove <img ...> tags entirely (logos, icons, figures)
        text = re.sub(r"<img\b[^>]*>", "", html, flags=re.I)
        # Remove all remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode common HTML entities
        text = (text
                .replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&nbsp;", " ")
                .replace("&#39;", "'")
                .replace("&quot;", '"'))
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            return text

    # 3. lines/spans (Marker v1 fallback)
    lines = blk.get("lines") or []
    if lines:
        parts = []
        for line in lines:
            for span in (line.get("spans") or []):
                parts.append(span.get("text", ""))
        text = " ".join(parts).strip()
        if text:
            return text

    return ""


def _html_to_cells(html: str) -> list[dict]:
    """
    Parse a simple HTML table string into a list of cell dicts:
      {"row": int, "col": int, "rowspan": int, "colspan": int,
       "text": str, "is_header": bool}

    Uses a basic regex approach — sufficient for well-formed Marker output.
    For production use, consider lxml or html.parser.
    """
    cells: list[dict] = []
    html_lower = html.lower()

    # Strip thead/tbody/tfoot wrappers
    html_clean = re.sub(r"</?(?:thead|tbody|tfoot)[^>]*>", "", html, flags=re.I)

    row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.I | re.S)
    cell_pattern = re.compile(r"<(th|td)([^>]*)>(.*?)</(?:th|td)>", re.I | re.S)
    span_pattern_r = re.compile(r'rowspan=["\']?(\d+)["\']?', re.I)
    span_pattern_c = re.compile(r'colspan=["\']?(\d+)["\']?', re.I)

    occupied: set[tuple[int, int]] = set()  # tracks cells already occupied by rowspan

    row_idx = 0
    for row_match in row_pattern.finditer(html_clean):
        row_html = row_match.group(1)
        col_idx = 0

        for cell_match in cell_pattern.finditer(row_html):
            tag = cell_match.group(1).lower()  # "th" or "td"
            attrs = cell_match.group(2)
            inner = cell_match.group(3)

            # Skip occupied cells (from previous rowspan)
            while (row_idx, col_idx) in occupied:
                col_idx += 1

            rowspan = int(span_pattern_r.search(attrs).group(1)) if span_pattern_r.search(attrs) else 1
            colspan = int(span_pattern_c.search(attrs).group(1)) if span_pattern_c.search(attrs) else 1

            # Mark future rows as occupied
            for r in range(row_idx, row_idx + rowspan):
                for c in range(col_idx, col_idx + colspan):
                    if r > row_idx or c > col_idx:
                        occupied.add((r, c))

            text = re.sub(r"<[^>]+>", " ", inner)
            text = re.sub(r"\s+", " ", text).strip()

            cells.append({
                "row": row_idx,
                "col": col_idx,
                "rowspan": rowspan,
                "colspan": colspan,
                "text": text,
                "is_header": tag == "th",
            })

            col_idx += colspan

        row_idx += 1

    return cells


# ──────────────────────────────────────────────────────────────────────────────
# Prediction metadata builders per UC type
# ──────────────────────────────────────────────────────────────────────────────

def build_scan_prediction(
    marker_json: dict,
    doc_id: str = "",
) -> dict:
    """
    Build benchmark prediction JSON for scan UC (UC01-03).

    Output schema:
        {
            "doc_id": str,
            "pages": [
                {"page_num": int, "full_text": str}
            ]
        }
    """
    blocks = _extract_blocks_from_marker(marker_json)

    # Group blocks by page, concatenate text in order
    pages_text: dict[int, list[str]] = {}
    for blk in blocks:
        pn = blk["page_num"]
        pages_text.setdefault(pn, [])
        if blk["text"]:
            pages_text[pn].append(blk["text"])

    pages = [
        {"page_num": pn, "full_text": "\n".join(texts)}
        for pn, texts in sorted(pages_text.items())
    ]

    return {"doc_id": doc_id, "pages": pages}


def build_table_prediction(
    marker_json: dict,
    doc_id: str = "",
) -> dict:
    """
    Build benchmark prediction JSON for table UC (UC04-06).

    Output schema:
        {
            "doc_id": str,
            "pages": [
                {
                    "page_num": int,
                    "tables": [
                        {
                            "table_id": int,
                            "html": str,
                            "cells": [
                                {
                                    "row": int, "col": int,
                                    "rowspan": int, "colspan": int,
                                    "text": str, "is_header": bool
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    """
    blocks = _extract_blocks_from_marker(marker_json)

    # Group table blocks by page
    pages_tables: dict[int, list[dict]] = {}
    table_counter: dict[int, int] = {}

    for blk in blocks:
        if blk["block_type"] not in TABLE_BLOCK_TYPES:
            continue
        pn = blk["page_num"]
        pages_tables.setdefault(pn, [])
        table_counter[pn] = table_counter.get(pn, 0) + 1

        html = blk["_raw_html"] or ""
        cells = _html_to_cells(html)

        pages_tables[pn].append({
            "table_id": table_counter[pn],
            "html": html,
            "cells": cells,
        })

    pages = [
        {"page_num": pn, "tables": tables}
        for pn, tables in sorted(pages_tables.items())
    ]

    return {"doc_id": doc_id, "pages": pages}


def build_text_layer_prediction(
    marker_json: dict,
    doc_id: str = "",
) -> dict:
    """
    Build benchmark prediction JSON for text-layer PDF UC (UC07-09).

    Output schema:
        {
            "doc_id": str,
            "pages": [
                {
                    "page_num": int,
                    "full_text": str,
                    "blocks": [
                        {
                            "block_id": int,
                            "bbox": [x1, y1, x2, y2],   # normalised 0-1
                            "text": str
                        }
                    ]
                }
            ]
        }
    """
    blocks = _extract_blocks_from_marker(marker_json)

    # Group blocks by page
    pages_blocks: dict[int, list[dict]] = {}
    for blk in blocks:
        pn = blk["page_num"]
        pages_blocks.setdefault(pn, [])
        pages_blocks[pn].append({
            "block_id": blk["block_id"],
            "bbox": blk["bbox"],
            "text": blk["text"],
        })

    pages = []
    for pn, page_blks in sorted(pages_blocks.items()):
        full_text = "\n".join(b["text"] for b in page_blks if b["text"])
        pages.append({
            "page_num": pn,
            "full_text": full_text,
            "blocks": page_blks,
        })

    return {"doc_id": doc_id, "pages": pages}


def build_split_prediction(
    marker_json: dict,
    doc_id: str = "",
) -> dict[str, dict]:
    """
    Build BOTH text and table prediction JSONs from a single Marker response.

    Marker already separates block_type=Table from text blocks — this function
    exploits that to run eval_scan AND eval_table on the same source file.

    Text prediction  — uses only non-Table blocks (Text, SectionHeader, ListGroup, …)
    Table prediction — uses only Table blocks

    Returns:
        {
            "text":  { "doc_id": str, "pages": [{"page_num", "full_text"}] },
            "table": { "doc_id": str, "pages": [{"page_num", "tables": [...]}] },
        }

    Usage:
        split = build_split_prediction(marker_json, doc_id="scan_vi_001")
        text_pred  = split["text"]   # → eval_scan()
        table_pred = split["table"]  # → eval_table()
    """
    # ── Text prediction (exclude Table blocks) ────────────────────────────────
    blocks = _extract_blocks_from_marker(marker_json)

    TEXT_BLOCK_TYPES = {
        "Text", "SectionHeader", "ListGroup",
        "PageHeader", "PageFooter", "Caption",
    }
    # Re-use skip logic but keep everything that isn't a Table
    pages_text: dict[int, list[str]] = {}
    for blk in blocks:
        # block_type Table/Form → goes to table prediction only
        if blk["block_type"] in TABLE_BLOCK_TYPES:
            continue
        pn = blk["page_num"]
        pages_text.setdefault(pn, [])
        if blk["text"]:
            pages_text[pn].append(blk["text"])

    text_pages = [
        {"page_num": pn, "full_text": "\n".join(texts)}
        for pn, texts in sorted(pages_text.items())
    ]
    text_pred = {"doc_id": doc_id, "pages": text_pages}

    # ── Table prediction ──────────────────────────────────────────────────────
    pages_tables: dict[int, list[dict]] = {}
    table_counter: dict[int, int] = {}

    for blk in blocks:
        if blk["block_type"] not in TABLE_BLOCK_TYPES:
            continue
        pn = blk["page_num"]
        pages_tables.setdefault(pn, [])
        table_counter[pn] = table_counter.get(pn, 0) + 1

        html = blk["_raw_html"] or ""
        cells = _html_to_cells(html)

        pages_tables[pn].append({
            "table_id": table_counter[pn],
            "html": html,
            "cells": cells,
        })

    table_pages = [
        {"page_num": pn, "tables": tables}
        for pn, tables in sorted(pages_tables.items())
    ]
    table_pred = {"doc_id": doc_id, "pages": table_pages}

    return {"text": text_pred, "table": table_pred}


def build_prediction_metadata(
    marker_json: dict,
    uc_type: str,
    doc_id: str = "",
) -> dict | dict[str, dict]:
    """
    Dispatch to the correct builder based on uc_type.

    Args:
        marker_json: Parsed JSON from Marker API full response.
        uc_type:     One of "scan" | "table" | "text_layer" | "split".
                     "split" returns {"text": ..., "table": ...} — two prediction
                     dicts from a single file (useful when source has mixed content).
        doc_id:      Document identifier string.

    Returns:
        - Single prediction dict for "scan" / "table" / "text_layer"
        - {"text": dict, "table": dict} for "split"

    Raises:
        ValueError: If uc_type is not recognised.
    """
    builders = {
        "scan": build_scan_prediction,
        "table": build_table_prediction,
        "text_layer": build_text_layer_prediction,
        "split": build_split_prediction,
    }
    if uc_type not in builders:
        raise ValueError(
            f"Unknown uc_type '{uc_type}'. Must be one of: {list(builders)}"
        )
    return builders[uc_type](marker_json, doc_id=doc_id)


# ──────────────────────────────────────────────────────────────────────────────
# Core convert
# ──────────────────────────────────────────────────────────────────────────────

def convert(
    source: str,
    output_dir: Path,
    langs: str = "",
    mode: str = "fast",
    force_ocr: bool = False,
    paginate: bool = True,
    timeout: int = 300,
    max_retries: int = 3,
    save_md: bool = True,
    save_json: bool = True,
    save_html: bool = True,
    save_full: bool = True,
    # Prediction metadata
    uc_type: str | None = None,
    doc_id: str = "",
    save_prediction: bool = True,
) -> dict[str, Path]:
    """
    Submit *source* (local path or URL) to Marker and write outputs to *output_dir*.

    Args:
        source:           Local file path or HTTP(S) URL.
        output_dir:       Directory to write output files.
        langs:            Comma-separated OCR language codes (e.g. "vi,en").
        mode:             Marker quality mode: "fast" | "balanced" | "accurate".
        force_ocr:        Force OCR even when text layer exists.
        paginate:         Include page separators in Markdown output.
        timeout:          Per-request timeout in seconds.
        max_retries:      Maximum upload retry attempts.
        save_md:          Save Markdown output.
        save_json:        Save Marker JSON output.
        save_html:        Save HTML output.
        save_full:        Save full Marker API response.
        uc_type:          Benchmark UC type: "scan" | "table" | "text_layer".
                          If provided (and save_prediction=True), writes
                          <stem>_prediction.json with benchmark-ready metadata.
        doc_id:           Document ID for prediction metadata. Defaults to stem.
        save_prediction:  Write the benchmark prediction JSON file.

    Returns:
        dict mapping output type → Path:
        {
            "markdown": Path,
            "json": Path,
            "html": Path,
            "full_response": Path,
            "prediction": Path,   # only when uc_type is set
        }
    """
    if not API_KEY:
        raise EnvironmentError(
            "MARKER_API_KEY not set. Add it to .env or export MARKER_API_KEY=..."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    # New API uses X-API-Key (uppercase)
    headers = {"X-API-Key": API_KEY}

    # ── Payload — /api/v1/convert params ─────────────────────────────────────
    # output_format: "markdown" | "html" | "json" | "chunks"
    # Only 1 format per request in convert API
    # Use "json" to get block-level data for prediction, fallback to markdown
    output_format = "json" if (save_json or save_prediction) else ("html" if save_html else "markdown")

    payload: dict = {
        "output_format": output_format,
        "mode": mode,
        "paginate": str(paginate).lower(),
        "skip_cache": "false",
        "disable_image_extraction": "true",
    }

    # Language hint (optional, helps OCR accuracy)
    if langs:
        payload["language"] = langs.split(",")[0]  # convert API takes single lang

    # ── Upload / submit ───────────────────────────────────────────────────────
    last_error: Exception | None = None
    result: dict | None = None

    for attempt in range(max_retries):
        try:
            if _is_url(source):
                stem = _stem_from_url(source)
                response = requests.post(
                    API_URL,
                    data={**payload, "file_url": source},  # convert API uses file_url
                    headers=headers,
                    timeout=timeout,
                )
            else:
                file_path = Path(source)
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                stem = file_path.stem
                mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
                with open(file_path, "rb") as fh:
                    response = requests.post(
                        API_URL,
                        files={"file": (file_path.name, fh, mime)},
                        data=payload,
                        headers=headers,
                        timeout=timeout,
                    )

            response.raise_for_status()
            result = response.json()

            if not result.get("success"):
                raise RuntimeError(f"Submission failed: {result.get('error', 'unknown')}")

            break

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"⚠️  Attempt {attempt + 1} failed, retrying in {wait}s…")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Upload failed after {max_retries} attempts: {last_error}"
                ) from last_error

    if result is None:
        raise RuntimeError(f"No result after retries: {last_error}")

    # ── Poll ──────────────────────────────────────────────────────────────────
    # Convert API returns request_id + request_check_url
    request_id = result.get("request_id") or result.get("lookup_key")
    check_url = result.get("request_check_url") or f"{API_URL}/{request_id}"

    if not request_id:
        raise RuntimeError(f"No request_id in API response. Keys: {list(result.keys())}")

    print(f"⏳ Processing", end="", flush=True)
    final = _poll(check_url, headers, poll_timeout=timeout * 2)

    # ── Save raw Convert API outputs ─────────────────────────────────────────
    outputs: dict[str, Path] = {}

    # Convert API trả markdown/html/json tùy output_format
    if save_md:
        md_content = final.get("markdown", "")
        if not md_content and output_format != "markdown":
            # Nếu request format là json, markdown không có — skip
            pass
        elif md_content:
            path = output_dir / f"{stem}.md"
            path.write_text(md_content, encoding="utf-8")
            outputs["markdown"] = path
            print(f"✅ Markdown  → {path}")

    if save_json:
        # Convert API: json output là object trực tiếp (children/blocks structure)
        json_content = final.get("json") or final
        path = output_dir / f"{stem}.json"
        path.write_text(
            json.dumps(json_content, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        outputs["json"] = path
        print(f"✅ JSON      → {path}")

    if save_html:
        html_content = final.get("html", "")
        if html_content:
            path = output_dir / f"{stem}.html"
            path.write_text(html_content, encoding="utf-8")
            outputs["html"] = path
            print(f"✅ HTML      → {path}")

    if save_full:
        path = output_dir / f"{stem}_full_response.json"
        path.write_text(
            json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        outputs["full_response"] = path
        print(f"✅ Full JSON → {path}")

    # ── Build & save prediction metadata ─────────────────────────────────────
    if uc_type and save_prediction:
        # Convert API: json field IS the root object (children/blocks)
        # Marker cũ: final["json"] là nested object
        marker_json = final.get("json") or final
        effective_doc_id = doc_id or stem
        prediction = build_prediction_metadata(marker_json, uc_type, doc_id=effective_doc_id)

        if uc_type == "split":
            # Save two files: <stem>_text_prediction.json + <stem>_table_prediction.json
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
                json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            outputs["prediction"] = path
            print(f"✅ Prediction → {path}")

    return outputs


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a file or URL to Markdown / JSON / HTML via Marker API "
                    "and optionally export benchmark prediction metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "source",
        help="Local file path or remote URL (http/https)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory (default: same folder as input file, or ./marker_output for URLs)",
    )
    parser.add_argument(
        "-l", "--langs",
        default="vi,en",
        help="OCR languages, comma-separated (default: vi,en)",
    )
    parser.add_argument(
        "-m", "--mode",
        default="balanced",
        choices=["fast", "balanced", "accurate"],
        help="OCR quality mode (default: balanced)",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR even if text layer exists",
    )
    parser.add_argument(
        "--no-paginate",
        action="store_true",
        help="Disable page separators in Markdown output",
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=300,
        help="Per-request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "-r", "--max-retries",
        type=int,
        default=3,
        help="Max upload retry attempts (default: 3)",
    )
    parser.add_argument("--no-md",   action="store_true", help="Skip saving Markdown")
    parser.add_argument("--no-json", action="store_true", help="Skip saving JSON")
    parser.add_argument("--no-html", action="store_true", help="Skip saving HTML")
    parser.add_argument("--no-full", action="store_true", help="Skip saving full API response JSON")

    # Prediction metadata options
    pred_group = parser.add_argument_group("prediction metadata")
    pred_group.add_argument(
        "--uc",
        choices=["scan", "table", "text_layer", "split"] + list(UC_TYPE.keys()),
        default=None,
        help=(
            "Benchmark UC type for prediction export. "
            "Accepts 'scan' | 'table' | 'text_layer' | 'split' or UC code e.g. 'UC01'. "
            "'split' builds BOTH text and table predictions from a single file, "
            "saving <stem>_text_prediction.json and <stem>_table_prediction.json. "
            "When set, writes prediction file(s) in benchmark format."
        ),
    )
    pred_group.add_argument(
        "--doc-id",
        default="",
        help="Document ID for prediction metadata (default: input file stem)",
    )
    pred_group.add_argument(
        "--no-prediction",
        action="store_true",
        help="Skip writing prediction metadata even when --uc is set",
    )

    args = parser.parse_args()

    # Resolve UC type string
    uc_type: str | None = None
    if args.uc:
        # "UC01"→"scan", "UC04"→"table", etc.  "split"/"scan"/... pass through as-is
        uc_type = UC_TYPE.get(args.uc, args.uc)

    # Resolve output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif _is_url(args.source):
        out_dir = Path("marker_output")
    else:
        out_dir = Path(args.source).parent / "marker_output"

    print(f"{'='*60}")
    print(f"  Marker Convert")
    print(f"{'='*60}")
    print(f"  Source  : {args.source}")
    print(f"  Output  : {out_dir}")
    print(f"  Langs   : {args.langs}  |  Mode: {args.mode}")
    if uc_type:
        print(f"  UC type : {uc_type}")
    print(f"{'='*60}\n")

    try:
        outputs = convert(
            source=args.source,
            output_dir=out_dir,
            langs=args.langs,
            mode=args.mode,
            force_ocr=args.force_ocr,
            paginate=not args.no_paginate,
            timeout=args.timeout,
            max_retries=args.max_retries,
            save_md=not args.no_md,
            save_json=not args.no_json,
            save_html=not args.no_html,
            save_full=not args.no_full,
            uc_type=uc_type,
            doc_id=args.doc_id,
            save_prediction=not args.no_prediction,
        )

        print(f"\n{'='*60}")
        print(f"  Done! {len(outputs)} file(s) saved to: {out_dir}")
        print(f"{'='*60}")

    except Exception as exc:
        print(f"\n❌ {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
