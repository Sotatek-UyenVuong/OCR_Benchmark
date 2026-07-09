"""
test_marker_pipeline.py
-----------------------
Test toàn bộ pipeline từ Marker response → prediction JSON → eval metrics.
Chạy offline: dùng full_response.json đã có sẵn, KHÔNG gọi API.

Usage:
    python -m pytest tests/test_marker_pipeline.py -v
    python tests/test_marker_pipeline.py          # chạy trực tiếp
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FULL_RESPONSE = ROOT / "raw/scan/en/marker_output/scan_en_001_full_response.json"

from ocr_benchmark.ocr_model.marker_convert import (
    _extract_blocks_from_marker,
    _block_text,
    _html_to_cells,
    build_scan_prediction,
    build_table_prediction,
    build_text_layer_prediction,
    build_split_prediction,
    build_prediction_metadata,
)
from ocr_benchmark.eval.scan import eval_scan
from ocr_benchmark.eval.table import eval_table
from ocr_benchmark.eval.text_layer import eval_text_layer
from ocr_benchmark.normalize import normalize_for_text_benchmark

# Adapters for old dict-returning metric functions used in tests
from ocr_benchmark.eval.scan import (
    _compute_cer_detail,
    _compute_wer_detail,
    _compute_nwer,
)
from ocr_benchmark.eval.table import _compute_teds


def compute_cer(ground_truth, prediction, doc_id="", page_num=1, include_alignment=False):
    result = _compute_cer_detail(ground_truth, prediction, include_alignment)
    return {
        "doc_id": doc_id, "page_num": page_num,
        "cer": result["cer"], "cer_detail": result["cer_detail"],
        "ground_truth": ground_truth, "prediction": prediction,
        "char_alignment": result.get("char_alignment"),
    }


def compute_wer(ground_truth, prediction, doc_id="", page_num=1):
    result = _compute_wer_detail(ground_truth, prediction)
    return {
        "doc_id": doc_id, "page_num": page_num,
        "wer": result["wer"], "wer_detail": result["wer_detail"],
        "ground_truth": ground_truth, "prediction": prediction,
    }


def compute_nwer(ground_truth, prediction, doc_id="", page_num=1):
    return {
        "doc_id": doc_id, "page_num": page_num,
        "nwer": _compute_nwer(ground_truth, prediction),
    }


def compute_teds(gt_html, pred_html, doc_id="", page_num=1, table_id=1):
    return _compute_teds(gt_html, pred_html, doc_id, page_num, table_id)


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

def load_marker_json() -> dict:
    with open(FULL_RESPONSE, encoding="utf-8") as f:
        return json.load(f)["json"]


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭  SKIP"

_results: list[dict] = []

def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    _results.append({"name": name, "status": status, "detail": detail})
    icon = status
    print(f"  {icon}  {name}" + (f"  — {detail}" if detail else ""))
    return condition


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────
# 1. Block extraction
# ─────────────────────────────────────────────────────────────────

def test_block_extraction(mj: dict):
    section("1. Block Extraction")
    blocks = _extract_blocks_from_marker(mj)

    check("blocks list không rỗng", len(blocks) > 0, f"{len(blocks)} blocks")
    check("mỗi block có đủ keys",
          all({"block_id","bbox","text","page_num","block_type"}.issubset(b) for b in blocks))

    # Không có Picture/PageHeader/PageFooter
    bad_types = {b["block_type"] for b in blocks if b["block_type"]
                 in {"Picture","Figure","Image","Caption","PageHeader","PageFooter"}}
    check("không có block type ảnh/header/footer bị lọt qua", len(bad_types) == 0,
          f"còn lọt: {bad_types}" if bad_types else "")

    # text không rỗng
    empty_text = [b for b in blocks if not b["text"].strip()]
    check("không có block text rỗng", len(empty_text) == 0,
          f"{len(empty_text)} block rỗng" if empty_text else "")

    # bbox normalised [0,1]
    bad_bbox = [b for b in blocks
                if not all(0.0 <= v <= 1.0 for v in b["bbox"])]
    check("bbox normalised trong [0,1]", len(bad_bbox) == 0,
          f"{len(bad_bbox)} block bbox ngoài [0,1]" if bad_bbox else "")

    # Image text không lọt vào
    img_leaked = [b for b in blocks if "IATA logo" in b["text"] or "Image:" in b["text"]]
    check("không có alt text ảnh lọt vào text", len(img_leaked) == 0,
          f"lọt: {[b['text'][:40] for b in img_leaked]}" if img_leaked else "")

    pages = sorted({b["page_num"] for b in blocks})
    check("page_num bắt đầu từ 1", min(pages) == 1, f"pages: {pages}")

    table_blocks = [b for b in blocks if b["block_type"] == "Table"]
    check("có Table blocks", len(table_blocks) > 0, f"{len(table_blocks)} tables")

    return blocks


# ─────────────────────────────────────────────────────────────────
# 2. _block_text
# ─────────────────────────────────────────────────────────────────

def test_block_text():
    section("2. _block_text()")

    # html path
    blk_html = {"html": "<p>Hello <b>World</b></p>"}
    check("html strip tags", _block_text(blk_html) == "Hello World",
          repr(_block_text(blk_html)))

    # img removed
    blk_img = {"html": '<p><img alt="logo" src="x.jpg"/> Some text</p>'}
    result = _block_text(blk_img)
    check("img tag bị xóa", "logo" not in result and "Some text" in result,
          repr(result))

    # html entities
    blk_ent = {"html": "<p>AT&amp;T &lt;Corp&gt;</p>"}
    check("html entities decoded", _block_text(blk_ent) == "AT&T <Corp>",
          repr(_block_text(blk_ent)))

    # markdown path
    blk_md = {"markdown": "Hello **World**", "html": ""}
    check("markdown path hoạt động", "Hello" in _block_text(blk_md),
          repr(_block_text(blk_md)))

    # markdown image stripped
    blk_md_img = {"markdown": "![logo](logo.png) Real text", "html": ""}
    result = _block_text(blk_md_img)
    check("markdown image bị strip", "logo.png" not in result and "Real text" in result,
          repr(result))

    # empty block
    check("empty block trả rỗng", _block_text({}) == "", repr(_block_text({})))


# ─────────────────────────────────────────────────────────────────
# 3. _html_to_cells
# ─────────────────────────────────────────────────────────────────

def test_html_to_cells():
    section("3. _html_to_cells()")

    simple = "<table><tr><th>Name</th><th>Age</th></tr><tr><td>Alice</td><td>30</td></tr></table>"
    cells = _html_to_cells(simple)
    check("simple table: 4 cells", len(cells) == 4, f"{len(cells)}")
    check("th → is_header=True", cells[0]["is_header"] is True)
    check("td → is_header=False", cells[2]["is_header"] is False)
    check("text đúng", cells[0]["text"] == "Name" and cells[2]["text"] == "Alice",
          f"[0]={cells[0]['text']!r} [2]={cells[2]['text']!r}")

    # colspan
    span_html = "<table><tr><th colspan='2'>Header</th></tr><tr><td>A</td><td>B</td></tr></table>"
    span_cells = _html_to_cells(span_html)
    check("colspan parsed", span_cells[0]["colspan"] == 2, f"colspan={span_cells[0]['colspan']}")

    # rowspan
    rs_html = "<table><tr><td rowspan='2'>R</td><td>A</td></tr><tr><td>B</td></tr></table>"
    rs_cells = _html_to_cells(rs_html)
    check("rowspan parsed", rs_cells[0]["rowspan"] == 2, f"rowspan={rs_cells[0]['rowspan']}")

    # empty html
    check("empty html → [] cells", _html_to_cells("") == [], repr(_html_to_cells("")))


# ─────────────────────────────────────────────────────────────────
# 4. Build prediction builders
# ─────────────────────────────────────────────────────────────────

def test_build_scan(mj: dict):
    section("4. build_scan_prediction()")
    pred = build_scan_prediction(mj, doc_id="scan_en_001")

    check("doc_id đúng", pred["doc_id"] == "scan_en_001")
    check("có pages", len(pred["pages"]) > 0, f"{len(pred['pages'])} pages")
    check("mỗi page có page_num + full_text",
          all("page_num" in p and "full_text" in p for p in pred["pages"]))
    check("page_num bắt đầu từ 1", pred["pages"][0]["page_num"] == 1)
    check("full_text không rỗng", all(len(p["full_text"]) > 0 for p in pred["pages"]))

    # Không có table HTML lọt vào full_text
    all_text = " ".join(p["full_text"] for p in pred["pages"])
    check("full_text không chứa <table> raw HTML", "<table>" not in all_text.lower())

    return pred


def test_build_table(mj: dict):
    section("5. build_table_prediction()")
    pred = build_table_prediction(mj, doc_id="scan_en_001")

    check("doc_id đúng", pred["doc_id"] == "scan_en_001")
    check("có pages với table", len(pred["pages"]) > 0, f"{len(pred['pages'])} pages")
    check("mỗi page có page_num + tables",
          all("page_num" in p and "tables" in p for p in pred["pages"]))

    for p in pred["pages"]:
        for t in p["tables"]:
            check(f"  page {p['page_num']} table {t['table_id']} có html",
                  len(t["html"]) > 0, f"{len(t['html'])} chars")
            check(f"  page {p['page_num']} table {t['table_id']} có cells",
                  len(t["cells"]) > 0, f"{len(t['cells'])} cells")
            for c in t["cells"][:1]:
                check(f"  cell có đủ keys",
                      all(k in c for k in ["row","col","rowspan","colspan","text","is_header"]))

    return pred


def test_build_text_layer(mj: dict):
    section("6. build_text_layer_prediction()")
    pred = build_text_layer_prediction(mj, doc_id="scan_en_001")

    check("doc_id đúng", pred["doc_id"] == "scan_en_001")
    check("có pages", len(pred["pages"]) > 0)
    check("mỗi page có full_text + blocks",
          all("full_text" in p and "blocks" in p for p in pred["pages"]))

    for p in pred["pages"][:2]:
        check(f"  page {p['page_num']} blocks không rỗng", len(p["blocks"]) > 0,
              f"{len(p['blocks'])} blocks")
        for b in p["blocks"][:1]:
            check("  block có block_id + bbox + text",
                  all(k in b for k in ["block_id","bbox","text"]))
            check("  bbox 4 values", len(b["bbox"]) == 4)

    return pred


# ─────────────────────────────────────────────────────────────────
# 7. build_split_prediction
# ─────────────────────────────────────────────────────────────────

def test_build_split(mj: dict):
    section("7. build_split_prediction()")
    split = build_split_prediction(mj, doc_id="scan_en_001")

    check("trả về dict có key 'text' và 'table'",
          "text" in split and "table" in split)

    text_pred = split["text"]
    table_pred = split["table"]

    check("text_pred có pages", len(text_pred["pages"]) > 0)
    check("table_pred có pages", len(table_pred["pages"]) > 0)

    # text_pred không chứa Table block text riêng (đã tách)
    # table_pred phải khớp với build_table_prediction
    table_standalone = build_table_prediction(mj, doc_id="scan_en_001")
    check("table trong split == build_table standalone",
          table_pred["pages"] == table_standalone["pages"])

    # Không có block nào bị đếm 2 lần:
    # page xuất hiện trong cả text + table là bình thường (page có cả 2 loại)
    # nhưng full_text của text_pred không được chứa raw HTML bảng
    for p in text_pred["pages"]:
        check(f"  text page {p['page_num']} không có raw <table> HTML",
              "<table" not in p["full_text"].lower())

    return split


# ─────────────────────────────────────────────────────────────────
# 8. build_prediction_metadata dispatch
# ─────────────────────────────────────────────────────────────────

def test_dispatch(mj: dict):
    section("8. build_prediction_metadata dispatch")

    for uc in ["scan", "table", "text_layer", "split"]:
        try:
            result = build_prediction_metadata(mj, uc_type=uc, doc_id="test")
            if uc == "split":
                check(f"dispatch '{uc}' → dict với text+table",
                      "text" in result and "table" in result)
            else:
                check(f"dispatch '{uc}' → có pages",
                      "pages" in result and len(result["pages"]) >= 0)
        except Exception as e:
            check(f"dispatch '{uc}'", False, str(e))

    # Invalid UC
    try:
        build_prediction_metadata(mj, uc_type="invalid")
        check("uc_type invalid raise ValueError", False, "không raise")
    except ValueError:
        check("uc_type invalid raise ValueError", True)


# ─────────────────────────────────────────────────────────────────
# 9. Metrics: CER / WER / TEDS
# ─────────────────────────────────────────────────────────────────

def test_metrics_cer():
    section("9a. compute_cer()")

    # Perfect match
    r = compute_cer("hello", "hello")
    check("perfect match → cer=0", r["cer"] == 0.0, f"cer={r['cer']}")

    # All wrong
    r = compute_cer("abc", "xyz")
    check("all substitutions → cer=1.0", r["cer"] == 1.0, f"cer={r['cer']}")
    check("substitutions=3", r["cer_detail"]["substitutions"] == 3)

    # Empty GT
    r = compute_cer("", "something")
    check("empty GT → cer=0", r["cer"] == 0.0)

    # Insertion
    r = compute_cer("ab", "abc")
    check("insertion counted", r["cer_detail"]["insertions"] >= 1)

    # Deletion
    r = compute_cer("abc", "ab")
    check("deletion counted", r["cer_detail"]["deletions"] >= 1)

    # With alignment
    r = compute_cer("abc", "axc", include_alignment=True)
    check("alignment list trả về", r["char_alignment"] is not None)
    check("alignment chứa substitution", any(a["type"] == "substitution" for a in r["char_alignment"]))


def test_metrics_wer():
    section("9b. compute_wer() + compute_nwer()")

    r = compute_wer("hello world", "hello world")
    check("wer perfect=0", r["wer"] == 0.0)

    r = compute_wer("hello world", "hello")
    check("wer deletion > 0", r["wer"] > 0)

    r = compute_nwer("Hello, World!", "hello world")
    check("nwer case+punct insensitive ≈ 0", r["nwer"] < 0.1, f"nwer={r['nwer']}")

    r = compute_wer("", "something")
    check("wer empty GT = 0", r["wer"] == 0.0)


def test_metrics_teds():
    section("9c. compute_teds()")

    gt = "<table><tr><th>Name</th><th>Age</th></tr><tr><td>Alice</td><td>30</td></tr></table>"
    pred_perfect = gt
    pred_wrong   = "<table><tr><td>Name</td><td>Age</td></tr><tr><td>Bob</td><td>25</td></tr></table>"
    pred_empty   = ""

    r = compute_teds(gt, pred_perfect)
    check("teds perfect=1.0", r["teds"] == 1.0, f"teds={r['teds']}")
    check("cell_diff rỗng khi perfect", len(r["cell_diff"]) == 0)

    r = compute_teds(gt, pred_wrong)
    check("teds wrong < 1.0", r["teds"] < 1.0, f"teds={r['teds']}")
    check("cell_diff có lỗi", len(r["cell_diff"]) > 0)

    r = compute_teds(gt, pred_empty)
    check("teds empty pred < 1.0", r["teds"] < 1.0, f"teds={r['teds']}")

    r = compute_teds("", "")
    check("teds both empty = 1.0", r["teds"] == 1.0)


# ─────────────────────────────────────────────────────────────────
# 10. eval functions với prediction từ Marker
# ─────────────────────────────────────────────────────────────────

def test_eval_scan_e2e(scan_pred: dict):
    section("10a. eval_scan() end-to-end")

    # Dùng prediction page 1 làm cả GT lẫn pred → score phải tốt
    page1 = scan_pred["pages"][0]
    gt_page = {"page_num": 1, "full_text": page1["full_text"]}
    result = eval_scan(gt_page, pred_text=page1["full_text"], doc_id="scan_en_001")

    check("eval_scan trả về dict", isinstance(result, dict))
    check("có key cer, wer, nwer", all(k in result for k in ["cer","wer","nwer"]))
    check("self-eval cer=0", result["cer"] == 0.0, f"cer={result['cer']}")
    check("self-eval wer=0", result["wer"] == 0.0, f"wer={result['wer']}")
    check("char_precision/recall keys có mặt",
          "char_precision" in result and "char_recall" in result)

    # Với pred rỗng → cer cao
    r2 = eval_scan(gt_page, pred_text="", doc_id="scan_en_001")
    check("cer cao khi pred rỗng", r2["cer"] > 0.5, f"cer={r2['cer']}")


def test_eval_table_e2e(table_pred: dict):
    section("10b. eval_table() end-to-end")
    if not table_pred["pages"]:
        check("skip — không có table pages", True, "SKIP")
        return

    page = table_pred["pages"][0]
    gt_page = {"page_num": page["page_num"], "tables": page["tables"]}

    result = eval_table(gt_page, pred_tables=page["tables"], doc_id="scan_en_001")

    check("eval_table trả về dict", isinstance(result, dict))
    check("có key avg_teds", "avg_teds" in result)
    check("self-eval avg_teds=1.0", result["avg_teds"] == 1.0, f"teds={result['avg_teds']}")
    check("có key tables", "tables" in result and len(result["tables"]) > 0)

    # Với pred rỗng → teds thấp
    r2 = eval_table(gt_page, pred_tables=[], doc_id="scan_en_001")
    check("avg_teds thấp khi pred rỗng", r2["avg_teds"] < 1.0, f"teds={r2['avg_teds']}")


def test_eval_text_layer_e2e(text_layer_pred: dict):
    section("10c. eval_text_layer() end-to-end")
    page = text_layer_pred["pages"][0]
    gt_page = {
        "page_num": page["page_num"],
        "full_text": page["full_text"],
        "blocks": page["blocks"],
    }

    result = eval_text_layer(
        gt_page,
        pred_blocks=page["blocks"],
        pred_full_text=page["full_text"],
        doc_id="scan_en_001",
    )

    check("eval_text_layer trả về dict", isinstance(result, dict))
    check("có key cer + mean_iou", all(k in result for k in ["cer","mean_iou"]))
    check("self-eval cer=0", result["cer"] == 0.0, f"cer={result['cer']}")
    check("mean_iou trong [0,1]", 0.0 <= result["mean_iou"] <= 1.0,
          f"iou={result['mean_iou']}")


# ─────────────────────────────────────────────────────────────────
# 11. Split eval: 1 file → 2 score riêng biệt
# ─────────────────────────────────────────────────────────────────

def test_split_eval_e2e(split: dict):
    section("11. Split eval: text score + table score từ 1 file")

    text_pred  = split["text"]
    table_pred = split["table"]

    # Text eval
    p = text_pred["pages"][0]
    gt_scan = {"page_num": p["page_num"], "full_text": p["full_text"]}
    scan_result = eval_scan(gt_scan, pred_text=p["full_text"], doc_id="split_test")
    check("split text → eval_scan cer=0", scan_result["cer"] == 0.0,
          f"cer={scan_result['cer']}")

    # Table eval
    if table_pred["pages"]:
        p2 = table_pred["pages"][0]
        gt_tbl = {"page_num": p2["page_num"], "tables": p2["tables"]}
        tbl_result = eval_table(gt_tbl, pred_tables=p2["tables"], doc_id="split_test")
        check("split table → eval_table teds=1.0", tbl_result["avg_teds"] == 1.0,
              f"teds={tbl_result['avg_teds']}")

    # Đảm bảo page_num trong text và table không bị nhầm
    text_pages  = {p["page_num"] for p in text_pred["pages"]}
    table_pages = {p["page_num"] for p in table_pred["pages"]}
    check("text pages và table pages có page_num hợp lệ",
          all(n >= 1 for n in text_pages | table_pages),
          f"text={sorted(text_pages)} table={sorted(table_pages)}")


# ─────────────────────────────────────────────────────────────────
# 12. normalize
# ─────────────────────────────────────────────────────────────────

def test_normalize():
    section("12. normalize_for_text_benchmark()")

    check("strip newlines", normalize_for_text_benchmark("a\nb\nc") == "a b c")
    check("collapse spaces", normalize_for_text_benchmark("a   b") == "a b")
    check("strip edges", normalize_for_text_benchmark("  hello  ") == "hello")
    check("empty string", normalize_for_text_benchmark("") == "")
    check("NFC unicode",
          normalize_for_text_benchmark("e\u0301") == normalize_for_text_benchmark("é"))


# ─────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────

def run_all():
    print(f"\n{'='*60}")
    print(f"  OCR Benchmark — Full Pipeline Test")
    print(f"  Source: {FULL_RESPONSE.relative_to(ROOT)}")
    print(f"{'='*60}")

    if not FULL_RESPONSE.exists():
        print(f"\n❌ File không tồn tại: {FULL_RESPONSE}")
        print("   Chạy marker_convert trước để tạo full_response.json")
        sys.exit(1)

    mj = load_marker_json()

    # Unit tests (không cần file)
    test_block_text()
    test_html_to_cells()
    test_metrics_cer()
    test_metrics_wer()
    test_metrics_teds()
    test_normalize()

    # Integration tests (dùng full_response.json)
    blocks      = test_block_extraction(mj)
    scan_pred   = test_build_scan(mj)
    table_pred  = test_build_table(mj)
    tl_pred     = test_build_text_layer(mj)
    split       = test_build_split(mj)
    test_dispatch(mj)

    # E2E eval
    test_eval_scan_e2e(scan_pred)
    test_eval_table_e2e(table_pred)
    test_eval_text_layer_e2e(tl_pred)
    test_split_eval_e2e(split)

    # Summary
    passed = sum(1 for r in _results if r["status"] == PASS)
    failed = sum(1 for r in _results if r["status"] == FAIL)
    total  = len(_results)

    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ: {passed}/{total} passed  |  {failed} failed")
    print(f"{'='*60}")

    if failed:
        print("\n  Failed tests:")
        for r in _results:
            if r["status"] == FAIL:
                print(f"    ❌ {r['name']}" + (f" — {r['detail']}" if r['detail'] else ""))
        sys.exit(1)
    else:
        print("\n  Tất cả tests passed! ✅\n")


if __name__ == "__main__":
    run_all()
