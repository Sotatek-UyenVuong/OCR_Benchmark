#!/usr/bin/env python3
"""
reorganize_raw.py
-----------------
Di chuyển file từ "data sota/" sang cấu trúc raw/ chuẩn cho runner.py.

Cấu trúc đích:
    raw/
        scan/vi/       ← UC01
        scan/en/       ← UC02
        scan/ja/       ← UC03
        table/vi/      ← UC04
        table/en/      ← UC05
        table/ja/      ← UC06
        text_layer/vi/ ← UC07
        text_layer/en/ ← UC08
        text_layer/ja/ ← UC09

Quy tắc rename file:
    - Lowercase
    - Thay khoảng trắng và ký tự đặc biệt bằng _
    - Bỏ dấu tiếng Việt / ký tự Unicode không phải ASCII
    - Đảm bảo không trùng tên (thêm _2, _3, ...)

Usage:
    python scripts/reorganize_raw.py                    # dry-run (chỉ in, không làm gì)
    python scripts/reorganize_raw.py --execute          # thực sự di chuyển file
    python scripts/reorganize_raw.py --execute --copy   # copy thay vì move (giữ nguyên gốc)
"""

from __future__ import annotations

import argparse
import re
import shutil
import unicodedata
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Mapping: folder nguồn → subfolder đích (relative to raw/)
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent  # OCR_Benchmark/
SRC_ROOT = BASE_DIR / "ocr_benchmark" / "data sota"
DST_ROOT = BASE_DIR / "raw"

FOLDER_MAP: dict[str, str] = {
    "vi scan":      "scan/vi",
    "en scan":      "scan/en",
    "ja_scan":      "scan/ja",
    "vi_table":     "table/vi",
    "en table":     "table/en",
    "ja_table":     "table/ja",
    "vi_text":      "text_layer/vi",
    "en text_layer": "text_layer/en",
    "ja_text":      "text_layer/ja",
}

# UC prefix để tạo doc_id có nghĩa
UC_PREFIX: dict[str, str] = {
    "scan/vi":       "scan_vi",
    "scan/en":       "scan_en",
    "scan/ja":       "scan_ja",
    "table/vi":      "table_vi",
    "table/en":      "table_en",
    "table/ja":      "table_ja",
    "text_layer/vi": "textlayer_vi",
    "text_layer/en": "textlayer_en",
    "text_layer/ja": "textlayer_ja",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    """
    Chuyển tên file về dạng ASCII slug an toàn:
      1. Map ký tự đặc biệt tiếng Việt / Latin mở rộng về ASCII gần nhất
      2. NFD decompose → bỏ combining diacritics còn lại
      3. Lowercase
      4. Thay mọi ký tự không phải [a-z0-9._-] bằng _
      5. Gộp nhiều _ liên tiếp thành 1
      6. Bỏ _ đầu/cuối stem
      7. Fix extension bị sai (ví dụ .pd_ → .pdf nếu thực chất là PDF)
    """
    # Bảng map thủ công cho ký tự không có trong NFD ASCII
    # (chủ yếu là đ/Đ tiếng Việt và ký tự Latin đặc biệt)
    EXTRA_MAP = str.maketrans({
        "đ": "d", "Đ": "D",
        "ł": "l", "Ł": "L",
        "ø": "o", "Ø": "O",
        "ß": "ss",
        "æ": "ae", "Æ": "AE",
        "œ": "oe", "Œ": "OE",
        "þ": "th", "Þ": "TH",
        "ð": "d",  "Ð": "D",
    })
    name = name.translate(EXTRA_MAP)

    # NFD decompose: "ồ" → "o" + combining grave + combining hook
    normalized = unicodedata.normalize("NFD", name)
    # Giữ lại ASCII
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    # Lowercase
    lowered = ascii_only.lower()

    # Tách stem và extension — dùng rsplit để xử lý "file.pd_test2.pdf" → stem="file.pd_test2", ext=".pdf"
    # Quy tắc: chỉ coi đuôi cuối cùng là extension nếu nó là định dạng tài liệu đã biết
    known_exts = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".png", ".jpg", ".jpeg", ".webp"}
    p = Path(lowered)
    if p.suffix in known_exts:
        stem = p.stem
        ext = p.suffix
    else:
        # Fallback: không có extension hợp lệ → toàn bộ là stem, không có ext
        stem = lowered
        ext = ""

    # Thay ký tự đặc biệt trong stem
    slug = re.sub(r"[^a-z0-9_-]", "_", stem)   # dấu chấm trong stem → _
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        slug = "doc"
    return slug + ext


def _unique_dst(dst_dir: Path, name: str, occupied: set[Path]) -> Path:
    """Trả về Path không trùng trong dst_dir, thêm _2/_3/... nếu cần."""
    p = Path(name)
    stem, ext = p.stem, p.suffix
    candidate = dst_dir / name
    counter = 2
    while candidate in occupied or candidate.exists():
        candidate = dst_dir / f"{stem}_{counter}{ext}"
        counter += 1
    occupied.add(candidate)
    return candidate


# ──────────────────────────────────────────────────────────────────────────────
# Main logic
# ──────────────────────────────────────────────────────────────────────────────

def plan_moves() -> list[dict]:
    """
    Tính toán danh sách các thao tác cần làm.
    Trả về list of {src, dst, renamed}.
    """
    moves: list[dict] = []
    occupied_per_dir: dict[str, set[Path]] = {}

    for src_folder_name, dst_subfolder in FOLDER_MAP.items():
        src_dir = SRC_ROOT / src_folder_name
        dst_dir = DST_ROOT / dst_subfolder

        if not src_dir.exists():
            print(f"  [WARN] Folder nguồn không tồn tại: {src_dir}")
            continue

        occupied = occupied_per_dir.setdefault(dst_subfolder, set())

        for src_file in sorted(src_dir.iterdir()):
            if not src_file.is_file():
                continue

            new_name = _slugify(src_file.name)
            dst_file = _unique_dst(dst_dir, new_name, occupied)

            moves.append({
                "src": src_file,
                "dst": dst_file,
                "renamed": src_file.name != dst_file.name,
                "dst_subfolder": dst_subfolder,
            })

    return moves


def print_plan(moves: list[dict]) -> None:
    """In kế hoạch di chuyển theo nhóm UC."""
    current_uc = None
    for m in moves:
        if m["dst_subfolder"] != current_uc:
            current_uc = m["dst_subfolder"]
            prefix = UC_PREFIX.get(current_uc, current_uc)
            print(f"\n  📁  raw/{current_uc}/   [{prefix}_NNN]")
            print(f"  {'─'*56}")

        rename_tag = f"  → {m['dst'].name}" if m["renamed"] else ""
        print(f"    {m['src'].name}{rename_tag}")


def execute_moves(moves: list[dict], copy: bool = False) -> None:
    """Thực hiện di chuyển / copy."""
    action = shutil.copy2 if copy else shutil.move
    verb = "Copied" if copy else "Moved"
    errors = 0

    for m in moves:
        m["dst"].parent.mkdir(parents=True, exist_ok=True)
        try:
            action(str(m["src"]), str(m["dst"]))
            status = "✅"
        except Exception as e:
            status = "❌"
            errors += 1
            print(f"  {status} ERROR: {m['src'].name} → {e}")
            continue

        rename_tag = f" → {m['dst'].name}" if m["renamed"] else ""
        print(f"  {status} {verb}: {m['src'].name}{rename_tag}")
        print(f"      └─ {m['dst']}")

    print(f"\n  {'─'*60}")
    print(f"  Tổng: {len(moves)} file | Lỗi: {errors}")


def print_final_tree() -> None:
    """In cây thư mục raw/ sau khi hoàn thành."""
    if not DST_ROOT.exists():
        return
    print(f"\n  📂 {DST_ROOT}")
    for uc_type in ["scan", "table", "text_layer"]:
        for lang in ["vi", "en", "ja"]:
            d = DST_ROOT / uc_type / lang
            if d.exists():
                files = sorted(d.glob("*"))
                print(f"    ├── {uc_type}/{lang}/  ({len(files)} files)")
                for f in files:
                    print(f"    │     {f.name}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reorganize 'data sota/' → 'raw/' theo cấu trúc benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Thực sự di chuyển file (mặc định chỉ dry-run)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy thay vì move (giữ nguyên folder gốc 'data sota/')",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  OCR Benchmark — Reorganize raw/")
    print(f"{'='*60}")
    print(f"  Nguồn : {SRC_ROOT}")
    print(f"  Đích   : {DST_ROOT}")
    print(f"  Mode   : {'EXECUTE (' + ('copy' if args.copy else 'move') + ')' if args.execute else 'DRY-RUN (không làm gì)'}")
    print(f"{'='*60}")

    moves = plan_moves()

    if not moves:
        print("\n  Không tìm thấy file nào để di chuyển.")
        return

    print(f"\n  Kế hoạch ({len(moves)} file):")
    print_plan(moves)

    if not args.execute:
        print(f"\n{'='*60}")
        print(f"  ⚠️  DRY-RUN — chưa có gì thay đổi.")
        print(f"  Chạy lại với --execute để thực hiện:")
        print(f"    python scripts/reorganize_raw.py --execute          # move")
        print(f"    python scripts/reorganize_raw.py --execute --copy   # copy (giữ gốc)")
        print(f"{'='*60}\n")
        return

    print(f"\n  Bắt đầu {'copy' if args.copy else 'di chuyển'}...")
    print(f"  {'─'*60}")
    execute_moves(moves, copy=args.copy)
    print_final_tree()
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
