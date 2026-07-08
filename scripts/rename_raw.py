#!/usr/bin/env python3
"""
rename_raw.py
-------------
Đổi tên file trong raw/ về format chuẩn:

    {uc_type}_{lang}_{seq:03d}.pdf

Ví dụ:
    raw/scan/vi/001.pdf    → scan_vi_001.pdf
    raw/table/en/001.pdf   → table_en_001.pdf
    raw/text_layer/ja/001  → textlayer_ja_001.pdf

Seq được đánh số liên tiếp theo alphabetical order của tên file hiện tại
để kết quả deterministic.

Usage:
    python scripts/rename_raw.py              # dry-run
    python scripts/rename_raw.py --execute    # thực sự rename
"""

from __future__ import annotations

import argparse
from pathlib import Path

RAW_ROOT = Path(__file__).parent.parent / "raw"


def _uc_prefix(uc_type: str, lang: str) -> str:
    """raw/text_layer/vi → 'textlayer_vi'  |  raw/scan/en → 'scan_en'"""
    return f"{uc_type.replace('_', '')}_{lang}"


def plan_renames() -> list[dict]:
    """
    Duyệt raw/<uc_type>/<lang>/ và lập danh sách rename.
    Trả về list of {src: Path, dst: Path}.
    """
    renames: list[dict] = []

    for uc_type_dir in sorted(RAW_ROOT.iterdir()):
        if not uc_type_dir.is_dir():
            continue
        uc_type = uc_type_dir.name   # "scan" | "table" | "text_layer"

        for lang_dir in sorted(uc_type_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            lang = lang_dir.name     # "vi" | "en" | "ja"
            prefix = _uc_prefix(uc_type, lang)

            files = sorted(f for f in lang_dir.iterdir() if f.is_file())

            for seq, src in enumerate(files, start=1):
                new_name = f"{prefix}_{seq:03d}{src.suffix.lower()}"
                dst = lang_dir / new_name
                renames.append({"src": src, "dst": dst, "changed": src.name != new_name})

    return renames


def print_plan(renames: list[dict]) -> None:
    current_dir: Path | None = None
    for r in renames:
        d = r["src"].parent
        if d != current_dir:
            current_dir = d
            rel = d.relative_to(RAW_ROOT)
            print(f"\n  📁  raw/{rel}/")
            print(f"  {'─'*54}")
        tag = "  ✏️ " if r["changed"] else "  ✔ "
        if r["changed"]:
            print(f"{tag}{r['src'].name}  →  {r['dst'].name}")
        else:
            print(f"{tag}{r['src'].name}  (không đổi)")


def execute_renames(renames: list[dict]) -> None:
    errors = 0
    changed = 0
    for r in renames:
        if not r["changed"]:
            continue
        try:
            r["src"].rename(r["dst"])
            print(f"  ✅ {r['src'].name}  →  {r['dst'].name}")
            changed += 1
        except Exception as e:
            print(f"  ❌ {r['src'].name}: {e}")
            errors += 1

    print(f"\n  {'─'*56}")
    print(f"  Đổi tên: {changed} | Giữ nguyên: {len(renames)-changed-errors} | Lỗi: {errors}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename raw/ files về format chuẩn.")
    parser.add_argument("--execute", action="store_true", help="Thực sự rename (mặc định dry-run)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Rename raw/ → {{uc_type}}_{{lang}}_{{seq:03d}}.pdf")
    print(f"{'='*60}")
    print(f"  Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")

    renames = plan_renames()

    if not renames:
        print("\n  Không tìm thấy file nào.")
        return

    print_plan(renames)

    if not args.execute:
        print(f"\n{'='*60}")
        print(f"  ⚠️  DRY-RUN — chưa có gì thay đổi.")
        print(f"  Chạy lại với --execute để rename thật:")
        print(f"    python scripts/rename_raw.py --execute")
        print(f"{'='*60}\n")
        return

    print(f"\n  Bắt đầu rename...")
    print(f"  {'─'*56}")
    execute_renames(renames)
    print()


if __name__ == "__main__":
    main()
