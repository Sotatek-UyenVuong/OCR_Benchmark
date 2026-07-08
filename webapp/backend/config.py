"""
config.py
---------
Tập trung tất cả path config cho webapp.

Mặc định dùng project root (cùng thư mục với pyproject.toml).
Override bằng env var khi deploy:

    DATA_ROOT=/data/ocr_benchmark uv run uvicorn ...

Cấu trúc thư mục mong đợi (relative to DATA_ROOT):
    raw/               — PDF gốc + marker output
    ground_truth/      — GT JSON đã review
    predictions/       — Prediction JSON của từng model
    benchmark_results/ — Kết quả eval CSV/JSON
"""

from __future__ import annotations
import os
from pathlib import Path

# ── Project root: thư mục chứa pyproject.toml ───────────────────
# Khi chạy từ webapp/backend/ → parents[2] = project root
# Khi chạy từ project root   → Path(__file__).parents[2] vẫn đúng
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]

# Cho phép override bằng env var DATA_ROOT khi deploy
PROJECT_ROOT = Path(os.environ.get("DATA_ROOT", str(_DEFAULT_ROOT))).resolve()

# ── Data directories ─────────────────────────────────────────────
RAW_ROOT       = PROJECT_ROOT / "raw"
GT_ROOT        = PROJECT_ROOT / "ground_truth"
PRED_ROOT      = PROJECT_ROOT / "predictions"
RESULT_ROOT    = PROJECT_ROOT / "benchmark_results"
MARKER_SUBDIR  = "marker_output"
