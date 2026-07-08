# OCR Benchmark

## Quickstart

### 1. Clone & cài dependencies

```bash
git clone https://github.com/Sotatek-UyenVuong/OCR_Benchmark.git
cd OCR_Benchmark

# Cài uv (nếu chưa có)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Cài dependencies
uv sync
```

### 2. Cấu hình API keys

Tạo file `.env` ở root:

```env
MARKER_API_KEY=your_marker_api_key      # https://www.datalab.to
OPENAI_API_KEY=your_openai_key          # tuỳ chọn
OPEN_ROUTER=your_openrouter_key         # tuỳ chọn
```

### 3. Chuẩn bị dataset

Đặt PDF vào đúng cấu trúc thư mục:

```
raw/
├── scan/vi/        scan_vi_001.pdf, scan_vi_002.pdf, ...
├── scan/en/        scan_en_001.pdf, ...
├── scan/ja/        ...
├── table/vi/       table_vi_001.pdf, ...
├── table/en/       ...
├── table/ja/       ...
├── text_layer/vi/  textlayer_vi_001.pdf, ...
├── text_layer/en/  ...
└── text_layer/ja/  ...
```

> Nếu có dataset sẵn với tên file không chuẩn, dùng script tổ chức lại:
> ```bash
> # Copy từ "data sota/" → raw/ và đổi tên về format chuẩn
> python3 scripts/reorganize_raw.py --execute --copy
> python3 scripts/rename_raw.py --execute
> ```

### 4. Chạy Web App

```bash
uv run uvicorn webapp.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Mở trình duyệt: **http://localhost:8000**

---

## Luồng sử dụng Web App

```
[1] Select File   — chọn PDF từ danh sách
       ↓
[2] Run OCR       — chọn model (Marker), click "Run OCR"
                    (idempotent: nếu đã chạy rồi sẽ skip)
       ↓
[3] Review GT     — PDF bên trái với bbox highlight
                    Editor bên phải (📝 Text / 📊 Table)
                    • Click vùng trên PDF → jump tới block tương ứng
                    • Sửa text/HTML bảng trực tiếp
                    • Ctrl+S hoặc "Save GT & Evaluate"
       ↓
[4] Scores        — CER, WER, nWER (text) + TEDS (table)
                    Breakdown per page
```

---

## Chạy CLI (không dùng web)

### Chạy OCR cho 1 file

```bash
uv run python -m ocr_benchmark.ocr_model.marker_convert \
  raw/scan/en/scan_en_001.pdf \
  --uc split \
  --doc-id scan_en_001 \
  -l en \
  -m balanced \
  -o raw/scan/en/marker_output
```

`--uc` options: `scan` | `table` | `text_layer` | `split` (tách cả text lẫn table)

### Chạy batch benchmark

```bash
uv run python -m ocr_benchmark.runner \
  --gt_dir ground_truth/ \
  --pred_dir predictions/marker/ \
  --model marker \
  --output_dir benchmark_results/
```

### Chạy tests

```bash
uv run pytest tests/ -v
```

---

## Use Cases

| UC | Tên | Mô tả |
|---|---|---|
| UC01 | scan_vi | Tài liệu scan tiếng Việt |
| UC02 | scan_en | Tài liệu scan tiếng Anh |
| UC03 | scan_ja | Tài liệu scan tiếng Nhật |
| UC04 | table_vi | Bảng biểu tiếng Việt |
| UC05 | table_en | Bảng biểu tiếng Anh |
| UC06 | table_ja | Bảng biểu tiếng Nhật |
| UC07 | text_layer_vi | PDF có text layer sẵn tiếng Việt |
| UC08 | text_layer_en | PDF có text layer sẵn tiếng Anh |
| UC09 | text_layer_ja | PDF có text layer sẵn tiếng Nhật |

3 nhóm chính × 3 ngôn ngữ (vi / en / ja):

- **Scan** — ảnh scan, không có text layer, model phải nhận dạng từ hình ảnh
- **Table** — tài liệu chứa bảng biểu, cần extract đúng cấu trúc hàng/cột
- **Text layer** — PDF có text layer sẵn, cần extract đúng nội dung và thứ tự đọc

---

## Bước 1: Metric theo từng UC

| UC | Loại | Metric chính | Metric phụ |
|---|---|---|---|
| Scan (vi/en/ja) | Text recognition | CER (Character Error Rate) | WER, Precision/Recall ký tự |
| Table (vi/en/ja) | Cấu trúc bảng | TEDS (Tree Edit Distance Similarity) | Cell accuracy, structure F1 |
| Text Layer (vi/en/ja) | PDF có text layer sẵn | CER + Layout Overlap (IoU) | — |

---

## Giải thích các độ đo

### 1. CER — Character Error Rate

Đo tỉ lệ ký tự bị nhận dạng sai so với ground truth.

```
CER = (S + D + I) / N

S = số ký tự bị thay thế sai (substitution)
D = số ký tự bị mất (deletion)
I = số ký tự thừa (insertion)
N = tổng số ký tự trong ground truth
```

**Ví dụ:**

```
Ground truth:  "Hợp đồng lao động"
Model output:  "Hợp đông lao đông"

S = 2 (ộ→ô ở "đồng" bị mất dấu, xảy ra 2 lần)
N = 18
CER = 2/18 = 11.1%
```

**Tại sao dùng CER thay WER cho vi/ja:**
- Tiếng Việt: dấu thanh là ký tự riêng, sai 1 dấu = 1 ký tự sai, CER đo chính xác hơn
- Tiếng Nhật: không có space giữa từ, WER không áp dụng được
- Mục tiêu thực tế: < 2% cho vi, < 1% cho en

#### Output schema cho CER

```json
{
  "doc_id": "scan_vi_001",
  "page_num": 1,
  "cer": 0.111,
  "cer_detail": {
    "substitutions": 2,
    "deletions": 0,
    "insertions": 0,
    "total_chars_gt": 18
  },
  "ground_truth": "Hợp đồng lao động",
  "prediction": "Hợp đông lao đông",
  "char_alignment": [
    {"gt": "ồ", "pred": "ô", "type": "substitution"},
    {"gt": "ồ", "pred": "ô", "type": "substitution"}
  ]
}
```

- `ground_truth` + `prediction` side-by-side → reviewer thấy ngay lỗi ở đâu
- S/D/I riêng lẻ giúp phân tích pattern: model hay bỏ sót (D cao), nhận sai (S cao), hay thêm thừa (I cao)
- `char_alignment` hữu ích khi debug sâu — thấy ngay model yếu ở dấu thanh, chữ Hán-Nôm, hay ký tự đặc biệt

---

### 2. WER — Word Error Rate (metric phụ)

Tương tự CER nhưng đơn vị là từ.

```
WER = (S + D + I) / N

S = số từ bị thay thế sai
D = số từ bị xoá (có trong GT, không có trong output)
I = số từ bị chèn thừa
N = tổng số từ trong Ground Truth
```

**Ví dụ:**

```
Ground truth:  "Hợp đồng lao động số 001"  → 5 từ
Model output:  "Hợp đồng lao đông số 001"  → 1 từ sai
WER = 1/5 = 20%
```

- Giá trị: 0 = hoàn hảo, càng thấp càng tốt
- Có thể > 1 nếu model chèn nhiều từ thừa
- Thư viện: `jiwer` — hàm `wer()`
- WER phạt nặng hơn CER — sai 1 ký tự = cả từ bị tính sai. Dùng làm metric phụ để cross-check, không làm metric chính cho vi/ja.

---

### 2b. nWER — Normalized Word Error Rate

WER sau khi bỏ phân biệt hoa/thường và dấu câu.

```
nWER = WER( lower(ref), lower(hyp) )

lower(x): chuyển lowercase, xóa toàn bộ dấu câu, gộp khoảng trắng
```

Đo độ đúng nội dung thuần túy, không tính hoa/thường và punctuation. Hữu ích để tách biệt lỗi nhận dạng nội dung vs lỗi format.

---

### 3. Precision / Recall ký tự (metric phụ)

```
Precision = ký tự đúng trong output / tổng ký tự output
Recall    = ký tự đúng trong output / tổng ký tự ground truth
```

**Ví dụ:**

```
Ground truth:  "abc"   (3 ký tự)
Model output:  "abcd"  (4 ký tự, thêm 'd' thừa)

Precision = 3/4 = 75%   ← output có 25% ký tự thừa/sai
Recall    = 3/3 = 100%  ← không mất ký tự nào
```

Dùng khi muốn biết model thiên về bỏ sót hay thêm thừa, hữu ích để debug model cụ thể.

---

### 4. TEDS — Tree Edit Distance Similarity

Đo độ chính xác của cấu trúc bảng — vừa đo text vừa đo cấu trúc (hàng, cột, merged cell).

```
TEDS = 1 - (edit_distance(output_tree, gt_tree) / max(|output_tree|, |gt_tree|))
```

Giá trị từ 0 đến 1, càng gần 1 càng tốt.

Tại sao không dùng CER cho bảng: CER chỉ đo text, không phát hiện được merged cell sai, hàng/cột bị đảo. TEDS bắt được cả hai.

#### Output schema cho TEDS

```json
{
  "doc_id": "table_vi_001",
  "page_num": 1,
  "table_id": 1,
  "teds": 0.82,
  "teds_detail": {
    "edit_distance": 5,
    "gt_tree_size": 24,
    "pred_tree_size": 22
  },
  "ground_truth_html": "<table><tr><th>Tên</th><th>Tuổi</th></tr><tr><td>Nguyễn A</td><td>30</td></tr></table>",
  "prediction_html": "<table><tr><td>Tên</td><td>Tuổi</td></tr><tr><td>Nguyễn A</td><td>30</td></tr></table>",
  "cell_diff": [
    {
      "row": 0, "col": 0,
      "gt_text": "Tên", "pred_text": "Tên",
      "gt_is_header": true, "pred_is_header": false,
      "match": false, "issue": "header_mismatch"
    }
  ]
}
```

**Tại sao kèm HTML cả hai phía:** TEDS score 0.82 không nói lên được lỗi do text sai hay cấu trúc sai. Có HTML là paste thẳng vào browser so sánh trực quan, hoặc dùng diff tool là thấy ngay.

**`cell_diff` phân loại lỗi:**

| issue | Ý nghĩa |
|---|---|
| `header_mismatch` | `<th>` vs `<td>` bị nhầm |
| `span_mismatch` | merged cell bị tách hoặc ngược lại |
| `text_mismatch` | đúng cấu trúc nhưng sai text |
| `missing_cell` | cell bị mất so với GT |
| `extra_cell` | cell thừa không có trong GT |

---

### 5. IoU — Layout Overlap (Intersection over Union)

Đo bounding box của text block mà model detect có khớp với ground truth không.

```
IoU = Diện tích giao nhau / Diện tích hợp nhau
```

Dùng chủ yếu cho text layer PDF — vì PDF có sẵn tọa độ text, cần check model có detect đúng vùng không, đặc biệt với layout phức tạp (2 cột, header/footer, sidebar).

#### Output schema cho IoU

```json
{
  "doc_id": "textlayer_vi_001",
  "page_num": 1,
  "mean_iou": 0.76,
  "blocks": [
    {
      "block_id": 1,
      "iou": 0.91,
      "gt_bbox": [0.10, 0.05, 0.90, 0.15],
      "pred_bbox": [0.11, 0.06, 0.89, 0.14],
      "gt_text": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
      "pred_text": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
      "match_status": "matched"
    },
    {
      "block_id": 2,
      "iou": 0.21,
      "gt_bbox": [0.10, 0.20, 0.45, 0.40],
      "pred_bbox": [0.10, 0.20, 0.90, 0.40],
      "gt_text": "Điều 1. Phạm vi...",
      "pred_text": "Điều 1. Phạm vi... Điều 2. Đối tượng...",
      "match_status": "over_merged",
      "issue": "two_columns_merged"
    }
  ]
}
```

**`match_status` các giá trị:**

| Giá trị | Ý nghĩa |
|---|---|
| `matched` | IoU ≥ threshold (thường 0.5) |
| `over_merged` | model gộp 2+ block thành 1 |
| `split` | model tách 1 block thành nhiều |
| `missed` | GT block không có prediction nào khớp |
| `extra` | prediction không khớp với GT block nào |

**Tại sao kèm text vào IoU output:** IoU thấp có 2 nguyên nhân khác nhau — bbox lệch nhưng text đúng (lỗi layout detection) vs bbox lệch *và* text sai (lỗi reading order). Kèm text vào phân biệt được ngay, tránh debug nhầm hướng.

---

### 6. punct_mIoU — Punctuation mean IoU

Đo độ khớp dấu câu giữa GT và output sau word alignment.

**Tập dấu câu:** `, . ? ! ; :`

**Cách tính:**
1. Tách ref và hyp thành danh sách từ (regex `\S+`)
2. Căn chỉnh từ bằng `jiwer.process_words()`
3. Với mỗi loại dấu, gắn nhãn binary cho từng vị trí aligned: có dấu = 1, không = 0
4. Tính `IoU = TP / (TP + FP + FN)` cho từng loại dấu
5. `punct_miou` = trung bình IoU các loại dấu có xuất hiện trong ref hoặc hyp

Giá trị: 0–1, càng cao càng tốt.

---

### 7. cap_mIoU — Capitalization mean IoU

Đo độ khớp viết hoa chữ cái đầu từ (chỉ tính trên từ có nội dung không rỗng).

**Cách tính:**
1. Căn chỉnh từ ref ↔ hyp (word alignment)
2. Với mỗi cặp từ aligned: `flag = 1` nếu chữ cái đầu viết hoa, ngược lại = 0
3. `cap_miou = TP / (TP + FP + FN)`

Giá trị: 0–1, càng cao càng tốt.

---

### 8. PCS — Punctuation & Capitalization Score

Trung bình cộng của punct_mIoU và cap_mIoU.

```
PCS = (punct_miou + cap_miou) / 2
```

Giá trị: 0–1, càng cao càng tốt.

---

### Luồng xử lý evaluation

```
[Step 1] Đọc GT (.md) và output OCR (.txt) cho từng trang PDF
    ↓
[Step 2] Chuẩn hóa text
         ref = normalize(GT)
         hyp = normalize(output)
         → gọi normalize_for_text_benchmark() trong plain_text_gt.py
    ↓
[Step 3] Tính metric từng trang
    ↓
[Step 4] Gom trung bình theo UC
         avg_metric_UC = (1/n) × Σ metric_i
         n = số trang successful trong UC
    ↓
[Step 5] Ghi CSV → benchmark_results/benchmark_table_page_screenshot.csv
```

**Dòng Average overall:**
```
avg_metric_overall = (1/N) × Σ metric_j
N = tổng số trang successful across all UC
```

---

### Tóm tắt tất cả metrics

| Cột CSV | Công thức | Thư viện / Hàm | Tốt khi |
|---|---|---|---|
| `avg_wer` | WER = (S+D+I)/N | `jiwer.wer()` | ↓ thấp |
| `avg_nwer` | WER sau normalize hoa/thường + dấu câu | `normalize_for_nwer()` + `jiwer.wer()` | ↓ thấp |
| `avg_punct_miou` | mean IoU dấu câu | `punct_class_miou()` | ↑ cao |
| `avg_cap_miou` | IoU viết hoa chữ đầu từ | `cap_miou()` | ↑ cao |
| `avg_pcs` | (punct_miou + cap_miou) / 2 | `compute_sample()` | ↑ cao |
| `cer` | CER = (S+D+I)/N (ký tự) | edit distance | ↓ thấp |
| `teds` | Tree Edit Distance Similarity | — | ↑ cao |
| `mean_iou` | Intersection over Union (bbox) | — | ↑ cao |

---

### Tóm tắt output fields theo metric

| Field | CER | TEDS | IoU | Mục đích |
|---|---|---|---|---|
| Score tổng | `cer` | `teds` | `mean_iou` | Ranking |
| Score breakdown | S, D, I, N | edit_distance, tree_size | per-block iou | Hiểu mức độ lỗi |
| GT representation | text | HTML | bbox + text | So sánh trực quan |
| Pred representation | text | HTML | bbox + text | So sánh trực quan |
| Diff / issue label | `char_alignment` | `cell_diff` | `match_status` | Phân loại lỗi nhanh |

---

## Bước 2: Ground Truth schema theo từng UC

### UC01-03: Scan documents

```json
{
  "doc_id": "scan_vi_001",
  "language": "vi",
  "pages": [
    {
      "page_num": 1,
      "full_text": "Nội dung toàn trang...",
      "lines": [
        {"line_id": 1, "text": "Dòng đầu tiên", "bbox": [x1, y1, x2, y2]},
        {"line_id": 2, "text": "Dòng thứ hai", "bbox": [x1, y1, x2, y2]}
      ]
    }
  ]
}
```

> Cần per-page full text, không cần bbox nếu chỉ đo CER.

### UC04-06: Table documents

```json
{
  "doc_id": "table_vi_001",
  "pages": [
    {
      "page_num": 1,
      "tables": [
        {
          "table_id": 1,
          "html": "<table><tr><td>Tên</td><td>Tuổi</td></tr>...</table>",
          "cells": [
            {
              "row": 0, "col": 0,
              "rowspan": 1, "colspan": 1,
              "text": "Tên",
              "is_header": true
            }
          ]
        }
      ]
    }
  ]
}
```

> Cần HTML representation của bảng để tính TEDS, plus cell-level text để tính cell accuracy.

### UC07-09: Text Layer PDF

```json
{
  "doc_id": "textlayer_vi_001",
  "pages": [
    {
      "page_num": 1,
      "blocks": [
        {
          "block_id": 1,
          "bbox": [x1, y1, x2, y2],
          "text": "Nội dung block"
        }
      ],
      "full_text": "Toàn bộ text theo reading order..."
    }
  ]
}
```

> bbox normalize 0-1 theo page size.

---

## Bước 3: Quy trình tạo Ground Truth

```
Tài liệu gốc
    ↓
[Step 1] Chạy qua model tốt nhất (Marker hoặc Gemini Pro)
    ↓
[Step 2] Xuất ra draft ground truth (text / HTML table / bbox)
    ↓
[Step 3] Human reviewer verify + correct
    - UC01-03: Đọc text, sửa lỗi nhận dạng sai
    - UC04-06: Check cấu trúc bảng, merge cell, header
    - UC07-09: Check reading order, bbox có đúng vùng không
    ↓
[Step 4] Lưu vào JSON schema chuẩn
    ↓
[Step 5] Cross-check: chạy 1 model khác so sánh,
         page nào score thấp bất thường → review lại
```

---

## Bước 4: Cấu trúc Dataset

```
ocr_benchmark/
├── raw/                        # File gốc (PDF, ảnh scan)
│   ├── scan/vi/                # UC01
│   ├── scan/en/                # UC02
│   ├── scan/ja/                # UC03
│   ├── table/vi/               # UC04
│   ├── table/en/               # UC05
│   ├── table/ja/               # UC06
│   ├── text_layer/vi/          # UC07
│   ├── text_layer/en/          # UC08
│   └── text_layer/ja/          # UC09
│
├── ground_truth/               # JSON theo schema trên
│   ├── scan/vi/
│   ├── table/vi/
│   └── ...
│
├── predictions/                # Output của từng model
│   ├── gpt4o/
│   ├── gemini_25_pro/
│   ├── azure_doc_intelligence/
│   ├── google_doc_ai/
│   ├── paddleocr/
│   └── ...
│
└── results/                    # Score sau khi chạy eval
    ├── cer_scores.csv
    ├── teds_scores.csv
    └── summary_leaderboard.csv
```

---

## Bước 5: Models cần benchmark

| Tier | Model | UC phù hợp |
|---|---|---|
| Cloud frontier | GPT-4o, Gemini 2.5 Pro | Tất cả |
| Cloud OCR chuyên biệt | Azure Document Intelligence, Google Document AI, AWS Textract, Marker API (datalab.to) | Scan + Table |
| Cloud mới nổi | Mistral OCR, Docling (IBM) | Scan + Text layer |
| Open source | PaddleOCR v4, Surya, DocTR, Tesseract 5, Marker OCR | Scan |
| Open source table | TableTransformer, TATR, Marker OCR | UC04-06 |
