# Table Scoring — Tài liệu kỹ thuật

## Tổng quan

Pipeline tính score cho bảng gồm 2 tầng:
1. **`eval/table.py`** — so sánh từng cặp bảng GT vs Prediction, trả về TEDS per-table
2. **`metrics/uet_metrics.py`** — tính tập hợp metrics đầy đủ (TEDS, Cell F1, Cell Similarity, Shape) trên toàn bộ danh sách bảng

---

## Input format

### GT và Prediction

Cả GT lẫn Prediction đều có schema JSON:

```json
{
  "doc_id": "scan_en_001",
  "pages": [
    {
      "page_num": 1,
      "tables": [
        {
          "table_id": 1,
          "html": "<table><tr><th>...</th></tr>...</table>",
          "cells": [
            {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "text": "...", "is_header": true}
          ]
        }
      ]
    }
  ]
}
```

- `html`: HTML table string — source of truth cho TEDS
- `cells`: list cells đã được flatten (dùng cho cell-level metrics)
- Bảng đã là HTML từ đầu, **không qua bước MD → HTML**

---

## Luồng tính score (call stack)

```
POST /api/ocr/eval
  └── run_eval() [ocr_runner.py]
       └── eval_table(gt_page, pred_tables) [eval/table.py]
            └── _compute_teds(gt_html, pred_html)   ← per-table TEDS
                 ├── extract_html_tables(gt_html)    → cell grid
                 ├── extract_html_tables(pred_html)  → cell grid
                 └── teds_similarity_table(grid1, grid2)
       └── compute_table_metrics_from_html(gt_html_list, pred_html_list) [uet_metrics.py]
            └── table_metrics(ref_md, pred_md)
                 ├── extract_tables()
                 ├── match_tables()        ← Hungarian matching
                 ├── teds_similarity_table()
                 ├── cell_exact_f1_aligned()
                 ├── avg_cell_text_similarity_aligned()
                 └── table_shape_similarity()
```

---

## Metric 1: TEDS (Tree Edit Distance Similarity)

### Công thức

```
TEDS = 1 - EditDistance(T_ref, T_pred) / max(|T_ref|, |T_pred|)
```

- `T_ref`, `T_pred`: cây DOM của bảng GT và Prediction
- `EditDistance`: số phép rename/insert/delete node tối thiểu
- `|T|`: số node trong cây

### Cách xây dựng cây

**Bước 1**: HTML → cell grid qua `extract_html_tables()`

```python
# BeautifulSoup parse HTML → list[list[str]]
# rowspan/colspan được flatten (ignored) trong grid, chỉ dùng text
# normalize_cell() được áp dụng cho mỗi cell text
```

**Bước 2**: cell grid → HTML chuẩn qua `table_to_html_grid()`

```html
<!-- Row 0 dùng <th>, các row còn lại dùng <td> -->
<table>
  <tr><th>A</th><th>B</th></tr>
  <tr><td>1</td><td>2</td></tr>
</table>
```

**Bước 3**: HTML chuẩn → cây DOM qua `html_table_to_tree()`

- Strip whitespace giữa tags (`>\s+<` → `><`) để tránh spurious text nodes
- Mỗi node `td`/`th` có label `"th:celltext"` hoặc `"td:celltext"`
- Text được `normalize_cell()` trước khi đưa vào label

**Bước 4**: APTED algorithm tính edit distance giữa 2 cây

```python
dist = APTED(ref_tree, pred_tree, TEDSConfig()).compute_edit_distance()
```

### Chi phí các phép biến đổi

| Phép | Chi phí |
|------|---------|
| Insert node | 1.0 |
| Delete node | 1.0 |
| Rename node cùng tag | `1 - normalized_edit_similarity(text1, text2)` |
| Rename node khác tag | 1.0 |

> Rename `td:text1` → `td:text2` cho phép TEDS phản ánh sự giống nhau về nội dung text, không chỉ structure.

### Matching nhiều bảng (Hungarian)

Khi trang có nhiều bảng, dùng **Hungarian algorithm** (`linear_sum_assignment`) để tìm matching tối ưu:

```python
sim[i, j] = teds_similarity_table(ref_tables[i], pred_tables[j])
# Hungarian: maximize tổng TEDS
row_ind, col_ind = linear_sum_assignment(1.0 - sim)
```

Score cuối: `teds_doc = sum(matched_scores) / max(len_ref, len_pred)` — penalty nếu số bảng không khớp.

---

## Metric 2: Cell Exact F1

**Đo**: tỷ lệ cells có vị trí `(row, col)` và text giống nhau chính xác.

```python
def cell_exact_f1_aligned(ref_table, pred_table):
    ref_items = [(r, c, normalize_cell(val)) for r, row in ref for c, val in row]
    pred_items = [(r, c, normalize_cell(val)) for r, row in pred for c, val in row]
    return multiset_prf(ref_items, pred_items)['f1']
```

- So sánh **multiset** (không phân biệt thứ tự khi tính precision/recall)
- `normalize_cell()` được áp dụng trước khi so sánh

---

## Metric 3: Cell Text Similarity

**Đo**: trung bình `normalized_edit_similarity` cho từng vị trí `(row, col)`.

```python
def avg_cell_text_similarity_aligned(ref_table, pred_table):
    # Với mỗi (r, c): similarity = 1 - edit_distance / max(len_ref, len_pred)
    # Trung bình trên tất cả positions của bảng lớn hơn
```

- Soft match — cell bị thiếu ở một phía = similarity 0.0
- Khác với Cell Exact F1 ở chỗ cho phép partial credit

---

## Metric 4: Row/Col Count Similarity

```python
row_sim = 1.0 - |rows_ref - rows_pred| / max(rows_ref, rows_pred)
col_sim = 1.0 - |cols_ref - cols_pred| / max(cols_ref, cols_pred)
```

---

## Normalize cell text

Hàm `normalize_cell()` được áp dụng trước **tất cả** phép so sánh:

```python
def normalize_cell(s):
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)   # strip HTML tags
    s = re.sub(r"\*\*|__|\*|_|`", "", s)  # strip Markdown emphasis
    s = s.replace("×", "X")          # × → X (OCR confusion)
    s = s.replace("✓", "X")          # checkmark → X
    s = s.replace("✗", "X")          # cross → X
    s = s.replace("\u0445", "x")     # Cyrillic х → Latin x
    return normalize_ws(s)           # collapse whitespace
```

---

## Xử lý `<br/>` trong cell

Marker OCR hay dùng `<br/>` để xuống dòng trong cell:
```html
<th>4C1<br/>4C2</th>
```

Khi parse: `cell.textContent` → `"4C14C2"` (mất space).

**Fix**: trong `_htmlToHotData` (frontend), `<br>` được replace bằng space trước khi extract text:

```js
tempDiv.innerHTML = cell.innerHTML.replace(/<br\s*\/?>/gi, ' ');
const text = tempDiv.textContent.trim().replace(/\s+/g, ' ');
// → "4C1 4C2"
```

---

## Output schema

### Per-table (từ `eval_table`)

```json
{
  "table_id": 1,
  "teds": 0.998,
  "teds_detail": {
    "edit_distance": 2,
    "gt_tree_size": 45,
    "pred_tree_size": 45
  },
  "cell_accuracy": 0.95,
  "cell_diff": [...]
}
```

### Summary (từ `compute_table_metrics_from_html`)

```json
{
  "ref_table_count": 6,
  "pred_table_count": 6,
  "table_count_f1": 1.0,
  "table_teds_doc": 0.997,
  "table_teds_matched_mean": 0.997,
  "table_cell_exact_f1_mean": 0.980,
  "table_cell_text_similarity_mean": 0.997,
  "table_row_count_similarity_mean": 1.0,
  "table_col_count_similarity_mean": 1.0
}
```

---

## Những điểm cần lưu ý

| Điểm | Mô tả |
|------|-------|
| rowspan/colspan trong TEDS | Được flatten vào grid (không tính span structure), chỉ tính text content |
| th vs td | Không ảnh hưởng đến TEDS grid, nhưng ảnh hưởng đến raw tree distance (fallback) |
| Whitespace trong HTML | `html_table_to_tree` strip `\n` giữa tags để tránh lệch score |
| Table matching | Hungarian — match bảng nào với bảng nào là tối ưu toàn cục, không phải 1-1 theo thứ tự |
| Score penalty | `teds_doc` bị penalty khi số bảng GT ≠ Prediction (chia cho `max(n_ref, n_pred)`) |
