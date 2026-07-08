# Hướng dẫn sử dụng OCR Benchmark Web Tool

---

## 1. Khởi động

### Yêu cầu

- Python 3.10+, `uv` đã cài
- File `.env` chứa `MARKER_API_KEY`
- PDF đã đặt vào thư mục `raw/<uc_type>/<lang>/`

### Chạy server

```bash
uv run uvicorn webapp.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Mở trình duyệt tại: **http://localhost:8000**

> **Khi deploy lên server:**
> ```bash
> DATA_ROOT=/var/data/ocr_benchmark uv run uvicorn webapp.backend.main:app --host 0.0.0.0 --port 8000
> ```

---

## 2. Giao diện tổng quan

<!-- TODO: chèn ảnh chụp màn hình giao diện chính -->

Giao diện gồm 3 khu vực chính:

| Khu vực | Mô tả |
|---|---|
| **Topbar** | Tiêu đề, badge thống kê, nút Dashboard và Refresh |
| **Sidebar** | Danh sách file PDF, có thể lọc theo tên |
| **Main** | Khu vực làm việc chính, gồm 4 bước |

---

## 3. Dashboard — Xem trạng thái tổng quan

Click nút **📊 Dashboard** trên topbar để mở bảng tổng quan.

<!-- TODO: chèn ảnh dashboard -->

Bảng hiển thị toàn bộ file với các cột:

| Cột | Ý nghĩa |
|---|---|
| **File** | Tên doc_id (ví dụ: `scan_en_001`) |
| **UC Type** | Loại use case: `scan`, `table`, `text_layer` |
| **Lang** | Ngôn ngữ: `vi`, `en`, `ja` |
| **OCR** | ✓ OCR — đã chạy Marker / — chưa chạy |
| **GT Status** | 🟢 Done / 🟠 In progress / ⚪ Not started |
| **Reviewer** | Tên người đã review |
| **Eval** | ✓ Eval — đã có kết quả đánh giá |
| **Updated** | Thời gian lưu GT gần nhất |

**Stats bar** trên cùng hiện tổng số file / đã hoàn thành / đang làm / chưa bắt đầu.

> **Tip:** Click vào bất kỳ dòng nào trong bảng để nhảy thẳng vào file đó.

---

## 4. Luồng làm việc — 4 bước

### Bước 1 — Select File

<!-- TODO: chèn ảnh step 1 -->

- Sidebar bên trái hoặc grid trung tâm hiển thị danh sách PDF
- Mỗi file có dot màu:
  - 🟢 **Xanh lá** — đã có GT được đánh giá đầy đủ
  - 🔵 **Xanh dương** — đã chạy OCR, chưa có GT
  - 🟠 **Cam** — chưa chạy OCR
- Gõ vào ô tìm kiếm để lọc theo tên file
- Click vào file → **tự động chuyển sang Bước 2**

---

### Bước 2 — Run OCR

<!-- TODO: chèn ảnh step 2 -->

#### Chọn model OCR

Hiện tại hỗ trợ: **Marker OCR** (datalab.to)

#### Trạng thái OCR

| Trạng thái | Hiển thị |
|---|---|
| Đã chạy rồi | ✅ OCR already done — prediction files ready |
| Chưa chạy | ⏳ Not yet run for this file |

#### Chạy OCR

1. Click **▶ Run OCR** để chạy lần đầu
2. Progress bar xuất hiện, polling tự động mỗi 2 giây
3. Khi hoàn thành → hiển thị ✅ OCR complete!

> **Idempotent:** Nếu OCR đã chạy rồi, hệ thống tự động skip. Dùng **Force re-run** nếu muốn chạy lại từ đầu.

4. Click **Next: Review GT →** để sang bước 3

---

### Bước 3 — Review GT

<!-- TODO: chèn ảnh step 3 — split view PDF + editor -->

Bước quan trọng nhất — review và chỉnh sửa Ground Truth.

#### Giao diện split view

```
┌─────────────────┬─────────────────┐
│   PDF Preview   │  GT Editor      │
│                 │                 │
│  [PDF với bbox  │  [Text / Table  │
│   highlight]    │   editor]       │
│                 │                 │
└─────────────────┴─────────────────┘
```

#### Tương tác với PDF

<!-- TODO: chèn ảnh highlight bbox trên PDF -->

- Mỗi block text/bảng được vẽ **khung màu** trên PDF:
  - 🟦 **Tím** — block văn bản (Text, SectionHeader, ListGroup)
  - 🟩 **Xanh lá** — block bảng (Table)
- **Click vào khung** → tự động:
  1. Nhảy tab đến đúng trang
  2. Switch sang mode Text hoặc Table tương ứng
  3. Highlight khung đó đậm hơn

#### Mode Text vs Table

Trên header editor có 2 nút toggle:

| Nút | Hiển thị |
|---|---|
| **📝 Text** | Textarea chỉnh `full_text` của từng trang |
| **📊 Table** | Textarea chỉnh HTML của từng bảng + Preview |

- Tab trang có dấu **T** nhỏ màu tím nếu trang đó có bảng
- Click tab trang → PDF tự scroll đến trang đó

#### Chỉnh sửa text

<!-- TODO: chèn ảnh text editor -->

- Chỉnh trực tiếp trong textarea
- Góc trên phải hiển thị số ký tự hiện tại
- Xoá dòng thừa, sửa lỗi nhận dạng, thêm nội dung bị bỏ sót

#### Chỉnh sửa bảng

<!-- TODO: chèn ảnh table editor -->

- Chỉnh HTML raw của bảng trực tiếp
- Click **👁 Preview** để xem bảng render thật
- Cells được tự động parse lại sau khi sửa HTML

#### Reset về draft gốc

Click **Reset Draft** để về output gốc của Marker (có confirm dialog).

---

### Bước 4 — Scores

<!-- TODO: chèn ảnh score panel -->

Sau khi Save & Evaluate xong, tự động chuyển sang bảng điểm:

#### Text Evaluation

| Metric | Ý nghĩa | Tốt khi |
|---|---|---|
| **CER** | Character Error Rate — % ký tự sai | Thấp (< 2%) |
| **WER** | Word Error Rate — % từ sai | Thấp |
| **nWER** | WER sau normalize hoa/thường + dấu câu | Thấp |

#### Table Evaluation

| Metric | Ý nghĩa | Tốt khi |
|---|---|---|
| **TEDS** | Tree Edit Distance Similarity — độ khớp cấu trúc bảng | Cao (> 0.9) |

Màu sắc điểm:
- 🟢 **Xanh** — tốt
- 🟠 **Cam** — trung bình
- 🔴 **Đỏ** — cần cải thiện

Per-page detail table phía dưới giúp xác định trang nào có vấn đề.

---

## 5. Save GT — Dialog xác nhận

<!-- TODO: chèn ảnh save modal -->

Khi click **💾 Save GT & Evaluate** hoặc nhấn **Ctrl+S**:

1. Dialog xuất hiện với 2 field:
   - **Reviewer name** — tên người review (được nhớ tự động cho lần sau)
   - **Review status** — chọn trạng thái:
     - 🟠 **In progress** — đang xử lý, chưa hoàn thiện
     - 🟢 **Done** — đã xác nhận xong, có thể dùng làm GT

2. Click **Save & Evaluate** hoặc nhấn **Enter**

3. Hệ thống tự động:
   - Lưu GT vào `ground_truth/<uc_type>/<lang>/<doc_id>.json`
   - Chạy eval so sánh GT với Marker prediction
   - Hiện kết quả ở Bước 4

> **GT JSON được lưu gồm:**
> ```json
> {
>   "doc_id": "scan_en_001",
>   "reviewer": "Uyên",
>   "status": "done",
>   "updated_at": "2026-07-08T06:25:00Z",
>   "pages": [...]
> }
> ```

---

## 6. Cấu trúc thư mục

```
raw/
├── scan/vi/          scan_vi_001.pdf, scan_vi_002.pdf, ...
├── scan/en/          scan_en_001.pdf, ...
│   └── marker_output/
│       ├── scan_en_001_text_prediction.json
│       ├── scan_en_001_table_prediction.json
│       └── scan_en_001_full_response.json
├── table/vi/
└── text_layer/en/

ground_truth/
├── scan/en/          scan_en_001.json   ← GT đã review
└── table/vi/

benchmark_results/
└── marker/
    └── scan/en/      scan_en_001_eval.json
```

---

## 7. Keyboard shortcuts

| Phím | Hành động |
|---|---|
| **Ctrl/Cmd + S** | Mở dialog Save GT (khi ở Bước 3) |
| **Enter** | Xác nhận save (khi dialog đang mở) |
| **Esc** | Đóng dialog / Dashboard |

---

## 8. Quy trình review đề xuất

```
Lần 1 (draft):
  → Chạy OCR cho tất cả file
  → Lưu với status "In progress"

Lần 2 (review):
  → So sánh từng trang với PDF gốc
  → Sửa lỗi nhận dạng
  → Lưu với status "Done"
```

**Nguyên tắc review tốt:**
- Đọc từng dòng theo PDF gốc, không đoán
- Giữ nguyên từ viết tắt, số, ký hiệu đặc biệt
- Với bảng: kiểm tra đúng số cột, hàng, merged cell
- Đánh dấu "In progress" nếu còn trang chưa kiểm tra
