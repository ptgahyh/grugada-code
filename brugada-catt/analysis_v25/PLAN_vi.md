# Bước 25: ngưỡng làm trơn ngoài mô hình Gauss và với nhiều mỏ neo trên Brugada. Kế hoạch (viết và commit TRƯỚC khi chạy)

Ngày viết: 11/11/2026. Chưa chạy phép tính nào của bước này. Kiểm lại các kết quả bước 20 và 24 (ngưỡng làm trơn) ở hai chỗ còn
thiếu: (a) mô phỏng chỉ có phân phối Gauss; (b) Brugada chỉ có một mỏ neo. Ngưỡng làm trơn, độ rộng h, công thức SD và các ước
lượng G1, G2 giữ nguyên như `analysis_v20/smooth_lib.py` và `analysis_v24/lib24.py`.

## 25a. Mô phỏng ngoài Gauss (hạt giống 2501)

Bốn kịch bản của bước 13 (`analysis_v13/nongauss_sim.py`, copula Gauss giữa f và mỏ neo): **lệch** (Gamma), **đuôi dày** (t₃),
**hai đỉnh**, **rời rạc** (điểm số làm tròn bước 0,25, khoảng 0,2 SD; lý thuyết căn bậc ba của ngưỡng thường không áp dụng được ở đây).
Mỗi kịch bản 16 ô: ρ ∈ {0; 0,6} × độ dời ∈ {0; −1,25} × n ∈ {100, 300, 1000, 3000}. 1000 mẫu mỗi ô cho SD và tổn thất; 100 mẫu mỗi ô
cho G1, G2.

| Mã | Nội dung | Đạt khi |
|---|---|---|
| N1 | RMSE của ngưỡng quanh t̃ (ngưỡng quần thể thường), làm trơn / thường | trung vị mỗi kịch bản ≤ 1,00 **và** trung vị 64 ô ≤ 0,85 |
| N2 | Tổn thất balanced accuracy so với tối ưu thật, làm trơn / thường | trung vị mỗi kịch bản ≤ 1,00 |
| N3 | Công thức SD (dạng chính xác, mật độ quần thể của kịch bản, h = trung vị h mẫu) so với SD quan sát | trung vị mỗi kịch bản ∈ [0,85; 1,15] |
| N4 | G1, G2 từ một mẫu: trung vị tỉ số và \|log\| theo kịch bản | mô tả (T2 bước 24 đã không đạt ở mô hình Gauss) |
| N5 | Số ECG cần cho ε = 0,15 và 0,30: thường / làm trơn, dùng hằng số quần thể | mô tả |

Kịch bản rời rạc: mật độ quần thể là các khối điểm trên lưới 0,25; dạng chính xác của công thức vẫn áp dụng được vì kernel làm trơn
khối điểm. Balanced accuracy tính theo quy tắc "điểm làm tròn ≥ t" như bước 13.

## 25b. Brugada, 5 mỏ neo (hạt giống 2502)

Điểm số bước 8 (9 thiết lập × 3 seed). Mỏ neo: CBM5 (`cbm5`, mỏ neo chính của bài), CBM3 (`cbm3`), ST_V2 (`stv2`), và hai mạng
thích ứng miền DANN (`dann`), CORAL (`coral`) dùng như mỏ neo phụ thuộc (cùng loại với CNNalt). Nhãn giả = điểm mỏ neo ≥ ngưỡng
Youden của mỏ neo trên validation nguồn. Lần chạy nào có lớp nhãn giả < 3 bản ghi thì bỏ và báo cáo.

| Mã | Nội dung (như bước 20 / 24) | Đạt khi |
|---|---|---|
| M1 | SD của ngưỡng qua 300 mẫu con (n = 20/35/50% N), làm trơn / thường | trung vị gộp ≤ 0,85 **và** trung vị từng mỏ neo ≤ 1,00 |
| M2 | BA trên đích với 50 và 100 ECG không nhãn (200 mẫu con, 7 thiết lập giữ lại 2–8), làm trơn − thường | gộp 5 mỏ neo ≥ −0,005 ở cả hai n |
| M3 | BA với toàn cohort đích, làm trơn − thường | \|hiệu gộp\| ≤ 0,010 |
| M4 | G2 (sandwich) từng mẫu đơn × √(1 − n/N), so với SD quan sát (20 mẫu con mỗi n) | gộp: trung vị tỉ số ∈ [0,80; 1,25] **và** \|log\| ≤ 0,20 |
| M5 | Số ECG cần (ε = 0,15; 0,30), thường / làm trơn, theo mỏ neo | mô tả |

Báo cáo từng mỏ neo, kể cả mỏ neo nào làm ngưỡng làm trơn kém hơn.

## Diễn giải định trước

- N1–N3 và M1–M4 đạt: ngưỡng làm trơn và công thức SD đủ vững để đưa vào bài làm biến thể chính.
- Một kịch bản hoặc một mỏ neo không đạt: báo cáo như điều kiện áp dụng (ví dụ "không dùng khi điểm số hai đỉnh").

## Tệp sẽ tạo

`nongauss25.py`, `brugada25.py`, `analyze25.py`, các `.csv`, `summary_v25.json`, `README_vi.md`.
