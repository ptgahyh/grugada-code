# Ba miền Brugada đã chuẩn hóa + baseline

## Dữ liệu (`huca.npz`, `claris.npz`, `mortara.npz`)
Mỗi file gồm `X` (N, 9, 75), `y` (1 = Brugada), `ids`, `leads`, `fs = 100`.

- **Chuyển đạo:** I, II, III, V1–V6. Mỗi bệnh nhân là 1 nhịp trung vị, 100 Hz, 75 mẫu (−290 đến +460 ms quanh QRS).
- **Căn chỉnh:** đỉnh năng lượng QRS đặt tại mẫu 28. Mỗi chuyển đạo được z-score.
- **HUCA** còn 356 ca (73 Brugada; mã 1 và mã 2 đều tính là Brugada). Đã bỏ 4 bản trùng và 3 bản ghi có đoạn đứng yên. Tín hiệu lọc 0.5–40 Hz, R được phát hiện trên năng lượng của 9 chuyển đạo.
- **Melo:** hạ từ 200 xuống 100 Hz (có lọc chống chồng phổ), rồi căn chỉnh lại theo cùng quy tắc.
  - Claris: 1154 ca, 596 Brugada.
  - Mortara: 370 ca, 137 Brugada.

Chạy lại: `python harmonize.py <thư mục HUCA> <thư mục melo_processed> <thư mục ra>`, rồi `python baselines.py <thư mục ra>`.

## Kết quả baseline (`baseline_results.csv` / `.json`)

**Ba mô hình:**
- **ST_V2:** một đặc trưng, không huấn luyện. Lấy ST ở V2 (60–90 ms sau QRS) trừ đi nền PR.
- **LogReg:** hồi quy logistic L2.
- **CNN:** CNN 1D nhỏ, ensemble 3 seed.

**Ba giao thức:**
- **within:** CV 5-fold trong cùng một miền.
- **cross:** train trên miền A, test trên miền B.
- **lodo:** train trên 2 miền, test trên miền còn lại.

**Ngưỡng:** chọn bằng Youden trên 15% validation tách từ tập train.

`domain classifier`: hồi quy logistic phân biệt 3 miền từ chính các nhịp đã chuẩn hóa. Dùng để kiểm tra "dấu vân tay miền", tức nguy cơ shortcut.

`mean_beats_by_domain.png`: nhịp trung vị theo miền và nhãn (II, V1, V2).
