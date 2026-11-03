# Bước 16: CATT theo độ nhạy cố định — kết quả trên Brugada

Kế hoạch `PLAN_vi.md` được commit lúc 06:51 UTC ngày 03/11/2026 (34c9516), trước khi tính bất kỳ đại lượng nào theo độ nhạy trên Brugada. Không có sai lệch so với kế hoạch.

## Kết quả theo tiêu chí

| Mã | Nội dung | Kết quả | Đạt? |
|---|---|---|---|
| S1a | Đẳng thức độ lệch (mỏ neo khái niệm) | Sai số trung vị 0,136 (tiêu chí ≤ 0,03); Spearman 0,56 | **Không** |
| S1b | Tính an toàn "độ nhạy không thiếu" (mỏ neo khái niệm) | 56% điểm có TPR ≥ s − 0,02 (tiêu chí ≥ 80%) | **Không** |
| S1c | Mô tả, mỏ neo hộp đen | Sai số 0,161; 52% đạt; CNNalt, ResNet thiếu độ nhạy trung vị −0,15 đến −0,17 | Đúng chiều dự đoán (hộp đen kém hơn) |
| S2 | Dao động của độ nhạy, công thức | Dự đoán/quan sát **0,97** (IQR 0,84–1,11), Spearman **0,97** | **Đạt** |
| S3 | Dao động của ngưỡng, bootstrap không nhãn | **0,99** (IQR 0,92–1,08), Spearman **0,98** | **Đạt** |

Thêm: dao động của ngưỡng khi biết các đại lượng trên cả cohort: 1,03, Spearman 0,97.

## Đọc kết quả

### Công thức dao động (C2): đúng, và tốt hơn bản Youden

- Đúng ở cả 7 mỏ neo (0,91–1,03) và ở cả 3 mục tiêu s (0,90–1,02).
- **Bootstrap không cần nhãn** cho 0,99, tốt hơn F1 của bản Youden (0,98, Spearman 0,93).
- Sai số giảm theo n^{−1/2}, nhanh hơn n^{−1/3} của Youden.

### Đẳng thức độ lệch (C1): không đúng trên dữ liệu thật

- Theo giả định độc lập, độ nhạy phải vượt mục tiêu khoảng +0,11. Thực tế chỉ vượt +0,01 (mỏ neo khái niệm) và −0,01 (hộp đen).
- **Lý do:** mỏ neo **không** độc lập có điều kiện với f, kể cả mỏ neo khái niệm.
  - Những ca âm bị mỏ neo gọi nhầm là dương (FDR khoảng 0,3) cũng là những ca f cho điểm cao, tức là các "ca âm khó".
  - Vì vậy chúng không kéo phân vị xuống như giả định độc lập dự đoán.
- **Hệ quả lâm sàng:** tính chất "CATT-độ nhạy không bao giờ thiếu độ nhạy" **không dùng được trong thực tế**. Chỉ 52–56% điểm đạt mục tiêu.
  - Ngưỡng gốc chọn theo độ nhạy trên nguồn (SRC-s) còn kém hơn: 41–42%.
  - Đổi lại, độ đặc hiệu của CATT thấp hơn (0,53 so với 0,70 ở mỏ neo khái niệm).

## Y văn

- **Đẳng thức C1 không mới.** Dưới giả định độc lập có điều kiện, đây là cách viết lại công thức "độ nhạy biểu kiến khi so với chuẩn tham chiếu không hoàn hảo" của Gart & Buck (1966) và Staquet et al. (1981); xem tổng quan của Chikere et al. (2021).
- **Hiện tượng "bị nhiễu nhãn thì ngưỡng thận trọng quá mức"** ở bài toán Neyman–Pearson đã được Yao, Rava, Tong & James (arXiv 2112.00314) phân tích.
- **C2 là tiệm cận chuẩn của phân vị mẫu** (Bahadur), áp cho nhóm nhãn giả dương.
- Phần có giá trị cho bài là **ứng dụng**: lập kế hoạch cỡ mẫu không cần nhãn cho CATT-độ nhạy, và bằng chứng rằng giả định độc lập bị vi phạm trên ECG thật.

## Tệp

- `sens_brugada.py`: script kiểm.
- `sens_c1.csv`: theo từng thiết lập × seed × mỏ neo × s.
- `sens_c2.csv`: theo từng cỡ mẫu.
- `sens_summary.json`: tổng hợp.

Chạy lại: `python analysis_v16/sens_brugada.py analysis_v11 analysis_v16` (cần `analysis_v11/scores`, khoảng 15 phút).
