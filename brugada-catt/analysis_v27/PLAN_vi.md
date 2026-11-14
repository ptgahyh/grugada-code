# Bước 27: chạy lại kết quả chính trên Brugada với bộ lọc chung cho mọi nguồn. Kế hoạch (viết và commit TRƯỚC khi chạy)

Ngày viết: 14/11/2026. Chưa chạy phép tính nào của bước này (chỉ kiểm code: tái tạo phần Melo của `harmonize.py` cho đúng
`claris.npz`, `mortara.npz` hiện có, sai khác 0,0).

## Lý do

Pipeline hiện tại không đối xứng: HUCA qua lọc Butterworth bậc 2, 0,5–40 Hz, hai chiều; nhịp Melo (nhịp đại diện của máy, 200 Hz)
chỉ được hạ mẫu bằng `resample_poly`. Bước 23 cho thấy đồng bộ dải thông lấp khoảng 20% khoảng cách tới oracle. Người phản biện sẽ hỏi
"ngưỡng hỏng có phải do chính tiền xử lý". Bước này chạy lại toàn bộ chuỗi chính với **một thay đổi duy nhất**.

## Thay đổi duy nhất

Nhịp Melo (Claris, Mortara) được lọc **thông thấp Butterworth bậc 2, 40 Hz, hai chiều (`filtfilt`) ở 200 Hz, trước khi hạ mẫu**.
Đây là cùng thành phần thông thấp mà HUCA đã qua. Thành phần thông cao 0,5 Hz không áp dụng được cho một nhịp đơn dài 0,75 s
(và nhịp đại diện của máy đã khử trôi nền); ghi rõ trong bài. HUCA, cách căn, z-score, mô hình, hạt giống: giữ nguyên.

## Chuỗi chạy (y như bước 5 và 8)

1. `analysis_v27/harmonize_v27.py` → `harmonized_v27/` (HUCA chép nguyên; Claris, Mortara lọc lại).
2. `concepts_v5/compare_v5.py harmonized_v27 …` (CNN-V1V3, CBM3, CBM5, các Hybrid; 9 thiết lập × 3 seed).
3. `analysis_v8/v8_core.py harmonized_v27 …` (CNN, AdaBN, TENT, DANN, CORAL, mỏ neo).
4. `analysis_v8/v8_analyze.py` (bootstrap ghép cặp 1000 lần, tỉ lệ bệnh 5%, đường cỡ lô).

## Tiêu chí (định trước)

| Mã | Nội dung | Đạt khi |
|---|---|---|
| U1 | Kết quả chính: CATT − SRC gộp 7 thiết lập held-out, bootstrap ghép cặp | hiệu > 0 và KTC 95% không chứa 0 |
| U2 | Số thiết lập hỏng (trong 9) | mô tả, so với 5/9 cũ |
| U3 | CATT − UPA, − GMM-2, − Otsu, − MED gộp held-out | mô tả, kèm KTC; so với bước 8 (+0,013 / +0,015 / +0,016 / +0,026) |
| U4 | AUROC đổi viện trung bình 9 thiết lập (CNN-V1V3, CBM5) | mô tả, so với 0,865 cũ |
| U5 | BA held-out: SRC, CATT, ORACLE | mô tả, so với 0,701 / 0,799 / 0,820 cũ |

## Diễn giải định trước

- U1 đạt: kết luận chính không phụ thuộc vào sự bất đối xứng tiền xử lý. **Đề xuất** (quyết định là của tác giả): dùng pipeline
  thống nhất làm pipeline chính của bài, pipeline cũ thành phân tích độ nhạy; các bước 13–25 trên Brugada khi đó cần chạy lại
  hoặc ghi rõ là trên pipeline cũ.
- U1 không đạt: lợi ích của CATT trên Brugada phần lớn do bất đối xứng tiền xử lý; phải báo cáo và đổi trọng tâm bài.

## Tệp sẽ tạo

`harmonize_v27.py`, `results/` (bảng tổng hợp của v5 và v8), `summary_v27.json`, `README_vi.md`. Điểm số từng bệnh nhân ở thư mục riêng.
