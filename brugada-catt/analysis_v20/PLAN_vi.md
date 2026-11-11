# Bước 20: ngưỡng Youden làm trơn. Kế hoạch (viết và commit TRƯỚC khi chạy)

Ngày viết: 11/11/2026. Chưa chạy phép tính nào của bước này.

## Ý tưởng

Ngưỡng CATT hiện tại làm lớn nhất một hàm bậc thang nên hội tụ chậm (n^(−1/3)): giảm một nửa sai số cần gấp 8 lần ECG.
Thay hàm chỉ thị bằng hàm phân phối chuẩn Φ với độ rộng h (như ước lượng maximum score làm trơn, Horowitz 1992):

  J_h(t) = (1/n₁)·Σ_{ỹ=1} Φ((f_i − t)/h) − (1/n₀)·Σ_{ỹ=0} Φ((f_i − t)/h),   t̂_h = argmax J_h.

t̂_h chính là điểm hai mật độ kernel của hai lớp nhãn giả cắt nhau. Lý thuyết dự đoán: tốc độ n^(−2/5), phân phối giới hạn chuẩn,
và bootstrap thường dùng được. Đổi lại có độ lệch bậc h². Đây **không phải ý tưởng mới** về thống kê; điều cần kiểm là nó có lợi
thật trong bài toán này không.

**Độ rộng (định trước):** h = 0,9·min(SD, IQR/1,34)·n_min^(−1/5), SD và IQR của f trên cả mẫu, n_min = cỡ lớp nhãn giả nhỏ hơn.
Độ nhạy: h × 0,5 và h × 2. Tính trên lưới 1024 ô đều từ min(f) − 3h đến max(f) + 3h (đếm theo ô rồi tích chập), lấy argmax toàn cục,
nội suy parabol quanh đỉnh. Ngưỡng thường t̂ tính như bước 18.

## S1. Mô phỏng (mô hình Gauss của bài, 4 cấu hình mỏ neo × n ∈ {100, 300, 1000, 3000}, 2000 mẫu mỗi ô; hạt giống 2001)

Ngưỡng quần thể t̃ và balanced accuracy thật tính giải tích/trên lưới như `analysis_v13/unified_sim.py`.

| Mã | Thước đo | Đạt khi |
|---|---|---|
| S1a | RMSE của ngưỡng quanh t̃: làm trơn / thường | trung vị 16 ô ≤ 0,85 **và** không ô nào > 1,00 |
| S1b | Tổn thất balanced accuracy trung bình so với ngưỡng tối ưu thật t*: làm trơn / thường | trung vị 16 ô ≤ 1,00 |
| S1c | Độ dốc log SD theo log n | mô tả; lý thuyết −0,40 (làm trơn) so với −0,33 (thường) |
| S1d | Bootstrap thường (B = 200) ước lượng SD của t̂_h, 200 mẫu mỗi ô, h giữ cố định trong bootstrap | trung vị tỉ số ∈ [0,80; 1,25] ở từng n **và** trung vị \|log tỉ số\| gộp ≤ 0,15 (ước lượng tốt nhất cho ngưỡng thường ở bước 18: 0,125) |

## S2. Brugada (điểm số bước 8, mỏ neo CBM-sd15, 9 thiết lập × 3 seed; hạt giống 2002)

| Mã | Thước đo | Đạt khi |
|---|---|---|
| S2a | SD của ngưỡng qua 300 mẫu con không hoàn lại, n = 20/35/50% cohort đích: làm trơn / thường | trung vị 81 điểm ≤ 0,85 |
| S2b | Balanced accuracy trên đích (nhãn chỉ để đánh giá) với ngưỡng từ 200 mẫu con cỡ n = 50 và n = 100, trung bình 7 thiết lập giữ lại (2–8) | làm trơn − thường ≥ −0,005 ở cả hai n |
| S2c | Như S2b nhưng dùng toàn bộ cohort đích | \|làm trơn − thường\| ≤ 0,010 |
| S2d | Phân vị 10% của balanced accuracy ở n = 50 và 100 | mô tả |

## Diễn giải định trước

- S1a, S1b, S2a, S2b, S2c đều đạt: đề xuất ngưỡng làm trơn là biến thể chính cho các phân tích **sau này** và báo cáo cả hai.
- S2b hoặc S2c không đạt (làm trơn hại độ chính xác trên dữ liệu thật): báo cáo là kết quả âm tính; giữ ngưỡng thường.
- Mọi kết quả đều được báo cáo, kể cả không đạt. Không đổi con số nào đã báo cáo ở các bước trước.

## Tệp sẽ tạo

`smooth_lib.py`, `smooth_sim.py`, `smooth_brugada.py`, các `.csv` tổng hợp, `summary_v20.json`, `README_vi.md`.
