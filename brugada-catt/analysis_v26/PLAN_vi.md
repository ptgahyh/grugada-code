# Bước 26: CATT có cổng, và so CATT với việc gán nhãn vài ECG đích. Kế hoạch (viết và commit TRƯỚC khi chạy)

Ngày viết: 14/11/2026. Chưa chạy phép tính nào của bước này. Dữ liệu: điểm số bước 8 (CNN-V1V3, mỏ neo CBM5), 9 thiết lập Brugada
× 3 seed. Phần RBBB/LVH của 26a được đăng ký ngay ở đây nhưng chỉ chạy sau khi có lại điểm số bước 14–15 (cần bộ Chapman/Ningbo).

Ký hiệu: t_src = ngưỡng Youden trên validation nguồn; t_c = ngưỡng CATT (Youden theo nhãn giả CBM5 trên toàn cohort đích);
t_h = ngưỡng CATT làm trơn (bước 20). BA = balanced accuracy trên toàn cohort đích (nhãn chỉ để đánh giá).
"Hỏng" như bài: BA(t_src) < BA(oracle) − 0,05. Brugada: thiết lập hỏng 0, 1, 2, 4, 6; không hỏng 3, 5, 7, 8; held-out 2–8.

## 26a. CATT có cổng

Lý do: CATT gây hại nhẹ ở RBBB/LVH (−0,008; −0,010), nơi ngưỡng nguồn vốn đã đúng. Cổng dùng chính công thức độ dao động của bài:
chỉ đổi ngưỡng khi ngưỡng mới cách ngưỡng nguồn xa hơn mức nhiễu lấy mẫu của nó.

  t_gate = t_c nếu |t_c − t_src| > z·σ̂, ngược lại t_src.

- **Bản thường:** σ̂ = 0,5134·(4Ṽ/(Ã²n))^(1/3), Ṽ và Ã từ KDE Scott trên toàn cohort đích (Hệ quả 2, không nhãn).
- **Bản làm trơn:** t_h thay t_c, σ̂ = G2 (sandwich, bước 24) trên toàn cohort đích.
- z = 2 (chính); z = 1 và 3 (độ nhạy). Không có bước chọn z trên dữ liệu.

| Mã | Nội dung | Đạt khi |
|---|---|---|
| G1 | Brugada, 5 thiết lập hỏng: cổng giữ được bao nhiêu phần lợi ích của CATT, (BA_gate − BA_src)/(BA_catt − BA_src), trung bình | ≥ 0,90 cho cả bản thường và bản làm trơn |
| G2 | Brugada, 4 thiết lập không hỏng: cổng mở bao nhiêu lần (trên 12 = 4 × 3 seed) | mô tả |
| G3 | RBBB và LVH (khi có điểm số): BA_gate − BA_src gộp, bootstrap bệnh nhân như bước 14–15 | ≥ −0,003 (tức hết hại) cho bản làm trơn với mỏ neo chính A3 (LVH) và mỏ neo chính bước 14 (RBBB) |
| G4 | RBBB và LVH: phần hại của CATT được cổng tránh | mô tả |

Dự đoán trước (không phải tiêu chí): cổng chỉ chặn được độ lệch cỡ nhiễu. Ở chỗ CATT lệch do mỏ neo phụ thuộc (ví dụ LVH, PTB → cn)
độ lệch lớn, nên cổng sẽ mở và không tránh được hại.

## 26b. So với gán nhãn k ECG đích

Đối thủ thực tế của CATT: nhờ bác sĩ đọc k ECG ở viện mới rồi đặt ngưỡng Youden trên k ECG đó.

- k ∈ {10, 20, 30, 50, 100, 200}; 300 lần rút ngẫu nhiên mỗi k; nếu k ECG thiếu một lớp thì dùng t_src. Đánh giá BA trên cohort đích.
- Hai bản: Youden thường và Youden làm trơn (cùng độ rộng h như bước 20) trên k nhãn thật.
- **Tỉ lệ bệnh tự nhiên** của cohort đích (20–52%) và **tỉ lệ bệnh 5%** như bước 8 (giữ mọi ca âm, rút ca dương cho đủ 5%, rút lại mỗi lần).
- So với CATT và CATT làm trơn dùng toàn bộ cohort đích không nhãn (ở 5%: cùng cohort đã rút).

| Mã | Nội dung | Báo cáo |
|---|---|---|
| L1 | k* = k nhỏ nhất mà BA trung bình (7 thiết lập held-out) của ngưỡng gán nhãn ≥ BA của CATT làm trơn | k*, ở tỉ lệ tự nhiên và 5% |
| L2 | Phân vị 10% của BA theo k | bảng |
| L3 | Ở 5%: số ca dương kỳ vọng trong k ECG | bảng |

Bước này mô tả, không có tiêu chí đạt/không đạt. Kết quả dùng để viết câu "CATT tương đương khoảng k ECG có nhãn" trong bài.

## Tệp sẽ tạo

`gate26.py`, `labelk26.py`, các `.csv` tổng hợp, `summary_v26.json`, `README_vi.md`.
