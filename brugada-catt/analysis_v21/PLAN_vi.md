# Bước 21: lệch căn thời gian và đường đáp ứng theo dạng sóng. Kế hoạch (viết và commit TRƯỚC khi chạy)

Ngày viết: 11/11/2026. Bước 19 (quan sát sau) thấy đáy sóng S ở V1 nằm tại 0 ms (HUCA) so với +20 ms (Claris, Mortara) quanh
điểm căn. Chưa chạy phép tính nào của bước này; chưa huấn luyện lại mô hình nào.

Mô hình: CNN-V1V3 như bước 8 (`concepts_v4/cbm_lib4.py`: `train_cnn`, tổ hợp 3 mạng, chia validation 85/15 theo `strat_val`),
9 thiết lập × 3 seed. Báo cáo sai khác lớn nhất so với điểm số bước 8 (`cnn_tgt`); nếu phiên bản PyTorch khác làm điểm số
không trùng từng số, báo cáo tương quan và AUROC thay cho yêu cầu trùng tuyệt đối.
"Hỏng" = balanced accuracy ở ngưỡng nguồn thấp hơn oracle quá 0,05 (như bài). 5 thiết lập hỏng gốc: 0, 1, 2, 4, 6.

## 21a. Dịch nhịp đích lúc chấm điểm (không huấn luyện lại)

Dịch nhịp đích s mẫu (10 ms mỗi mẫu; đệm bằng giá trị biên như `harmonize.align`), s ∈ {−3, …, +3}, rồi chấm điểm bằng mô
hình nguồn giữ nguyên. Độ dịch "khớp căn" s*: đích HUCA với nguồn Melo: +2; đích Melo với nguồn HUCA: −2.

| Mã | Thước đo (trung bình 5 thiết lập hỏng, 3 seed) | Diễn giải định trước |
|---|---|---|
| A1 | Tỉ lệ khoảng cách được lấp = (BA_SRC(s*) − BA_SRC(0)) / (BA_ORACLE(0) − BA_SRC(0)) | ≥ 0,20: lệch căn đóng góp đáng kể; ≤ 0,05: không đáng kể; ở giữa: nhỏ |
| A2 | Mức giảm \|Δ\| (độ dịch trung bình logit lớp âm so với validation nguồn) tại s* | mô tả |
| A3 | Đường BA_SRC và Δ theo s ∈ {−3…+3} | mô tả |

## 21b. Huấn luyện lại với cách căn chung

Căn lại **cả ba bộ** theo từng bản ghi: đáy sóng S của V1 (cực tiểu trong mẫu 18–39) đặt tại mẫu 28; chỉ dịch, không chuẩn hóa
lại. Huấn luyện và đánh giá y như bước 8.

| Mã | Thước đo | Diễn giải định trước |
|---|---|---|
| B1 | Số thiết lập hỏng trong 5 thiết lập gốc (theo trung bình 3 seed) | ≤ 2/5: căn chung sửa phần lớn; 5/5 và BA_SRC đổi < 0,02: không tác dụng; còn lại: một phần |
| B2 | BA_SRC trung bình 5 thiết lập hỏng, trước và sau | mô tả, kèm BA_ORACLE và AUROC đổi viện (để chắc không mất khả năng phân biệt) |
| B3 | \|Δ\| lớp âm trước và sau | mô tả |

## 21c. Đường đáp ứng của f theo tham số dạng sóng (chủ yếu mô tả)

Mô hình: ba mô hình một nguồn (HUCA, Claris, Mortara; seed validation 0). Nhịp nền: toàn bộ bản ghi **âm** của từng bộ.
t_J = đáy S của V1 + 40 ms (theo từng bản ghi). Biến đổi áp lên V1 và V2 (đơn vị z-score), rồi z-score lại từng chuyển đạo
như pipeline:

| Mã | Biến đổi | Mức |
|---|---|---|
| P1 | ST dạng vòm: cộng a·exp(−(t − t_J)/80 ms) với t ≥ t_J | a = 0; 0,25; …; 2,0 |
| P2 | Sóng T âm: trừ a·(cửa sổ Hann trên [t_J + 120, t_J + 280] ms) | a = 0; 0,25; …; 2,0 |
| P3 | r′/điểm J nhô: cộng a·Gauss(tâm t_J − 10 ms, σ = 15 ms) | a = 0; 0,25; …; 2,0 |
| P4 | QRS rộng: kéo giãn trục thời gian quanh mẫu 28 | hệ số 1,0; 1,1; …; 1,5 |
| P5 | Lọc thông thấp: làm trơn Gauss | σ = 0; 5; 10; 15; 20 ms |
| P6 | Dịch thời gian | −3 … +3 mẫu |

Thước đo: trung bình thay đổi logit của f theo mức, cho 3 mô hình × 3 bộ nhịp nền = 9 cặp.

| Mã | Nội dung | Đạt khi |
|---|---|---|
| R1 | f tăng đơn điệu theo ST dạng vòm (P1) | Spearman(mức, đáp ứng trung bình) ≥ 0,9 ở cả 9 cặp |
| R2 | Mô hình học ở các viện khác nhau đáp ứng giống nhau không | mô tả: tương quan Pearson của vectơ đáp ứng (mọi biến đổi × mức, cùng bộ nhịp nền) giữa từng cặp mô hình |
| R3 | Biến đổi nào tạo độ dịch cỡ độ dịch quan sát (1,3–3,4 logit) | mô tả |

Các biến đổi là phantom toán học đơn giản, không phải ECG bệnh lý thật; kết luận chỉ ở mức "mô hình nhạy với thay đổi nào".

## Tệp sẽ tạo

`train_v21.py` (huấn luyện + 21a + 21b, lưu mô hình ở thư mục riêng), `response_v21.py`, `analyze_v21.py`, các `.csv` tổng hợp,
`summary_v21.json`, hình, `README_vi.md`. Điểm số từng bệnh nhân và mô hình không đẩy lên kho.
