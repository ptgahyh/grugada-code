# Bước 23: độ trơn / dải thông của nhịp đại diện có gây trôi ngưỡng không. Kế hoạch (viết và commit TRƯỚC khi chạy)

Ngày viết: 11/11/2026. Chưa tính phổ, chưa huấn luyện, chưa chấm điểm gì của bước này.

## Giả thuyết

Bước 21c: f rất nhạy với làm trơn tín hiệu (σ = 10–20 ms làm điểm tăng 1,0–2,3 logit), cỡ độ dịch thật khi đổi viện. Hai nguồn
tạo nhịp khác nhau (`harmonized/harmonize.py`): HUCA = trung vị các nhịp đã lọc Butterworth bậc 2, 0,5–40 Hz, ở 100 Hz; Melo =
nhịp đại diện do máy tính, 200 Hz, hạ mẫu bằng `resample_poly` (lọc chống chồng phổ), không qua bộ lọc 0,5–40 Hz. Nếu dải thông
khác nhau là nguyên nhân chính, đưa các bộ về cùng dải thông sẽ làm ngưỡng nguồn bớt hỏng.

Mô hình: CNN-V1V3 như bước 8 (bước 21 đã tái lập trùng tuyệt đối), 9 thiết lập × 3 seed. "Hỏng" như bài (BA ở ngưỡng nguồn thấp
hơn oracle quá 0,05). 5 thiết lập hỏng gốc: 0, 1, 2, 4, 6.

## 23a. Phổ (mô tả, không dùng nhãn)

- Mỗi bản ghi, V1–V3: trừ trung bình, nhân cửa sổ Hann, |FFT|², chuẩn hóa tổng bằng 1, trung bình 3 chuyển đạo → 38 tần số (0–50 Hz).
- Mỗi bộ: phổ trung vị; **tỉ lệ năng lượng cao tần** HF = năng lượng ≥ 15 Hz / tổng (trung vị, IQR). Báo cáo theo lớp và cho cả 9
  chuyển đạo (độ nhạy).
- **Bộ tham chiếu** = bộ có HF trung vị (V1–V3) thấp nhất. Mỗi bộ còn lại d được làm trơn Gauss (theo thời gian, cả 9 chuyển đạo,
  `gaussian_filter1d`, mode "nearest", rồi z-score lại từng chuyển đạo) với σ_d chọn trên lưới 0, 2, 4, …, 30 ms để cực tiểu khoảng
  cách L2 giữa log phổ trung vị (5–40 Hz) của d đã làm trơn và của bộ tham chiếu. Dùng mọi bản ghi, không dùng nhãn.

## 23b. Đồng bộ dải thông lúc chấm điểm (không huấn luyện lại)

Mô hình nguồn giữ nguyên. Làm trơn **mọi nhịp tham gia** (validation nguồn và đích) theo σ của bộ chứa nó (bộ tham chiếu σ = 0),
chấm điểm lại, đặt lại ngưỡng nguồn trên validation nguồn đã làm trơn, áp lên đích đã làm trơn.

| Mã | Thước đo (trung bình 5 thiết lập hỏng, 3 seed) | Diễn giải định trước |
|---|---|---|
| A1 | Tỉ lệ khoảng cách được lấp = (BA_SRC(sau) − BA_SRC(trước)) / (BA_ORACLE(trước) − BA_SRC(trước)) | ≥ 0,20: dải thông đóng góp đáng kể; ≤ 0,05: không đáng kể; ở giữa: nhỏ |
| A2 | \|Δ\| lớp âm (trung bình logit đích − validation nguồn), trước và sau | mô tả |
| A3 | Đường liều–đáp ứng: σ nhân hệ số 0; 0,5; 1; 1,5; 2 | mô tả, kèm AUROC |

## 23c. Huấn luyện lại với dải thông đồng bộ

Làm trơn mọi bản ghi của mỗi bộ theo σ của bộ đó (như 23b), rồi huấn luyện và đánh giá y như bước 8.

| Mã | Thước đo | Diễn giải định trước |
|---|---|---|
| B1 | Số thiết lập hỏng trong 5 thiết lập gốc | ≤ 2/5: đồng bộ dải thông sửa phần lớn; 5/5 và BA_SRC đổi < 0,02: không tác dụng; còn lại: một phần |
| B2 | BA_SRC, BA_ORACLE (5 thiết lập hỏng), AUROC (9 thiết lập), trước và sau | mô tả; AUROC giảm > 0,02 thì ghi rõ là đổi lấy mất khả năng phân biệt |
| B3 | \|Δ\| lớp âm trước và sau | mô tả |

**23d (độ nhạy).** Như 23c nhưng làm trơn **cả ba bộ** với cùng σ = 20 ms (mọi bộ đều trơn, bất kể phổ gốc). Cùng thước đo B1–B3.

## Không làm

Không khử trơn (làm sắc) bộ trơn hơn: bài toán ngược, không ổn định. Không đổi con số nào đã báo cáo.

## Tệp sẽ tạo

`spectra_v23.py`, `train_v23.py`, `analyze_v23.py`, `spectra_v23.csv`, `summary_v23.json`, `fig_v23_spectra.png`, `README_vi.md`.
Điểm số từng bệnh nhân ở thư mục riêng.
