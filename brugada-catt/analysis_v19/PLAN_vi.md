# Bước 19: kế hoạch (viết và commit TRƯỚC khi chạy)

Ngày viết: 11/11/2026. Trước khi commit tệp này chỉ làm một việc trên dữ liệu: kiểm rằng chỉ số tập validation nguồn
tái lập được từ `harmonized/*.npz` (khớp `y_val`, `dom_val`, `y_tgt` ở cả 27 tệp điểm số bước 8). Chưa gom cụm, chưa tính
đại lượng nào dưới đây.

## Câu hỏi

Thầy hỏi (06/11): đặc điểm chính có giống nhau giữa các bộ dữ liệu không. Bước 17 trả lời ở mức "mô hình nhìn vào đâu".
Bước này hỏi ở mức **dạng sóng**: khi đổi viện, ngưỡng trôi vì

- (a) **tỉ lệ các dạng sóng khác nhau** giữa hai viện (thành phần ca bệnh), hay
- (b) **cùng một dạng sóng nhưng điểm số khác nhau** (đo đạc, vị trí điện cực, cách máy tạo nhịp đại diện)?

## Dữ liệu và giới hạn

- Nhịp trung vị đã chuẩn hóa (`harmonized/huca.npz`, `claris.npz`, `mortara.npz`; 356 + 1154 + 370 bản ghi).
- Điểm số bước 8 (`scores_{thiết lập}_{seed}.npz`): f = logit(`cnn_tgt`), mỏ neo `cbm5_*`, 9 thiết lập × 3 seed.
- Điểm số phía nguồn chỉ có trên **tập validation nguồn** (15%: 54–229 bản ghi; HUCA chỉ 54, trong đó 11 dương).
  Máy chạy không cài được PyTorch nên không huấn luyện lại để lấy điểm ngoài mẫu cho toàn bộ nguồn. Đây là giới hạn chính.
- Vì vậy phân tích chính dùng **lớp âm** (đủ số lượng hơn, và chỉ số độ dịch của bài cũng định nghĩa trên lớp âm).

## Gom cụm dạng sóng (không dùng nhãn)

- Đặc trưng: V1 và V2, mẫu 24–53 (−40 đến +250 ms quanh đỉnh QRS; gồm cuối QRS, điểm J, ST, đầu sóng T) → 60 chiều.
- Khớp trên mẫu cân bằng: 350 bản ghi ngẫu nhiên mỗi bộ (hạt giống 1901). PCA 8 thành phần → k-means, `n_init=20`,
  `random_state=1901`. Gán cụm cho toàn bộ 1880 bản ghi.
- **K = 5 là phân tích chính**; K = 3 và K = 8 là độ nhạy.
- Mô tả từng cụm: dạng sóng trung bình (hình), tỉ lệ Brugada theo từng bộ, và ở HUCA tỉ lệ `basal_pattern` = 1.
  Tên gọi lâm sàng của cụm (nếu đặt) là diễn giải sau, không phải bác sĩ đọc.

## Phân tích và tiêu chí

Ký hiệu cho một thiết lập nguồn S → đích T, lớp y: w_S,k và w_T,k là tỉ lệ cụm k; m_S,k và m_T,k là trung bình logit của f
trong cụm k (nguồn: tập validation; đích: toàn cohort). Nhãn đích chỉ dùng để phân tích.

**D1 (chính). Tách độ dịch trung bình logit của lớp âm.**
Δ = m_T − m_S = Σ_k (w_T,k − w_S,k)·m_S,k [thành phần] + Σ_k w_T,k·(m_T,k − m_S,k) [trong cụm].
Cụm không có bản ghi nào ở validation nguồn: dùng trung bình cả lớp m_S thay cho m_S,k, và báo cáo tỉ trọng đích bị ảnh hưởng.
Gộp trên **5 thiết lập hỏng** (0, 1, 2, 4, 6), trung bình 3 seed: tỉ phần trong cụm = Σ(trong cụm·dấu Δ) / Σ|Δ|.
KTC 95% bằng bootstrap bản ghi (1000 lần, lấy lại trong từng lớp ở validation nguồn và ở đích; cụm giữ cố định).

| Kết quả | Diễn giải định trước |
|---|---|
| tỉ phần trong cụm ≥ 2/3 (cận dưới KTC > 1/2) | cùng dạng sóng, khác điểm số: độ trôi chủ yếu do đo đạc/xử lý, không do thành phần ca |
| tỉ phần trong cụm ≤ 1/3 | chủ yếu do thành phần dạng sóng khác nhau |
| còn lại | hỗn hợp; không kết luận một phía |

Báo cáo cùng phép tách cho lớp dương (phụ, cỡ mẫu nguồn nhỏ) và cho 4 thiết lập không hỏng (đối chứng: kỳ vọng |Δ| nhỏ).

**D2. Ngưỡng phản thực tế "chỉ sửa thành phần".**
Gán trọng số cho bản ghi validation nguồn để tỉ lệ cụm trong từng lớp bằng của đích (w_T,k|y / w_S,k|y; cụm vắng ở nguồn
thì bỏ qua và chuẩn hóa lại), tính ngưỡng Youden có trọng số t_comp, rồi đo balanced accuracy trên đích.
Tỉ lệ khoảng cách được lấp = (BA(t_comp) − BA(t_SRC)) / (BA(t_ORACLE) − BA(t_SRC)), trung bình 5 thiết lập hỏng.
Định trước: ≥ 50% → thành phần giải thích phần lớn việc hỏng ngưỡng; ≤ 20% → không. So với CATT (cùng điểm số) để tham chiếu.

**D3. Thành phần cụm giữa các bộ.** Khoảng cách biến thiên toàn phần (TV) của phân bố cụm, riêng lớp âm và lớp dương.
Tiêu chí: TV(HUCA, Claris) và TV(HUCA, Mortara) đều lớn hơn TV(Claris, Mortara) ở lớp âm (khớp với bước 17: hai máy cùng
viện giống nhau hơn hai viện). KTC bootstrap.

**D4 (mô tả). Lỗi nằm ở dạng sóng nào.** Ở 5 thiết lập hỏng, tại ngưỡng nguồn: tỉ lệ dương tính giả (hoặc âm tính giả,
tùy chiều hỏng) theo cụm, và độ giàu = tỉ phần lỗi của cụm / tỉ phần bản ghi của cụm. Báo cáo cụm có độ giàu lớn nhất;
không có tiêu chí.

## Không làm trong bước này

- Đường đáp ứng của f theo từng tham số dạng sóng (phantom): cần mô hình đã huấn luyện, tức cần PyTorch. Để sang bước sau,
  chạy trên máy có PyTorch.
- Không đổi con số nào đã báo cáo.

## Tệp sẽ tạo

`cluster_v19.py` (gom cụm + D1–D4), `clusters_summary.csv`, `decomp_v19.csv`, `summary_v19.json`, `fig_v19_clusters.png`,
`README_vi.md`. Nhãn cụm từng bệnh nhân không đẩy lên kho.
