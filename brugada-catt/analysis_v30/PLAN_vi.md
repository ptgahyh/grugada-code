# Bước 30: độ dao động do huấn luyện lại (5 lần huấn luyện). Kế hoạch (viết và commit TRƯỚC khi chạy)

Ngày viết: 17/11/2026. Lúc viết: chưa chạy lần huấn luyện mới nào. Đã biết: lần 0 của mọi bài toán (dưới đây); và RBBB
CATT(A2) − SRC = −0,008 (11/11) so với +0,007 (16/11) với KTC bootstrap bệnh nhân không chồng nhau (quan sát sau, bước 28).

## Câu hỏi

KTC 95% bootstrap theo bệnh nhân giữ mô hình cố định, nên không tính độ dao động do huấn luyện lại mô hình. (1) Độ dao động này
lớn cỡ nào so với KTC bootstrap? (2) Kết luận nào về CATT còn đứng khi tính cả nó?

## Thiết kế

- **Lần huấn luyện r = 0..4.** Lần r cộng 1000·r vào hạt giống của **mọi thành phần được huấn luyện**: `cbm_lib4.train_cnn`
  (mô hình đích f và CNNalt), `cbm_lib4.train_cbm2` (mỏ neo CBM của Brugada), `v8_core.train_uda` (DANN, CORAL). Biến môi trường
  `SEED_OFFSET`; r = 0 trùng mã cũ. **Không đổi:** dữ liệu, cách chia validation (seed validation 0–2), giới hạn 12.000 bản ghi,
  mỏ neo không huấn luyện (RBBB A1, A2; LVH A1–A3 là hồi quy logistic tất định), mọi phân tích.
- **r = 0 là lần đã có:** RBBB, LVH, LVH mV: lần chạy 16/11 trên máy tác giả (dữ liệu bản 3). Brugada: pipeline lọc chung bước 27
  (đã tái lập trùng tuyệt đối ở bước 29). Lần 11/11 của RBBB/LVH (dữ liệu và máy khác) chỉ báo cáo bên cạnh, không gộp.
- Brugada dùng pipeline lọc chung (đề xuất làm pipeline chính, bước 29); nếu tác giả chọn pipeline gốc thì chạy lại bước này cho nó.
- Chạy: RBBB/LVH/LVH mV trên máy tác giả (`chay_nhieu_hat_giong.py`), Brugada trên máy cloud (`run30_brugada.sh`). Mỗi lần dùng
  đúng phân tích P4 (bước 14/15) và `v8_analyze.py` (Brugada), gồm bootstrap bệnh nhân 1000 lần.

## Đại lượng chính (định trước)

Δ = hiệu BA gộp, CATT với mỏ neo chính trừ phương pháp so sánh:

| Bài toán | CATT | Gộp trên | So với |
|---|---|---|---|
| RBBB | CATT(A2) | 6 thiết lập | SRC, GMM-2, Otsu |
| LVH | CATT(A3) | 6 thiết lập | SRC, GMM-2, Otsu |
| LVH mV | CATT(A3) | (a) 2 thiết lập hỏng **của lần 0** (cn→PTB-CS, cn→PTB-AT); (b) 4 còn lại; (c) cả 6 | SRC, GMM-2, Otsu |
| Brugada | CATT (CBM-sd15) | (a) "held-out" như bước 8/27; (b) tỉ lệ bệnh 5% | SRC, GMM-2, Otsu |

Với mỗi Δ: trung bình 5 lần (Δ̄), SD giữa các lần (s), nửa độ rộng KTC bootstrap trung vị qua 5 lần (h), tỉ số s / (h / 1,96),
số lần cùng dấu với Δ̄, và khoảng t 95% của Δ̄ qua 5 lần: Δ̄ ± 2,776·s/√5.

## Tiêu chí và quy tắc diễn giải (định trước)

| Mã | Nội dung | Quy tắc |
|---|---|---|
| V1 | Bootstrap có đánh giá thấp độ không chắc chắn không | "Có", nếu s > h/1,96 (SD giữa các lần lớn hơn SE bootstrap) ở Δ CATT − SRC của bài toán đó |
| V2 | **Quy tắc báo cáo trong bài** | Một Δ được viết là "CATT tốt hơn/kém hơn" chỉ khi khoảng t 95% không chứa 0 **và** ≥ 4/5 lần cùng dấu. Ngược lại viết "không phân biệt được" |
| V3 | Kết luận chính của Brugada có đứng không | Đứng nếu Δ(CATT − SRC, held-out) theo V2 là "tốt hơn" |
| V4 | Số thiết lập hỏng có ổn định không (SRC thấp hơn ORACLE > 0,05, trung bình 3 seed validation) | mô tả: số thiết lập hỏng ở từng lần, mọi bài toán |

Dự đoán trước khi chạy (để đối chiếu, không phải tiêu chí): V1 "có" ở RBBB; Brugada và LVH mV (thiết lập hỏng) đứng vững vì hiệu
lớn (≈ 0,10–0,13) so với độ dao động; các hiệu nhỏ (|Δ| ≤ 0,01, như CATT so với GMM-2/Otsu ở tỉ lệ tự nhiên) có thể thành
"không phân biệt được".

## Kiểm tra an toàn trước khi tin số

- Lần r ≥ 1 phải **khác** lần 0 (nếu điểm số f trùng tuyệt đối thì `SEED_OFFSET` không có tác dụng: dừng, sửa).
- Nhãn y, kích thước đích, cách chia validation phải trùng lần 0 ở mọi tệp điểm.

Mọi kết quả đều báo cáo, kể cả khi kết luận chính không đứng. Điểm số từng bản ghi không đẩy lên repo.

## Tệp sẽ tạo

`run30_brugada.sh`, `chay_nhieu_hat_giong.py`, `aggregate30.py`, `results/` (bảng tổng hợp), `summary_v30.json`, `README_vi.md`.
