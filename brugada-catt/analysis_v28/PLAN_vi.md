# Bước 28: ngưỡng làm trơn, quy tắc cỡ mẫu (hệ số an toàn 1,5) và cổng trên RBBB/LVH. Kế hoạch (viết và commit TRƯỚC khi chạy)

Ngày viết: 15/11/2026. Lúc viết, bộ ecg-arrhythmia 1.0.0 mới chuyển lên được một phần (chưa ghép, chưa chuẩn hóa), chưa có điểm số
bước 14–15 trên máy này, chưa chạy phép tính nào của bước này. Các con số đã biết khi viết: bảng tổng hợp bước 14–15 trong repo
(P1–P4 của ngưỡng thường) và kết quả bước 20, 24–26 trên Brugada.

## Dữ liệu

Điểm số chạy lại của bước 14 (RBBB) và bước 15 (LVH): 6 thiết lập × 3 seed validation mỗi bài toán, đúng script cũ
(`run_rbbb.py`, `run_lvh.py`). f = logit của `f_tgt` (như `common15.logit`). Nhãn giả = mỏ neo ≥ ngưỡng Youden của mỏ neo trên
validation nguồn. **Mỏ neo chính:** RBBB A2 (RBBB-LR), LVH A3, tức hai mỏ neo đã cho con số −0,008 và −0,010 trong README bước
14–15. (Kế hoạch bước 26 ghi "mỏ neo chính bước 14" mà không nêu tên; ghi rõ ở đây trước khi chạy.) Các mỏ neo còn lại: mô tả.
Trước hết kiểm việc chạy lại: số bản ghi từng miền và bảng P4 phải khớp bản đã commit (sai khác BA ≤ 0,005 mỗi ô); nếu không
khớp thì báo cáo và dùng điểm số mới cho mọi phần dưới.

Ngưỡng làm trơn t̂_h, độ rộng h, ước lượng G2: đúng `analysis_v20/smooth_lib.py` và `analysis_v24/lib24.py`, không sửa.
Hạt giống 2801.

## 28a. Độ dao động của ngưỡng làm trơn (mỏ neo chính; các mỏ neo khác mô tả)

Mỗi (thiết lập, seed), n ∈ {500, 1000, 2000}: 300 mẫu con không hoàn lại từ miền đích; SD quan sát của t̂_h và của ngưỡng thường t̂.
Với 20 mẫu con mỗi n: G2 (sandwich, không nhãn) × √(1 − n/N). Bỏ mẫu con có < 5 nhãn giả ở một lớp (đếm và báo cáo).

| Mã | Nội dung | Đạt khi (từng bài toán) |
|---|---|---|
| R1 | Công thức G2 từng mẫu đơn so với SD quan sát | trung vị tỉ số ∈ [0,80; 1,25] **và** trung vị \|log tỉ số\| ≤ 0,20 (như T5 bước 24) |
| R2 | Làm trơn có giảm dao động không: SD(t̂_h)/SD(t̂) | trung vị qua các ô ≤ 0,80 |
| R3 (mô tả) | Tốc độ: hệ số góc log SD theo log n của t̂_h và t̂ | bảng |

## 28b. Quy tắc cỡ mẫu một giai đoạn với hệ số an toàn cố định 1,5

Mẫu thử m = 300 (không nhãn); ε ∈ {0,15; 0,30} (logit); 6 mẫu thử mỗi (thiết lập, seed), 120 lần thu mỗi kế hoạch.
n_∞ = **1,5** × [R·Ṽ·m_q^(1/5)/(c·Ã²·ε²)]^(5/4) (`n_needed_smooth`, Ṽ và Ã từ KDE Scott trên mẫu thử, m_q = min(q, 1 − q));
n_plan = max(300, ⌈n_∞/(1 + n_∞/N)⌉). Kế hoạch > 0,8N: "không khả thi". SD đạt được = SD của t̂_h qua 120 mẫu con cỡ n_plan.
Hệ số 1,5 lấy từ mô tả sau của bước 24 (mô phỏng), **không chỉnh** trên dữ liệu này.

| Mã | Nội dung | Đạt khi |
|---|---|---|
| C1 | Tỉ lệ kế hoạch khả thi đạt SD ≤ 1,2ε, mỏ neo chính | ≥ 80% ở cả hai ε, **từng bài toán** (RBBB, LVH) |
| C2 (mô tả) | Như C1 không có hệ số an toàn (×1,0); hệ số cần cho 80%; tỉ lệ khả thi; trung vị n_plan so với quy tắc ngưỡng thường (bước 14–15) | bảng |

## 28c. Cổng trên RBBB/LVH (tiêu chí G3, G4 đã đăng ký ở kế hoạch bước 26, commit 474f20c; không đổi)

t_gate = t nếu |t − t_src| > z·σ̂, ngược lại t_src; z = 2 (chính), 1 và 3 (độ nhạy); bản thường và bản làm trơn, trên toàn cohort đích.
G3: BA_gate − BA_src gộp 6 thiết lập, bootstrap bệnh nhân 1000 lần phân tầng theo nhãn (như P4), đạt khi ước lượng điểm ≥ −0,003
cho bản làm trơn với mỏ neo chính, từng bài toán. Trong mỗi lần bootstrap, ngưỡng và σ̂ được tính lại trên mẫu bootstrap.
G4 (mô tả): số lần cổng mở trên 18; phần hại của CATT được tránh; |t − t_src|/σ̂ từng thiết lập.

## 28d. Bước 22 (LVH, f nhận mV)

Chạy đúng `analysis_v22/PLAN_vi.md` (M1–M5, không đổi). Nếu M2 > 0 (có thiết lập hỏng): chạy thêm 28a–28c trên điểm số mV, ghi là mô tả.

## 28e (mô tả). BA của CATT làm trơn

Toàn cohort đích: BA của SRC, CATT, CATT làm trơn, ORACLE (mỏ neo chính), hiệu gộp kèm KTC bootstrap. Với n = 500 không nhãn:
BA trung bình và phân vị 10% của CATT thường và làm trơn (300 mẫu con).

## Diễn giải định trước

- R1 và C1 đạt ở cả hai bài toán: "Hệ quả 3" (ngưỡng làm trơn + quy tắc cỡ mẫu ×1,5) vào bài như kết quả đã kiểm ngoài Brugada.
- R1 đạt, C1 không: công thức SD dùng để **báo cáo** độ chính xác; quy tắc cỡ mẫu ghi là hạn chế, kèm hệ số thực tế cần.
- R1 không đạt: ngưỡng làm trơn chỉ trình bày trên Brugada và mô phỏng, nêu rõ không lặp lại được.
- G3 đạt: đề xuất "CATT có cổng" làm quy trình mặc định. Không đạt: báo cáo là CATT cần mỏ neo độc lập, cổng không cứu được.

Mọi kết quả đều báo cáo, đạt hay không. Điểm số từng bản ghi không đẩy lên repo.

## Tệp sẽ tạo

`run28.py` (28a, 28b, 28c, 28e), các `.csv` tổng hợp, `summary_v28.json`, `README_vi.md`; `analysis_v22/results/`.
