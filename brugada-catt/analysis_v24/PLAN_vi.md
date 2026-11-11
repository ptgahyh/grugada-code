# Bước 24: công thức độ dao động và cỡ mẫu cho ngưỡng CATT làm trơn. Kế hoạch (viết và commit TRƯỚC khi chạy)

Ngày viết: 11/11/2026. Chưa chạy phép tính nào của bước này. Ngưỡng làm trơn t̂_h như bước 20 (`analysis_v20/smooth_lib.py`,
độ rộng h = 0,9·min(SD, IQR/1,34)·n_min^(−1/5)).

## Kết quả lý thuyết cần kiểm (khai triển delta; không phải định lý mới)

J_h có đạo hàm −D̂_h, với D̂_h(t) = p̂₁,h(t) − p̂₀,h(t) là hiệu hai mật độ kernel Gauss độ rộng h của f trong hai lớp nhãn giả.
Vậy t̂_h là nghiệm của D̂_h = 0. Gọi t_h là nghiệm quần thể, φ_h(u) = φ(u/h)/h.

- **Dạng chính xác (h cố định):** ảnh hưởng của bản ghi i: ψ_i = (ỹ_i/q)(φ_h(f_i − t_h) − m₁) − ((1 − ỹ_i)/(1 − q))(φ_h(f_i − t_h) − m₀),
  m_k = E_k[φ_h(f − t_h)]. Khi đó SD(t̂_h) ≈ √(Var ψ / n) / |D_h′(t_h)|, với Var ψ = v₁/q + v₀/(1 − q), v_k = Var_k[φ_h(f − t_h)].
- **Dạng đóng (h nhỏ):** v_k ≈ p_k(t)·R/h, R = 1/(2√π) ≈ 0,282. Thu được

  SD(t̂_h) ≈ √(R·Ṽ / (n·h)) / Ã,

  **cùng hai hằng số Ṽ, Ã của Định lý 1**, nên vẫn ước lượng được không cần nhãn.
- **Quy tắc cỡ mẫu:** với h = c·(n·m)^(−1/5), c = 0,9·min(SD_f, IQR_f/1,34), m = min(q, 1 − q):
  muốn SD ≤ ε cần n ≥ [R·Ṽ·m^(1/5) / (c·Ã²·ε²)]^(5/4). So với ngưỡng thường: n ≥ 0,54·Ṽ/(Ã²·ε³).
- **Hệ số quần thể hữu hạn:** t̂_h tuyến tính tiệm cận nên lý thuyết dự đoán số mũ cổ điển: SD nhân √(1 − n/N) (số mũ 1/2),
  khác ngưỡng thường (bước 18a: 0,24).
- Độ lệch t_h − t̃ bậc h² không nằm trong SD; báo cáo riêng.

## Mô phỏng (mô hình Gauss của bài, 4 cấu hình mỏ neo; hạt giống 2401–2404)

**T1. Công thức với đại lượng quần thể.** n ∈ {100, 300, 1000, 3000}; SD quan sát qua 2000 mẫu mỗi ô. Dự đoán: dạng chính xác,
dùng mật độ quần thể của hai lớp nhãn giả (tích phân số trên lưới), h = trung vị các h mẫu của ô, t_h là nghiệm quần thể với h đó.
Đạt khi: trung vị dự đoán/quan sát qua 16 ô ∈ [0,90; 1,10] **và** trung vị theo từng n ∈ [0,85; 1,15]. Dạng đóng: mô tả.

**T2. Ước lượng không nhãn từ một mẫu** (200 mẫu mỗi ô, so với SD quan sát của T1):

| Mã | Ước lượng |
|---|---|
| G1 | Dạng đóng, Ṽ và Ã từ KDE Scott (như E0 bước 18), h của mẫu |
| G2 | Sandwich cùng h: SD = sd(ψ̂)/√n / \|D̂_h′(t̂_h)\|, tất cả tính trên mẫu với cùng h |
| G3 | Lai: tử số của G2, mẫu số Ã của G1 |
| G4 | Bootstrap thường (B = 200, h cố định), như S1d bước 20 |

Đạt khi: ước lượng tốt nhất trong G1–G3 có trung vị \|log tỉ số\| ≤ 0,15 **và** trung vị tỉ số theo từng n ∈ [0,80; 1,25].
Đối chiếu: ngưỡng thường, bootstrap số 0,125 (bước 18); F1 0,244.

**T3. Hệ số quần thể hữu hạn.** Như bước 18a (N ∈ {400, 1200, 4000}, n/N ∈ {0,1; 0,2; 0,35; 0,5; 0,8}), 100 kho × 200 mẫu con,
SD_iid qua 2000 mẫu. Khớp số mũ γ của (1 − n/N). Đạt khi γ ∈ [0,40; 0,60].

**T4. Lập kế hoạch cỡ mẫu.** Mẫu thử m ∈ {100, 300}, 200 lần mỗi cấu hình; ước lượng Ṽ, Ã (KDE Scott), c, q từ mẫu thử; tính
n_plan cho ε ∈ {0,15; 0,30} (logit). SD thật tại n_plan: nội suy log–log từ bảng mô phỏng (30 giá trị n từ 50 đến 30 000, 1000 lần).
Đạt khi: với m = 300, tỉ lệ lần đạt SD_thật(n_plan) ≤ 1,2ε ≥ 80% ở cả hai ε. m = 100: mô tả.
Mô tả thêm: tỉ số n_plan(làm trơn)/n_plan(thường) với hằng số quần thể, cho từng cấu hình và ε; độ lệch tại n_plan.

## Brugada (điểm số bước 8, mỏ neo CBM-sd15, 9 thiết lập × 3 seed; hạt giống 2405)

**T5.** SD quan sát của t̂_h qua 300 mẫu con không hoàn lại, n = 20/35/50% cohort đích. Với 20 mẫu con mỗi n: G1–G3 nhân
√(1 − n/N). Đạt khi: ước lượng tốt nhất trong G1–G3 có trung vị tỉ số từng mẫu đơn ∈ [0,80; 1,25] **và** trung vị \|log\| ≤ 0,20
(đối chiếu bước 18c, ngưỡng thường: F1 0,231; tốt nhất 0,207).
**T6 (mô tả).** Trên toàn cohort đích (không nhãn): n cần để SD ≤ 0,15 và ≤ 0,30, ngưỡng làm trơn so với ngưỡng thường.

## Diễn giải định trước

- T1, T2, T5 đạt: thêm "Hệ quả 3" (ngưỡng làm trơn, công thức SD và cỡ mẫu) vào bài, ghi rõ là khai triển delta chuẩn của ước
  lượng kernel, không phải định lý mới; trích Horowitz (1992) cho ý tưởng làm trơn.
- T1 đạt nhưng T2/T5 không: công thức đúng nhưng ước lượng không nhãn chưa đủ tốt; báo cáo như hạn chế.
- T1 không đạt: không đưa công thức vào bài.

## Tệp sẽ tạo

`theory24.py` (đại lượng quần thể), `sim24.py`, `fpc24.py`, `plan24.py`, `brugada24.py`, các `.csv` tổng hợp, `summary_v24.json`,
`README_vi.md`.
