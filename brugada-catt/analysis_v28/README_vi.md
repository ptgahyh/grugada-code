# Bước 28: RBBB/LVH chạy lại, ngưỡng làm trơn, cỡ mẫu, cổng. Kết quả

Kế hoạch: `PLAN_vi.md` (commit trước khi có dữ liệu). Chạy 16–17/11/2026 trên máy của tác giả: `python chay_rbbb_lvh.py`
(16 nhân, RAM 7 GB, 3 tiến trình; khoảng 3,7 giờ).

## Dữ liệu

PTB-XL 1.0.3: khớp đúng số cũ (RBBB: CS 12.403/297, AT 8.278/244; LVH: CS 12.843/941, AT 8.352/810).

ecg-arrhythmia 1.0.0 lấy từ **bản sao HuggingFace**, không phải PhysioNet: zip có 89.960 header cho 45.152 mã bản ghi (gần như mọi
bản ghi lặp hai lần) và một số tệp `.mat` hỏng. Script chuẩn hóa (bản 3) gom mọi bản sao của một mã bản ghi và thử lần lượt:

| | Số |
|---|---|
| Mã bản ghi | 45.152 (đúng bản gốc) |
| Đọc được ở bản sao đầu | 43.073 |
| Phải dùng bản sao khác | 2.054 |
| Hỏng ở mọi bản sao | 24 |
| Header sai định dạng | 1 (bản gốc: 2) |
| RBBB: số bản ghi dùng (dương tính) | **43.380 (1.733)**; lần 11/11: 43.403 (1.734) |
| LVH: số bản ghi dùng (dương tính) | **38.221 (637)**; lần 11/11: 38.242 (637) |

Mất 23 bản ghi RBBB (1 dương tính) và 21 bản ghi LVH (0 dương tính) so với bản PhysioNet. Ghi là hạn chế nhỏ.

## Kiểm việc chạy lại bước 14–15 (tiêu chí của PLAN: P4 lệch ≤ 0,005 mỗi ô)

**Không đạt.** RBBB: 25/54 ô lệch > 0,005; LVH: 27/60 ô. ORACLE khớp (lệch ≤ 0,004), nên dữ liệu đích gần như trùng; phần lệch đến
từ việc huấn luyện lại mô hình (máy, phiên bản PyTorch, số tiến trình khác; 23 bản ghi ít hơn). Theo kế hoạch, mọi phần sau dùng
điểm số mới.

| Hiệu gộp (6 thiết lập, KTC 95% bootstrap bệnh nhân) | Lần 11/11 | Lần 16/11 |
|---|---|---|
| RBBB: CATT(A2) − SRC | −0,008 (−0,013; −0,005) | **+0,007 (0,004; 0,010)** |
| RBBB: CATT(A2) − GMM-2 | −0,008 | +0,005 |
| RBBB: CATT(A2) − Otsu | −0,023 | −0,006 |
| LVH: CATT(A3) − SRC | −0,010 (−0,012; −0,005) | −0,010 (−0,013; −0,004) |
| LVH: CATT(A3) − GMM-2 | +0,039 | +0,033 |
| LVH: CATT(A3) − Otsu | +0,046 | +0,039 |

**Phát hiện quan trọng (quan sát sau):** với RBBB, dấu của CATT(A2) − SRC **đổi chiều** giữa hai lần huấn luyện, dù cả hai KTC đều
không chứa 0. KTC bootstrap bệnh nhân không tính độ dao động do huấn luyện lại mô hình. Phần lệch nằm ở hai thiết lập cn→PTB
(CATT(A2): 0,925 → 0,964 và 0,927 → 0,968). Hệ quả cho bài: hiệu nhỏ cỡ ±0,01 trên RBBB **không được diễn giải là có hay không có
lợi**; muốn kết luận cần nhiều lần huấn luyện (ví dụ 5 hạt giống huấn luyện) và báo cáo độ dao động giữa các lần. LVH ổn định.

Định lý 1 (P1) giữ: RBBB dự đoán/quan sát 0,81 (cũ 0,80), Spearman 0,80 (cũ 0,87); LVH 1,07 (cũ 1,06), Spearman 0,99.
Quy tắc dừng (P3): RBBB vẫn không đạt (48–62% đạt đích ±20%); LVH bản một giai đoạn tụt (77% so với 83–87%), hai giai đoạn
81–87% (cũ 89–90%).

## 28a–28e: kết quả theo tiêu chí định trước

Mỏ neo chính: RBBB A2 (RBBB-LR), LVH A3 (LVH-LR). "LVH mV" = điểm số bước 22 (mô tả, theo 28d vì M2 = 2 > 0).

| Mã | Tiêu chí | RBBB | LVH | LVH mV |
|---|---|---|---|---|
| R1 | G2 từng mẫu đơn: trung vị tỉ số ∈ [0,80; 1,25] và trung vị \|log\| ≤ 0,20 | 0,22 / 1,53 **không đạt** | 0,94 / 0,17 **đạt** | 0,97 / 0,12 đạt |
| R2 | SD ngưỡng làm trơn / thường ≤ 0,80 | 0,78 **đạt** | 0,53 **đạt** | 0,46 đạt |
| R3 | Hệ số góc log SD theo log n: làm trơn / thường | −0,47 / −0,45 | −0,43 / −0,33 | −0,47 / −0,33 |
| C1 | Quy tắc cỡ mẫu ×1,5: tỉ lệ kế hoạch đạt SD ≤ 1,2ε (ε = 0,15 / 0,30) ≥ 80% | 58% / 47% **không đạt** | 87% / 98% **đạt** | 98% / 98% đạt |
| C2 | Như C1 không hệ số an toàn | 48% / 39% | 81% / 93% | 93% / 94% |
| G3 | Cổng z = 2 (làm trơn) − SRC, gộp 6 thiết lập, ≥ −0,003 | +0,008 (0,005; 0,010) **đạt** | −0,008 (−0,012; −0,005) **không đạt** | +0,019 (0,015; 0,022) đạt |
| G4 | Số lần cổng mở (trên 18) | 16 | 15 | 18 |

28e (mô tả, mỏ neo chính, n = 500 ECG không nhãn): BA trung bình / phân vị 10% của CATT thường so với làm trơn:
RBBB 0,965 / 0,939 so với 0,973 / 0,964; LVH 0,801 / 0,778 so với 0,804 / 0,791; LVH mV 0,858 / 0,834 so với 0,861 / 0,850.
Trên toàn cohort, làm trơn và thường ngang nhau (hiệu −0,003 đến +0,003, KTC chứa 0 ở cả ba).

## Diễn giải (theo bảng định trước của PLAN, kết quả hỗn hợp nên ghi từng bài toán)

1. **Ngưỡng làm trơn giảm dao động ở cả ba bài toán (R2)** và giúp rõ nhất ở phân vị thấp với 500 ECG. Phần này lặp lại được ngoài Brugada.
2. **Công thức SD (G2) và quy tắc cỡ mẫu đúng trên LVH, sai rất xa trên RBBB.** Theo diễn giải định trước: công thức chỉ dùng được
   khi điều kiện của nó thỏa; không trình bày như kết quả chung.
   *Quan sát sau (chưa kiểm):* RBBB hiếm (2,4–4%) và mô hình tách rất tốt (BA 0,97). Với n = 500 chỉ có khoảng 10–20 nhãn giả dương,
   SD quan sát rất lớn (1,0–1,9 logit) và tỉ số G2/quan sát tăng dần theo n (ví dụ cn→PTB-CS: 0,31 → 0,61 → 0,81 ở n = 500, 1000, 2000).
   Giả thuyết: G2 là xấp xỉ tiệm cận, hỏng khi số ca dương trong mẫu quá ít. Nếu đúng thì quy tắc cần thêm điều kiện "đủ số ca dương
   giả" (cần kế hoạch mới để kiểm).
3. **Cổng không cứu được LVH (G3 không đạt).** Theo diễn giải định trước: "CATT cần mỏ neo độc lập; cổng không cứu được".
   Lý do thấy ngay trong số liệu: với hàng nghìn ECG, ngưỡng CATT lệch khỏi ngưỡng nguồn 3–37 SD, nên cổng gần như luôn mở. Cổng chỉ
   kiểm *độ dao động lấy mẫu*; nó không thấy *độ lệch do mỏ neo* (Mệnh đề 2). Ở PTB→cn, CATT kéo ngưỡng sai hướng mà vẫn "có ý nghĩa".
   Trên Brugada cổng có tác dụng vì đích nhỏ (vài trăm ECG) và thiết lập không hỏng chỉ lệch 1,2–1,7 SD.
4. **LVH mV:** cổng mở 18/18, nên kết quả G3 (+0,019) chính là CATT − SRC: lợi ở 2 thiết lập hỏng lớn hơn hại ở 4 thiết lập còn lại.
5. **RBBB G3 đạt nhưng không mang thông tin:** CATT − SRC đổi dấu giữa hai lần huấn luyện (xem trên).

Hệ quả cho bài: phần đóng góp "lập kế hoạch được" phải nói rõ phạm vi. Công thức SD + cỡ mẫu: đúng ở Brugada (bước 24, 29), LVH,
LVH mV; sai ở RBBB (hiếm, tách quá tốt). Cổng: chỉ hữu ích khi đích nhỏ; với đích lớn cần thêm cách phát hiện lệch do mỏ neo.

## Tệp

`rerun_v14/`, `rerun_v15/`: số liệu tổng hợp của lần chạy lại (cùng định dạng `analysis_v14/results`, `analysis_v15/results`).
`results/`: đầu ra 28a–28e (`a_spread_*`, `b_plans_*`, `c_gate_*`, `c_boot_*` là bootstrap gộp theo thiết lập, `summary_v28_*.json`).
Bước 22: `analysis_v22/results/`, `analysis_v22/README_vi.md`.
