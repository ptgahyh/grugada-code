# Bước 13: sửa công thức cho đúng hơn (Mệnh đề 5) và kiểm lại sau phản biện

**Vấn đề của bước 11–12:** hai công thức tách rời nhau.
- Mệnh đề 1 cho thiên lệch, đúng với mọi mỏ neo.
- Mệnh đề 3 (n_eff) cho nhiễu lấy mẫu, nhưng chỉ đúng khi mỏ neo độc lập có điều kiện.

Bước 12 chỉ ra thêm hai điểm yếu:
- Với mỏ neo phụ thuộc, n_eff dự đoán sai.
- Khi n nhỏ, công thức thận trọng quá 1.5–2 lần.

**Mệnh đề 5** sửa cả hai. Phát biểu đầy đủ, điều kiện và khung chứng minh nằm trong `MENH_DE_5_vi.md`. Tóm tắt:
- **Định lý 5a (đã chứng minh, theo Kim–Pollard 1990):** t̂ − t̃ ≈ a·Z, với Z ~ Chernoff và a = (4Ṽ/(Ã²n))^{1/3}.
  - Ṽ = p̃₁(t̃)/q + p̃₀(t̃)/(1−q).
  - Ã = |J̃″(t̃)|.
  - Khi mỏ neo độc lập, công thức quay về đúng luật n_eff.
- **Xấp xỉ 5b (heuristic):** tổn thất ≈ E_Z[BA(t\*) − BA(t̃ + aZ)].
- **Hệ quả 5c:** t̃, Ṽ, Ã không cần nhãn đích. Vì vậy nguồn mới tính được độ bất định của ngưỡng trước khi có nhãn. Quy tắc dừng: n ≥ 0.54·Ṽ/(Ã²ε³).

## 1. Mô phỏng Gauss (`unified_sim.py`)

64 ô × 500 lần. So trung vị dự đoán/quan sát của tổn thất giữa công thức mới và cũ (Mệnh đề 1 + Mệnh đề 3):

| | Toàn bộ | ρ = 0.9 | n = 100 |
|---|---|---|---|
| Mới | 1.03 | 1.00 | 1.09 |
| Cũ | 1.28 | 1.41 | 1.79 |

R²(log): mới 0.993, cũ 0.933.

## 2. Sau phản biện: những gì đã sửa

| Nhận xét | Đã làm | Kết quả |
|---|---|---|
| "Định lý" chỉ đúng một nửa | Tách thành Định lý 5a, Xấp xỉ 5b, Hệ quả 5c | `MENH_DE_5_vi.md` |
| Kiểm trên Brugada chủ yếu đo thiên lệch | **Kiểm trực tiếp độ phân tán của t̂**, không dính thiên lệch hay đường BA | Mục 3 |
| Lệch tâm t_kde / t_c, BA lởm chởm | Dùng cùng tâm t̃ (KDE) và cùng đường BA đã làm trơn (`brugada_unified_v2.py`) | Mục 4 |
| Giải thích ST_V2 sai | Rút lại. KDE tính trên f, không trên ST_V2 | Mục 3 |
| Độ nhạy băng thông | Chạy KDE ×0.5, ×1, ×2 | Mục 5 |
| Mô phỏng chỉ trong mô hình Gauss | Thêm phân phối lệch, đuôi dày, hai đỉnh, điểm số rời rạc (`nongauss_sim.py`) | Mục 6 |
| Var Chernoff 0.271 | Lưới mịn gấp 8 lần, 40.000 mẫu (`chernoff.py`) | 0.261 ± 0.002, khớp lý thuyết 0.2636 |
| Hiệu chỉnh quần thể hữu hạn là heuristic | Ghi rõ trong `MENH_DE_5_vi.md` | |

Kết quả Brugada của bản 1 (`brugada_unified.py`: tỉ số 1.07, Spearman 0.94) **không còn dùng làm bằng chứng cho Mệnh đề 5**. Như phản biện chỉ ra, phần lớn con số đó đến từ thiên lệch tính bằng nhãn.

## 3. Kiểm trực tiếp Định lý 5a trên Brugada: độ phân tán của ngưỡng CATT

**Thiết kế:**
- 9 thiết lập × 3 seed × 7 mỏ neo × 3 cỡ mẫu (20/35/50% cohort) = 561 điểm.
- Mỗi điểm: 300 lần lấy mẫu con, đo SD của t̂, so với a·SD(Z).

| | Dự đoán/quan sát (trung vị) | Khoảng tứ phân vị của tỉ số | Spearman |
|---|---|---|---|
| a tính trên cả cohort | **1.09** | 0.96–1.26 | **0.91** |
| Mỏ neo khái niệm | 1.10 | 0.95–1.30 | 0.88 |
| Mỏ neo hộp đen | 1.09 | 0.96–1.22 | 0.93 |
| **Ước lượng không cần nhãn** (Hệ quả 5c; a tính từ đúng 1 mẫu con không nhãn) | **1.08** | 0.93–1.31 | **0.86** |

**Theo mỏ neo (a tính trên cả cohort):**

| Mỏ neo | CBM | CBM-sd15 | ResNet | LogReg9 | CNN9 | CNNalt | ST_V2 |
|---|---|---|---|---|---|---|---|
| Dự đoán/quan sát | 1.07 | 1.06 | 1.04 | 1.08 | 1.11 | 1.12 | 1.17 |

- Theo cỡ mẫu (20/35/50% cohort): 1.08 / 1.09 / 1.10.
- 70% số điểm có tỉ số nằm trong khoảng 0.8–1.25.

**Đọc kết quả:**
- **Lõi của Mệnh đề 5 đúng trên dữ liệu thật**, cho cả mỏ neo phụ thuộc lẫn mỏ neo khái niệm. Công thức dự đoán dư khoảng 9%, cùng chiều với mô phỏng ở n nhỏ.
- **Hệ quả 5c dùng được:** chỉ với f và nhãn giả của một mẫu không nhãn, độ bất định của ngưỡng được ước lượng sai số trung vị 8%.
- **ST_V2:** dự đoán dư nhiều nhất (1.17). Nguyên nhân chưa xác định; giải thích "biến rời rạc" của bản trước là sai và đã rút lại.

## 4. Tổn thất trên Brugada (cùng tâm, đường BA làm trơn)

Thông số Scott ×1:

| | Toàn bộ | Khái niệm | Hộp đen |
|---|---|---|---|
| Tổn thất: dự đoán/quan sát | 0.90 | 0.98 | 0.88 |
| Spearman tổn thất | 0.96 | 0.88 | 0.98 |
| Tỉ trọng thiên lệch trong tổn thất quan sát | 60% | 46% | 70% |
| Spearman phần lấy mẫu (dự đoán vs quan sát) | 0.19 | **0.68** | −0.03 |
| Số điểm phần lấy mẫu dự đoán âm | 77 | 0 | 77 |

**Đọc kết quả:**
- Sau khi sửa lỗi lệch tâm, mỏ neo khái niệm hết hẳn trường hợp phần lấy mẫu âm. Phần lấy mẫu ở nhóm này khớp khá (Spearman 0.68).
- Ở **hộp đen**, thiên lệch chiếm 70% tổn thất. t̃ nằm xa t\*, trên đoạn đường BA cong lồi. Ở đó nhiễu quanh t̃ có thể làm tổn thất giảm, nên phần lấy mẫu có thể âm thật: quan sát cũng thấy 33 điểm âm.
  - Phần lấy mẫu là hiệu của hai số lớn gần nhau, nên tổn thất **không phải** phép thử tốt cho Mệnh đề 5 ở nhóm này.
  - Phép thử đúng là độ phân tán ở mục 3.
- Ở các điểm thiên lệch chiếm dưới một nửa tổn thất (233 điểm), phần lấy mẫu khớp Spearman 0.72.
- **Kết luận:** Xấp xỉ 5b dùng được khi nhiễu chiếm phần đáng kể. Khi thiên lệch áp đảo thì tổn thất do thiên lệch quyết định, đó là việc của Mệnh đề 1–2.

## 5. Độ nhạy theo băng thông KDE

| Băng thông (× Scott) | 0.5 | 1 | 2 |
|---|---|---|---|
| Độ phân tán: dự đoán/quan sát | 0.91 | 1.09 | 1.30 |
| Tổn thất: dự đoán/quan sát | 0.91 | 0.90 | 0.90 |

- Độ phân tán nhạy vừa phải với băng thông: đổi băng thông 4 lần thì dự đoán thay đổi khoảng ±20%.
- Theo lý thuyết, a ∝ Ã^{−2/3}, và Ã (độ cong) bị làm phẳng khi băng thông lớn.
- Quy tắc Scott nằm giữa và cho sai số khoảng 9%.
- Khi viết bài phải nêu rõ quy tắc chọn băng thông và báo cáo độ nhạy này.

## 6. Ngoài mô hình Gauss (`nongauss_sim.py`)

Mỗi kịch bản có 16 ô (ρ ∈ {0, 0.6}, độ lệch mỏ neo ∈ {0, −1.25}, n ∈ {100, 300, 1000, 3000}), 400 lần mỗi ô.

| Kịch bản | Độ phân tán: dự đoán/quan sát (khoảng) | Tổn thất: dự đoán/quan sát (khoảng) |
|---|---|---|
| Lệch (Gamma) | 1.06 (0.91–1.11) | 1.08 (0.94–1.24) |
| Đuôi dày (Student t₃) | 1.03 (0.91–1.13) | 1.00 (0.86–1.34) |
| Hai đỉnh | 1.01 (0.92–1.10) | 1.04 (0.97–1.14) |
| Rời rạc (bước 0.25 ≈ 0.2 SD) | 1.03 (0.89–1.16) | 1.07 (1.00–1.21) |

- Công thức giữ đúng ngoài mô hình đã dùng để dẫn ra nó.
- Với điểm số rời rạc, tỉ số giảm còn 0.95 ở n = 3000: khi a nhỏ gần bằng bước lưới, lý thuyết liên tục bắt đầu sai. Điều kiện (C2) của Định lý 5a cần điểm số liên tục.

## 7. Kết luận sau phản biện

| Phần | Trạng thái | Bằng chứng |
|---|---|---|
| Định lý 5a (phân phối của t̂) | Chứng minh được (Kim–Pollard), cần viết chặt điều kiện | Mô phỏng 4 họ phân phối: 1.01–1.06; Brugada: 1.09, Spearman 0.91 |
| Hệ quả 5c (không cần nhãn) | Hệ quả trực tiếp của 5a + ước lượng plug-in | Brugada: 1.08, Spearman 0.86 |
| Xấp xỉ 5b (tổn thất) | Heuristic | Mô phỏng tốt; Brugada tốt khi nhiễu không quá nhỏ so với thiên lệch |
| Hiệu chỉnh quần thể hữu hạn | Heuristic | Tỉ số ổn định theo n/N (1.08–1.10) |

**Điểm nên đặt làm đóng góp chính:** Hệ quả 5c, tức một nguồn mới biết được ngưỡng CATT chắc chắn đến đâu và cần bao nhiêu ECG không nhãn, mà không cần nhãn nào. Luật n_eff không làm được việc này.

## 8. Đánh giá thứ hai: quy tắc dừng, độ cong đa thức cục bộ, UCI Heart

### 8a. Định lý 5a trên bộ dữ liệu thứ hai (`heart_unified.py`)

UCI Heart Disease: 9 thiết lập × 3 seed × 4 mỏ neo × 3 cỡ mẫu = 318 điểm. Cỡ mẫu nhỏ, n từ 29 đến 152.

| Ước lượng | Dự đoán/quan sát | Khoảng tứ phân vị của tỉ số | Spearman |
|---|---|---|---|
| a, KDE trên cả nguồn đích | 1.12 | 0.93–1.35 | 0.87 |
| a, độ cong đa thức cục bộ (LQ) | 1.26 | 0.99–2.12 | 0.88 |
| **Không cần nhãn, 1 mẫu con** | **1.04** | 0.89–1.25 | **0.89** |

**Theo mỏ neo (KDE):**

| Mỏ neo | EX | EX-LR | LR-same | RF-same |
|---|---|---|---|---|
| Dự đoán/quan sát | 1.12 | 1.03 | 1.08 | 1.21 |

Định lý 5a và Hệ quả 5c đúng trên một bộ dữ liệu thứ hai, khác loại hoàn toàn: dữ liệu lâm sàng, không phải ECG.

### 8b. Độ cong bằng đa thức cục bộ (LQ)

LQ khớp đa thức bậc hai vào đường Youden nhãn giả **thực nghiệm**, trong cửa sổ ±3·a·SD(Z).

| | Brugada (n vài trăm) | UCI Heart (n 29–152) |
|---|---|---|
| KDE | 1.09 | 1.12 |
| LQ | **1.05** (Spearman 0.93) | 1.26 |

- LQ tốt hơn khi n lớn.
- LQ kém hơn khi n nhỏ, vì đường thực nghiệm lởm chởm.
- **Giữ KDE (Scott) làm ước lượng chính**, đã định trước; LQ chỉ là phân tích độ nhạy.

### 8c. Kiểm trực tiếp quy tắc dừng (`stopping_rule.py`)

**Thiết kế:**
1. Lấy một mẫu thử không nhãn gồm m = 50 hoặc 100 ECG.
2. Từ mẫu thử, tính số ECG cần thu để SD(t̂) ≤ ε, với ε = 0.2 hoặc 0.3 trên thang logit.
3. Thu đúng số đó (120 lần lặp) và đo SD thực tế đạt được.

6 mẫu thử cho mỗi cấu hình (thiết lập × seed × mỏ neo).

| Mẫu thử, ước lượng độ cong | ε | SD đạt/ε (trung vị) | Khoảng tứ phân vị | Tỉ lệ đạt ε | Tỉ lệ ≤ 1.2ε | Số ECG kế hoạch (trung vị) |
|---|---|---|---|---|---|---|
| 50, KDE | 0.2 | 0.91 | 0.62–1.24 | 59% | 71% | 263 |
| 50, KDE | 0.3 | 0.90 | 0.63–1.18 | 60% | 76% | 180 |
| 100, KDE | 0.2 | 0.89 | 0.59–1.20 | 62% | 75% | 266 |
| 100, KDE | 0.3 | 0.86 | 0.63–1.12 | 65% | 81% | 180 |
| 100, LQ | 0.3 | 0.93 | 0.70–1.23 | 55% | 73% | 140 |

**Đọc kết quả:**
- **Trung bình, quy tắc dừng đúng và hơi thận trọng.** Với KDE, SD đạt được bằng khoảng 0.86–0.91 lần mục tiêu.
- **Từng lần riêng lẻ thì phân tán rộng,** vì mẫu thử 50–100 ECG chỉ có khoảng 5–20 ca nhãn giả dương, nên ước lượng Ṽ, Ã còn nhiễu.
  - Muốn 80% lần thu đạt mục tiêu, phải nhân số ECG kế hoạch với hệ số an toàn: **×1.8** (mẫu thử 100, KDE) hoặc **×2.2** (mẫu thử 50, KDE).
  - Hệ số này lớn đúng như phản biện dự đoán: a sai 20–30% thì n sai khoảng 2 lần, vì n ∝ a⁻³.
- **LQ không hợp với mẫu thử nhỏ:** dự đoán thiếu (trung vị 1.05–1.20), hệ số an toàn cần ×2.8–4.4.
- 148/2.096 mẫu thử 50–100 ECG có dưới 5 ca nhãn giả dương, không lập được kế hoạch. Với các mục tiêu chặt, 10–34% kế hoạch vượt 80% cohort.
- **Đề xuất cho bài:** dùng quy tắc **hai giai đoạn**: lập kế hoạch từ mẫu thử, thu một nửa, ước lượng lại, rồi thu nốt. Hoặc báo cáo kèm hệ số an toàn. Quy tắc hai giai đoạn chưa được kiểm.

### 8d. Những giới hạn đã ghi nhận (đồng ý với phản biện)

- **Chỉ cho ngưỡng Youden.** Ngưỡng theo độ nhạy cố định là ước lượng phân vị, tốc độ n^{−1/2}, cần một công thức khác.
- **Cần điểm số liên tục.** Điểm số rời rạc sai khi n lớn.
- **Băng thông.** Nhạy ±20–30% với a, tức khoảng 2 lần với n.
- **Cần mỏ neo có cấu trúc khác mô hình đích.**
- **Bằng chứng ECG thật vẫn chỉ có một bài toán (Brugada).** UCI Heart là dữ liệu lâm sàng, không phải ECG.

## Script

| Script | Việc làm | Đầu ra |
|---|---|---|
| `chernoff.py` | 40.000 mẫu Chernoff, lưới 2.5e-4 | `chernoff_samples.npy` |
| `unified_sim.py` | Mô phỏng Gauss | `unified_sim.csv`, `unified_sim_summary.json` |
| `nongauss_sim.py` | Mô phỏng ngoài Gauss | `nongauss_sim.csv`, `nongauss_sim_summary.json` |
| `brugada_unified.py` | Bản 1, đã thay thế (giữ để đối chiếu) | `brugada_unified.csv`, `brugada_unified_summary.json` |
| `brugada_unified_v2.py` | Bản sửa: độ phân tán, plug-in, băng thông | `brugada_unified_v2.csv`, `brugada_unified_v2_summary.json` |
| `stopping_rule.py` | Kiểm quy tắc dừng, độ cong LQ trên Brugada | `stopping_rule.csv/_summary.json`, `curvature_lq_spread.csv` |
| `heart_unified.py` | Định lý 5a, Hệ quả 5c trên UCI Heart | `heart_unified.csv/_summary.json` |
| `make_fig_v13.py`, `make_fig_v13b.py` | Hình | `fig_v13.png`, `fig_v13b.png` |

Chạy lại:

```
python analysis_v13/chernoff.py analysis_v13
python analysis_v13/unified_sim.py analysis_v13                          # khoảng 40 phút
python analysis_v13/nongauss_sim.py analysis_v13                         # khoảng 20 phút
python analysis_v13/brugada_unified_v2.py analysis_v11 analysis_v13      # khoảng 40 phút, cần analysis_v11/scores
python analysis_v13/make_fig_v13b.py analysis_v13
```

`analysis_v11/scores/` bị `.gitignore` chặn vì chứa nhãn từng bệnh nhân. Muốn có nó, tạo lại bằng `python analysis_v11/labeller_scores.py` (khoảng 30 phút), không nên đẩy lên GitHub.
