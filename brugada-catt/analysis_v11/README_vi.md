# Bước 11: lý thuyết thiên lệch và cỡ mẫu của CATT (03/11/2026)

**Mục tiêu:** biến phần lý thuyết của CATT từ "dùng lại kết quả của Menon 2015" thành một đóng góp lý thuyết riêng.
Menon 2015 chỉ cho biết ngưỡng CATT **đúng** khi lỗi nhãn giả độc lập có điều kiện với điểm của CNN. Bước này trả lời ba câu hỏi Menon không trả lời:
1. Khi **không** độc lập, ngưỡng CATT rơi vào đâu, và lệch bao nhiêu?
2. Khi độc lập, cần bao nhiêu ECG không nhãn, và một ECG không nhãn "đáng giá" bao nhiêu ECG có nhãn?
3. Các công thức này có giải thích được dữ liệu thật không?

| Script | Việc làm | Đầu ra |
|---|---|---|
| `theory_sim.py` | Kiểm các mệnh đề trong mô hình Gauss (giống mô phỏng của bài, mục 2.9) | `theory_population.csv`, `theory_sampling.csv`, `theory_invariance.csv`, `theory_summary.json` |
| `labeller_scores.py` | Huấn luyện lại 4 bộ gán nhãn hộp đen của bước 6 (CNN9, ResNet, LogReg9, CNNalt) để có điểm từng bệnh nhân | `scores/lab_<thiết lập>_<seed>.npz` (bị .gitignore chặn) |
| `theory_real.py` | Kiểm lý thuyết trên 9 thiết lập × 3 seed × 7 bộ gán nhãn | `theory_real_raw.csv`, `theory_real_points.csv`, `theory_real_by_labeller.csv`, `theory_real_summary.json` |
| `make_fig_theory.py` | Hình 3 panel | `fig_theory.png/.pdf` |

Kiểm tra tái lập: α, β của cả 7 bộ gán nhãn ở các thiết lập đã đối chiếu **trùng đúng** số của bước 6.

---

## 1. Ký hiệu

- f: điểm (logit) của mô hình đích CNN-V1V3. fₖ(t): mật độ của f ở lớp k (k = 0 âm, 1 dương). πₖ: tỉ lệ lớp ở viện đích.
- g: điểm của bộ gán nhãn (mỏ neo), nhãn giả ỹ = 1[g ≥ c], c là ngưỡng nguồn của g.
- α = P(ỹ = 1 | y = 0), β = P(ỹ = 0 | y = 1), q = P(ỹ = 1).
- **Tỉ lệ gán dương cục bộ:** aₖ(t) = P(ỹ = 1 | f = t, y = k). Nếu độc lập: a₀(t) = α và a₁(t) = 1 − β với mọi t.
- t\*: ngưỡng Youden thật (nơi f₀ = f₁); J(t) = độ nhạy + độ đặc hiệu − 1; balanced accuracy = (1 + J)/2.
- J̃(t): chỉ số Youden đo bằng nhãn giả; ngưỡng CATT tổng thể t̃ = argmax J̃.

## 2. Bốn mệnh đề

### Mệnh đề 1 (điều kiện chính xác, không cần giả định độc lập)
Đạo hàm của J̃ là dJ̃/dt = −S(t) / [q(1 − q)], với

> **S(t) = Σₖ πₖ fₖ(t) [aₖ(t) − q]**

Ngưỡng CATT tổng thể là nghiệm của S(t) = 0 (nơi S đổi dấu từ âm sang dương).

- Chứng minh ngắn: P(f ≥ t, ỹ = 1) = Σₖ πₖ ∫ₜ^∞ fₖ(s) aₖ(s) ds; lấy đạo hàm hai phân số rồi quy đồng.
- **Trường hợp riêng (Menon 2015):** nếu độc lập, S(t) = π₀π₁(1 − α − β)[f₁(t) − f₀(t)], nên t̃ = t\* khi α + β < 1. Mệnh đề 1 chứa kết quả của Menon như một trường hợp đặc biệt.

### Mệnh đề 2 (thiên lệch bậc nhất: "lỗi ở ranh giới" mới quan trọng)
Viết D(t) = log f₁(t)/f₀(t) (tỉ số khả dĩ của chính CNN, D(t\*) = 0). Điều kiện S = 0 tương đương D(t̃) = Γ(t̃), với

> **Γ(t) = log[(q − a₀(t)) / (q − α)] − log[(a₁(t) − q) / (1 − β − q)]**

Γ ≡ 0 khi độc lập. Khai triển quanh t\*:

> **t̃ − t\* ≈ Γ(t\*) / [D′(t\*) − Γ′(t\*)]**, và tổn thất balanced accuracy ≈ ¼ |J″(t\*)| (t̃ − t\*)².

Khi lệch nhỏ, Γ(t\*) ≈ [(β\* − β)/π₀ − (α\* − α)/π₁] / (1 − α − β), với α\* = a₀(t\*), β\* = 1 − a₁(t\*) là tỉ lệ lỗi **của bộ gán nhãn đúng tại ranh giới quyết định của CNN**.

**Ý nghĩa:** CATT lệch khi lỗi của mỏ neo **dồn vào đúng những ECG mà CNN đang phân vân**, và lệch càng nặng khi 1 − α − β nhỏ. Độ tương quan chung (cond_rho trong bài) chỉ là đại diện thô của đại lượng này. Một bộ gán nhãn chính xác hơn nhưng mắc lỗi ở cùng chỗ với CNN sẽ có Γ lớn: đây là giải thích lý thuyết cho kết quả "góc nhìn khác hơn mô hình chính xác hơn".

### Mệnh đề 3 (luật cỡ mẫu, mỏ neo độc lập)
Ngưỡng Youden thực nghiệm hội tụ với tốc độ n^(−1/3) (lý thuyết cube-root của Kim và Pollard 1990). Với nhãn giả, độ cong của J̃ bị nhân với k = π₀π₁(1 − α − β)/[q(1 − q)], phương sai cục bộ thành f(t\*)/[q(1 − q)]. Suy ra:

> **Tổn thất ≈ ¼ |J″| · Var(Z) · [4 f(t\*) / (J″² π₀π₁ n_eff)]^(2/3)**, với **n_eff = n · π₀π₁(1 − α − β)² / [q(1 − q)]**

(Z là biến Chernoff, Var(Z) ≈ 0.264.) Cùng công thức với n_eff = n cho ngưỡng Youden dùng nhãn thật.

**Cách nói dễ hiểu:** CATT với n ECG **không nhãn** cho sai số như ngưỡng Youden với n_eff ECG **có nhãn**. Ở dữ liệu thật, mỏ neo CBM-sd15 có n_eff/n ≈ **0.21**: 100 ECG không nhãn ≈ 21 ECG có nhãn.

### Mệnh đề 4 (bất biến theo ngưỡng mỏ neo)
Nếu độc lập, t̃ không phụ thuộc ngưỡng c của mỏ neo (miễn α + β < 1). Ngược lại, nếu t̃ thay đổi theo c thì giả định độc lập bị vi phạm. Về nguyên tắc đây là một phép kiểm **không cần nhãn đích** (xem mục 4: trên dữ liệu thật nó không đủ nhạy).

## 3. Kiểm chứng bằng mô phỏng (`theory_sim.py`)

Mô hình Gauss như mục 2.9 của bài: π = 0.2; f | y=0 ~ N(0, 1), f | y=1 ~ N(1.6, 1.6²); g | y ~ N(0 hoặc 2.5, 1); tương quan trong lớp ρ; ngưỡng mỏ neo = điểm giữa (1.25) + độ dời.

**A. Mệnh đề 1 khớp Monte Carlo** (n = 400.000, 3 lần): sai khác lớn nhất 0.038 đơn vị logit (bằng nhiễu mẫu). ρ = 0: thiên lệch bằng 0 tuyệt đối ở mọi độ dời (2.9 × 10⁻¹⁵).

**B. Tổn thất tổng thể (không còn nhiễu mẫu), balanced accuracy:**

| ρ | độ dời 0 | −0.75 | −1.25 | −1.75 | −2.25 |
|---|---|---|---|---|---|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.3 | 0.004 | 0.017 | 0.030 | 0.045 | 0.061 |
| 0.6 | 0.007 | 0.029 | 0.051 | 0.075 | 0.100 |
| 0.9 | 0.003 | 0.030 | 0.064 | 0.106 | 0.147 |

- Dòng ρ = 0.9 (0.030, 0.064, 0.106, 0.147) **trùng** các con số 0.03, 0.07, 0.11, 0.15 mà bài trước đây chỉ có từ mô phỏng. Giờ chúng là hệ quả của một công thức.
- Phụ thuộc mà ngưỡng mỏ neo đúng: tổn thất rất nhỏ (≤ 0.007). Lệch ngưỡng mà độc lập: 0. Cần **cả hai** mới hại, đúng như bài đã quan sát.
- Công thức bậc nhất (Mệnh đề 2) chính xác khi lệch nhỏ (ρ ≤ 0.6, độ dời +0.75: sai khác < 0.001) nhưng **phóng đại** khi lệch lớn; khi đó dùng nghiệm đúng của Mệnh đề 1.

**C. Luật cỡ mẫu (Mệnh đề 3)**, ρ = 0, n = 100–6.400, 600 lần lặp, 5 mức 1 − α − β (0.16–0.79):

| Hồi quy log(tổn thất) theo | Lý thuyết | Thực nghiệm (n ≥ 800) | Thực nghiệm (mọi n) |
|---|---|---|---|
| log n | −0.667 | −0.642 | −0.578 |
| log(1 − α − β) | −1.333 | −1.313 | −1.196 |
| log q(1 − q) | +0.667 | +0.753 | +0.729 |

- Hằng số: dự đoán / quan sát = 1.09 (n ≥ 800), 1.17 (mọi n). Với ngưỡng dùng nhãn thật: 1.07.
- Ở n_eff rất nhỏ (n nhỏ và 1 − α − β nhỏ), tổn thất quan sát thấp hơn dự đoán vì bị chặn trên; công thức là tiệm cận.

## 4. Kiểm chứng trên dữ liệu thật (`theory_real.py`)

63 điểm (7 bộ gán nhãn × 9 thiết lập, trung bình 3 seed). Mật độ của CNN được tóm bằng phân phối chuẩn hai lớp trên logit đích; tỉ lệ lỗi cục bộ α\*, β\* ước lượng bằng trọng số kernel quanh ngưỡng oracle (băng thông 0.3 SD). Nhãn đích chỉ dùng để phân tích.

**Độ lệch ngưỡng dự đoán so với quan sát (t_CATT − t_ORACLE):** Spearman **0.975**, Pearson **0.94**; hồi quy quan sát theo dự đoán: hệ số 0.70, hằng số 0.14.

| Bộ gán nhãn | Lệch quan sát | Lệch dự đoán | Khoảng cách tới oracle (quan sát) | Tổn thất do lấy mẫu (dự đoán) | n_eff / n |
|---|---|---|---|---|---|
| CBM-sd15 | +0.07 | +0.07 | 0.018 | 0.014 | 0.21 |
| CBM | −0.05 | −0.07 | 0.025 | 0.015 | 0.22 |
| ST_V2 | +0.14 | +0.14 | 0.026 | 0.015 | 0.20 |
| LogReg9 | −0.11 | −0.33 | 0.039 | 0.031 | 0.16 |
| CNN9 | −0.10 | −0.24 | 0.060 | 0.027 | 0.11 |
| ResNet | −0.26 | −0.77 | 0.098 | 0.026 | 0.21 |
| CNNalt | −0.38 | −1.19 | 0.105 | 0.033 | 0.20 |

Đọc kết quả:
- **Bộ gán nhãn khái niệm:** độ lệch dự đoán khớp gần đúng tuyệt đối; khoảng cách tới oracle (0.018–0.026) chủ yếu là **nhiễu lấy mẫu** (dự đoán 0.014–0.015). Chúng gần như không thiên lệch.
- **Bộ gán nhãn hộp đen:** khoảng cách 0.04–0.10 lớn hơn nhiều phần lấy mẫu (0.03): phần dư là **thiên lệch do phụ thuộc**, đúng chiều lý thuyết dự đoán (ngưỡng bị kéo xuống, CNN thành dễ dãi). Công thức bậc nhất phóng đại độ lớn khi lệch lớn, như trong mô phỏng.
- Tổn thất dự đoán (lấy mẫu + thiên lệch, không ước lượng tham số nào) xếp hạng khoảng cách quan sát với Spearman **0.92**. Hồi quy tuyến tính một biến: R² = 0.68. Mô hình mô tả cũ của bài (cond_rho và 1 − α − β, hai hệ số khớp từ dữ liệu) có R² = 0.82 nhưng không phải là dự đoán từ lý thuyết.
- |Γ| một mình xếp hạng khoảng cách với Spearman 0.83.

**Kết quả âm tính (nói thẳng):**
- **Phép kiểm không cần nhãn (Mệnh đề 4) không phân biệt được** trên dữ liệu thật: độ dốc khi dời ngưỡng mỏ neo ±0.5 SD là 0.134 ở bộ khái niệm và 0.131 ở hộp đen; tương quan với khoảng cách chỉ 0.43. Lý do: với 356–1.154 ECG, nhiễu lấy mẫu lớn hơn tín hiệu, và bộ khái niệm cũng có phụ thuộc vừa phải.
- Đo "lỗi dồn ở ranh giới" trên validation **nguồn** (để chọn mỏ neo trước khi triển khai) cũng không dự đoán được khoảng cách ở đích (Spearman −0.27), vì validation nguồn quá nhỏ và ngưỡng mỏ neo ở nguồn luôn đúng theo định nghĩa.
- Vì vậy lý thuyết **giải thích** và **định lượng** được CATT, nhưng chưa cho một công cụ phát hiện thất bại không cần nhãn. Lời khuyên thực hành vẫn là chọn mỏ neo có cấu trúc khác hẳn mô hình đích.

## 5. Ý nghĩa cho bài báo

Phần lý thuyết mới có thể thành một mục riêng (ví dụ "2.9 Theory of concept-anchored threshold transfer") và một hình (`fig_theory.png`):
1. **Mệnh đề 1:** điều kiện chính xác cho ngưỡng CATT khi không độc lập; chứa Menon 2015 như trường hợp riêng.
2. **Mệnh đề 2:** thiên lệch do "lỗi dồn ở ranh giới quyết định", giải thích vì sao bộ gán nhãn chính xác hơn có thể tệ hơn.
3. **Mệnh đề 3:** luật n_eff: một ECG không nhãn đáng giá khoảng π₀π₁(1 − α − β)²/[q(1 − q)] ECG có nhãn.
4. Kiểm chứng: mô phỏng (khớp đến nhiễu mẫu) và 63 điểm dữ liệu thật (r = 0.94 cho độ lệch ngưỡng).

**Giới hạn cần ghi:**
- Mệnh đề 3 là kết quả tiệm cận; chứng minh chặt cần điều kiện của Kim và Pollard (1990) và nên viết trong phụ lục. Ở n_eff nhỏ, công thức phóng đại tổn thất.
- Mệnh đề 2 là xấp xỉ bậc nhất; khi lệch lớn phải dùng nghiệm của Mệnh đề 1.
- Áp lên dữ liệu thật cần nhãn đích (để ước lượng α\*, β\*); phép kiểm không cần nhãn chưa hoạt động.
- Tác giả không phải người làm thống kê lý thuyết: nên nhờ một người có chuyên môn (ví dụ giảng viên toán ứng dụng) đọc lại phần chứng minh trước khi nộp.

## 6. Chạy lại

```
python analysis_v11/theory_sim.py analysis_v11                         # khoảng 25 phút trên 2 CPU
python analysis_v11/labeller_scores.py harmonized concepts_v5 analysis_v11 0 2 &
python analysis_v11/labeller_scores.py harmonized concepts_v5 analysis_v11 1 2   # khoảng 30 phút
python analysis_v11/theory_real.py analysis_v11
python analysis_v11/make_fig_theory.py analysis_v11
```
