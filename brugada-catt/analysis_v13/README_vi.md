# Bước 13: sửa công thức cho đúng hơn (Mệnh đề 5, công thức tổn thất hợp nhất)

**Vấn đề của bước 11–12:** hai công thức tách rời nhau.
- Mệnh đề 1 cho thiên lệch, đúng với mọi mỏ neo.
- Mệnh đề 3 (n_eff) cho nhiễu lấy mẫu, nhưng chỉ đúng khi mỏ neo độc lập có điều kiện.

Bước 12 chỉ ra hai điểm yếu:
1. Với mỏ neo phụ thuộc, n_eff dự đoán sai: lỗi tương quan làm phương sai nhỏ đi.
2. Khi n nhỏ, công thức thận trọng quá 1.5–2 lần, vì tổn thất được xấp xỉ bằng parabol.

## Mệnh đề 5 (công thức mới)

Gọi t̃ là ngưỡng CATT trên quần thể, tức là điểm cực đại của đường Youden theo nhãn giả J̃(t).

- **Phân phối của ngưỡng ước lượng:** t̂ − t̃ ≈ a·Z, với Z ~ phân phối Chernoff và a = (4Ṽ / (Ã²·n))^{1/3}.
  - Ṽ = p(t̃ | ỹ=1)/q + p(t̃ | ỹ=0)/(1−q): mật độ điểm số f trong từng lớp nhãn giả, tại t̃.
  - Ã = |J̃″(t̃)|: độ cong của đường Youden nhãn giả tại t̃.
- **Tổn thất kỳ vọng:** E[tổn thất] = E_Z[BA(t\*) − BA(t̃ + a·Z)]. Dùng đúng đường balanced accuracy, không khai triển parabol.

Các ý chính:
- **Thiên lệch và nhiễu nằm chung một công thức.** Khi t̃ ≠ t\*, Z được cộng quanh t̃ chứ không quanh t\*.
- **Không cần giả định độc lập.** Ṽ và Ã tính trực tiếp từ phân phối nhãn giả. Sự phụ thuộc giữa lỗi mỏ neo và f đã nằm sẵn trong hai đại lượng này.
- **Trường hợp riêng:** khi mỏ neo độc lập thì t̃ = t\*, Ṽ = φ/[q(1−q)], Ã = k|J″|. Công thức quay về đúng luật n_eff của Mệnh đề 3.
- **Mẫu nhỏ:** vì dùng đường BA thật (bị chặn, phẳng dần ở xa đỉnh), công thức không còn phóng đại tổn thất khi a lớn.
- **Lấy mẫu không hoàn lại** từ cohort N người: thay n bằng n/(1 − n/N).
- **Chứng minh:** dạng a·Z là định lý argmax cube-root của Kim–Pollard (1990) áp cho tiêu chí J̃_n. Hàm mục tiêu là hiệu hai hàm phân phối thực nghiệm; hằng số 4Ṽ/Ã² theo cách tính của Banerjee–McKeague (2007). Phần cần viết chặt là điều kiện đều: J̃ có cực đại duy nhất, đạo hàm bậc hai khác 0, mật độ liên tục quanh t̃.

## Kết quả

| | Mô phỏng (64 ô × 500 lần) | Brugada, 3 cohort (189 điểm) |
|---|---|---|
| Trung vị dự đoán/quan sát: **mới** | **1.03** | **1.07** |
| Trung vị dự đoán/quan sát: cũ (MĐ 1 + MĐ 3) | 1.28 | 1.58 |
| Sai số điển hình \|log tỉ số\|: mới / cũ | 0.03 / 0.24 | 0.19 / 0.46 |
| R²(log) mô phỏng; Pearson(log) Brugada: mới / cũ | 0.993 / 0.933 | 0.96 / 0.96 |

**Mô phỏng theo mức phụ thuộc ρ (trung vị dự đoán/quan sát):**

| ρ | 0 | 0.3 | 0.6 | 0.9 |
|---|---|---|---|---|
| Mới | 1.09 | 1.04 | 1.01 | 1.00 |
| Cũ | 1.18 | 1.35 | 1.41 | 1.41 |

**Mô phỏng theo cỡ mẫu n:**

| n | 100 | 300 | 1000 | 3000 |
|---|---|---|---|---|
| Mới | 1.09 | 1.02 | 1.01 | 1.02 |
| Cũ | 1.79 | 1.41 | 1.17 | 1.11 |

**Brugada theo cỡ mẫu (n/N = 0.20 / 0.35 / 0.50):**
- Mới: 1.03 / 1.08 / 1.07
- Cũ: 1.94 / 1.58 / 1.38

**Brugada theo mỏ neo (mới):**

| Mỏ neo | CBM | CBM-sd15 | CNN9 | ResNet | CNNalt | LogReg9 | ST_V2 |
|---|---|---|---|---|---|---|---|
| Dự đoán/quan sát | 1.02 | 1.03 | 0.95 | 1.02 | 1.07 | 1.12 | 1.44 |

Ngoại lệ là ST_V2: biến rời rạc, đo bằng tay, nên mật độ nhãn giả khó ước lượng bằng KDE.

**Đọc kết quả:**
- Cả hai điểm yếu của bước 12 đều được sửa.
- Công thức mới đúng cho cả mỏ neo phụ thuộc (ρ = 0.9: 1.00) và cho mẫu nhỏ (n = 100: 1.09; Brugada 20% cohort: 1.03).
- Trên Brugada, xếp hạng không đổi (Spearman 0.94 ở cả hai), nhưng **độ lớn** dự đoán đúng hơn hẳn: sai số điển hình giảm từ 46% xuống 19%.

## Nên trình bày thế nào trong bài

- **Mệnh đề 5 là định lý chính.**
- **n_eff (Mệnh đề 3) là hệ quả dễ hiểu** cho mỏ neo độc lập: "n ECG không nhãn đáng giá n_eff ECG có nhãn". Đây là câu thông điệp cho người đọc lâm sàng.
- **Γ (Mệnh đề 2) là hệ quả cho phần thiên lệch.**

| Script | Việc làm | Đầu ra |
|---|---|---|
| `unified_sim.py` | Mô phỏng Gauss (ρ, độ lệch mỏ neo, n) | `unified_sim.csv`, `unified_sim_summary.json` |
| `brugada_unified.py` | Kiểm trên 9 thiết lập × 7 mỏ neo × 3 cỡ mẫu (cần `analysis_v11/scores`, `analysis_v12/brugada_neff.csv`) | `brugada_unified.csv`, `brugada_unified_summary.json` |
| `make_fig_v13.py` | Hình 2 panel | `fig_v13.png/.pdf` |

Chạy lại:

```
python analysis_v13/unified_sim.py analysis_v13                                # khoảng 40 phút
python analysis_v13/brugada_unified.py analysis_v11 analysis_v12 analysis_v13  # khoảng 5 phút
python analysis_v13/make_fig_v13.py analysis_v13
```
