# Bước 14: kiểm định lại công thức trên bộ dữ liệu ECG đa nguồn thứ hai (RBBB)

Kế hoạch phân tích `PLAN_vi.md` được commit ngày 03/11/2026 lúc 04:34 UTC (commit c28e07f), **trước** khi
huấn luyện bất kỳ mô hình nào. Mọi tiêu chí dưới đây được báo cáo đúng như kế hoạch, đạt hay không đạt.

## Dữ liệu (mục 2 của kế hoạch)

| Miền | Nguồn | N | RBBB (+) | Bị loại |
|---|---|---|---|---|
| D1 `ptb_cs` | PTB-XL 1.0.3, máy CS | 12 403 | 297 (2,4%) | 0 do ít nhịp |
| D2 `ptb_at` | PTB-XL 1.0.3, máy AT | 8 278 | 244 (2,9%) | 1 do ít nhịp |
| D3 `cn` | ecg-arrhythmia 1.0.0 (Chapman/Ningbo) | 43 403 | 1 734 (4,0%) | 1 501 do ít nhịp, 2 header hỏng |

Ngoài ra, ở PTB-XL bị loại 1 117 bản ghi chỉ có IRBBB, ở D3 bị loại 246 bản ghi chỉ có IRBBB (theo kế hoạch).
Chi tiết ở `results/harmonize_log*.json`. Sai lệch SL1 (mã SNOMED của D3) được ghi trong `PLAN_vi.md` mục 6.
Sai lệch này được phát hiện từ metadata, trước khi huấn luyện.

Hài hòa tín hiệu giống hệt `harmonized/harmonize.py`: 9 chuyển đạo, 100 Hz, lọc 0,5–40 Hz, nhịp trung vị 75 mẫu.

## Chạy lại

```
python harmonize_rbbb.py <thư mục PTB-XL> <ecg-arrhythmia-1.0.0.zip> <npz>
python run_rbbb.py <npz> <scores> <chunk> <số chunk>        # 18 job, ~25 phút với 4 tiến trình CPU
python analyze_rbbb.py <scores> chernoff_samples.npy <out> p1   # rồi p3, p4, summary (~20 phút)
```

Điểm từng bản ghi (`scores_*.npz`) **không** được đẩy lên kho vì chứa nhãn từng bản ghi. Thư mục `results/` chỉ chứa
số liệu tổng hợp.

AUROC trên miền đích (f / CNNalt / A1 / A2) nằm trong khoảng 0,985–0,999 / 0,985–0,999 / 0,936–0,941 / 0,986–0,997.
RBBB là bài toán dễ, nên dịch chuyển miền làm hỏng ngưỡng ít hơn nhiều so với Brugada.

## Kết quả theo tiêu chí định trước

### P1: Định lý 5a (SD của t̂ khi biết Ṽ, Ã trên cả miền đích): **ĐẠT, nhưng sát biên**

| | trung vị dự đoán/quan sát | IQR | Spearman | số điểm |
|---|---|---|---|---|
| Tất cả | **0,80** | 0,61–1,00 | **0,87** | 162 |
| A1 QRS_rms | 0,77 | 0,57–0,92 | 0,75 | 54 |
| A2 RBBB-LR | 0,73 | 0,60–0,93 | 0,90 | 54 |
| A3 CNNalt | 0,93 | 0,72–1,04 | 0,75 | 54 |
| băng thông ×0,5 | 0,74 | | 0,82 | 162 |
| băng thông ×2 | 0,91 | | 0,80 | 162 |

- Tiêu chí là trung vị trong [0,8; 1,25] và Spearman ≥ 0,7. Kết quả 0,8008 và 0,87 nên đạt, nhưng chỉ đạt sát ngưỡng dưới.
- R² trên thang log của công thức là 0,77. Mô hình chỉ dùng n (SD ∝ (n/(1−n/N))^{−1/3}) cho R² = 0,14.
  Công thức giải thích phần lớn khác biệt giữa các thiết lập và mỏ neo.
- **Khác với Brugada (tỉ số 1,09), ở đây công thức dự đoán thấp (khoảng 20%).** Rõ nhất ở n = 500 (A1 0,70; A2 0,68).
  Sai lệch giảm khi n tăng (A2: 0,68 → 0,73 → 0,75; A3: 0,89 → 0,94 → 0,96). Không có điểm nào bị bỏ vì < 5 nhãn giả dương.

*Chẩn đoán sau (post hoc, không thuộc kế hoạch):* phân phối t̂ có đuôi phải rất dài khi mỗi mẫu con chỉ có khoảng 15–25 ca
nhãn giả dương. Ví dụ ptb_at→ptb_cs với A2, n = 500: SD = 1,79 trong khi SD suy từ IQR chỉ là 1,15. Phân vị 5/50/95 của t̂
là −4,5 / −3,6 / +0,5. Khi nhóm dương nhỏ, J̃ trong mẫu con thỉnh thoảng có đỉnh thứ hai xa đỉnh chính. Xấp xỉ Chernoff
(chỉ mô tả dao động quanh một đỉnh) không bao được những lần nhảy đó, nên dự đoán thấp hơn quan sát.
Đây là điều kiện C1 của Định lý 5a bị vi phạm ở n nhỏ, không phải lỗi hằng số.

### P2: Hệ quả 5c (Ṽ, Ã ước lượng từ một mẫu con không nhãn): **KHÔNG ĐẠT**

Trung vị 0,78 (IQR 0,56–1,02), Spearman 0,83. Trung vị nằm ngoài [0,8; 1,25] một chút. Nguyên nhân giống P1, cộng thêm
nhiễu khi ước lượng Ã từ mẫu con nhỏ (A2: 0,54–0,65).

### P3: Quy tắc dừng: **KHÔNG ĐẠT**

| Quy tắc | ε | khả thi (≤ 0,8N) | đạt SD ≤ ε | đạt SD ≤ 1,2ε | n kế hoạch (trung vị) | hệ số an toàn cho 80% |
|---|---|---|---|---|---|---|
| 1 giai đoạn | 0,15 | 215/324 | 37% | 49% | 5 692 | 6,9 |
| 1 giai đoạn | 0,30 | 284/324 | 36% | 52% | 2 612 | 7,0 |
| 2 giai đoạn | 0,15 | 208/324 | 44% | **58%** | 5 537 | 4,6 |
| 2 giai đoạn | 0,30 | 292/324 | 37% | **54%** | 2 433 | 4,7 |

- Tiêu chí là ≥ 80% đạt SD ≤ 1,2ε ở quy tắc hai giai đoạn. Không đạt (54–58%).
- Quy tắc hai giai đoạn tốt hơn một giai đoạn: tỉ lệ đạt cao hơn và hệ số an toàn giảm từ khoảng 7 xuống khoảng 4,6.
  Với A2, ε = 0,15, tỉ lệ đạt tăng từ 49% lên 81%.
- Muốn có bảo đảm 80% trên dữ liệu này, cỡ mẫu tính theo công thức phải nhân khoảng 4,6 (hai giai đoạn) hoặc 7
  (một giai đoạn). Tương đương ε thực tế lớn hơn ε danh nghĩa khoảng 1,7–1,9 lần.
- Không khả thi ≤ 0,8N gặp chủ yếu ở các miền đích PTB nhỏ với ε = 0,15.

### P4: CATT so với các phương pháp không nhãn: **không có thiết lập "hỏng"**

Không thiết lập nào có SRC < ORACLE − 0,05. Khoảng cách lớn nhất là ptb_cs→cn: SRC 0,932, ORACLE 0,971 (−0,040).
Theo kế hoạch, phần so sánh trên thiết lập hỏng được báo cáo là **không áp dụng**, và định nghĩa không được đổi.

Balanced accuracy (trung bình 3 seed validation; `results/p4_point.csv`):

| | SRC | MED | UPA | GMM-2 | Otsu | CATT(A1) | CATT(A2) | CATT(A3) | ORACLE |
|---|---|---|---|---|---|---|---|---|---|
| cs→at | 0,964 | 0,977 | 0,902 | 0,986 | 0,988 | 0,935 | **0,993** | 0,981 | 0,993 |
| cs→cn | 0,932 | 0,650 | 0,805 | 0,964 | 0,968 | 0,944 | **0,970** | 0,955 | 0,971 |
| at→cs | 0,983 | 0,663 | 0,995 | 0,964 | 0,995 | 0,899 | 0,988 | 0,991 | 0,996 |
| at→cn | 0,948 | 0,653 | 0,883 | 0,942 | 0,968 | 0,911 | 0,966 | 0,963 | 0,970 |
| cn→cs | 0,995 | 0,993 | 0,984 | 0,975 | 0,995 | 0,925 | 0,925 | 0,992 | 0,996 |
| cn→at | 0,993 | 0,991 | 0,984 | 0,985 | 0,993 | 0,931 | 0,927 | 0,991 | 0,993 |

Hiệu gộp trên cả 6 thiết lập (điểm, KTC 95% bootstrap 1 000 lần, phân tầng theo nhãn):
CATT(A2) − SRC = −0,008 (−0,013; −0,005); CATT(A2) − Otsu = −0,023 (−0,028; −0,021);
CATT(A1) − SRC = −0,045 (−0,053; −0,041). Đầy đủ trong `results/summary.json`.

Mô tả:
- Ở hai thiết lập PTB→cn, nơi SRC lệch nhiều nhất, CATT(A2) gần như bằng oracle: +0,039 và +0,018 so với SRC.
  Đây đúng là trường hợp phương pháp được thiết kế cho.
- Ở hai thiết lập cn→PTB, SRC đã gần oracle. Mỏ neo A2 học trên cn chuyển sang PTB kém, nên CATT(A2) mất khoảng 0,07.
  Mỏ neo thủ công A1 kém ở mọi thiết lập. CATT chỉ tốt khi mỏ neo chuyển miền tốt hơn mô hình đích.
  Điều này đúng như Mệnh đề 1–2 cảnh báo (1 − α − β của mỏ neo).
- Dự đoán "A3 phụ thuộc có khoảng cách tới oracle lớn hơn A1/A2" **không được ủng hộ** trên RBBB. A3 ổn định nhất.
  Khi mô hình đích gần hoàn hảo (AUROC ≈ 0,99), lỗi chung giữa f và A3 rất ít, nên sự phụ thuộc không gây hại.
- Otsu và GMM-2 rất mạnh ở bài toán này vì phân phối điểm của f gần như tách hẳn thành hai mode.

Lưu ý: bootstrap lấy lại theo bản ghi. PTB-XL có một số bệnh nhân có nhiều bản ghi, nên KTC có thể hơi hẹp.

## Kết luận cho bài báo

1. Định lý 5a **tổng quát hóa được về hình dạng**: Spearman 0,87, R² 0,77 so với 0,14 của mô hình chỉ theo n, trên một
   bệnh khác, các thiết bị khác và một quốc gia khác. Nhưng **hằng số không tổng quát hóa được ở n nhỏ khi nhóm dương hiếm**:
   công thức dự đoán thấp khoảng 20%, P1 chỉ đạt sát biên và P2 không đạt.
2. Quy tắc dừng **không giữ được bảo đảm 80%** trên dữ liệu thứ hai. Khi viết bài, cần nói rõ đây là quy tắc *ước lượng*
   có hệ số an toàn thực nghiệm khoảng 5 (hai giai đoạn), không phải bảo đảm. Hoặc đưa thêm điều kiện đủ
   (ví dụ số nhãn giả dương kỳ vọng ≥ 50 trong mẫu).
3. Lợi ích của CATT phụ thuộc vào mỏ neo. Trên RBBB không có thiết lập hỏng, nên dữ liệu này không kiểm được giả thuyết
   chính về khả năng sửa ngưỡng hỏng. Nó chỉ cho thấy CATT không gây hại khi mỏ neo tốt (A2 PTB→cn) và có hại khi mỏ neo kém
   (A1; A2 cn→PTB).
