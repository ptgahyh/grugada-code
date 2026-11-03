# Bước 15: kế hoạch phân tích định trước — bài toán ECG thứ ba (LVH, nhiều nguồn) và bản sửa công thức

**Ghi lại và commit ngày 03/11/2026, TRƯỚC khi trích nhịp LVH, tính đặc trưng điện thế hay huấn luyện bất kỳ mô hình LVH nào.**
Đã xem trước khi viết: (i) metadata nhãn và giới tính của PTB-XL và ecg-arrhythmia (số đếm ở mục 2);
(ii) toàn bộ kết quả bước 14 (RBBB) và kết quả khám phá bước 15a (`explore_rbbb.py`, mục 5).
Tín hiệu PTB-XL và ecg-arrhythmia đã có trên đĩa từ bước 14 (nhịp đã trích cho RBBB, z-score), nhưng chưa có số liệu LVH nào
được tính từ tín hiệu. Mọi thay đổi sau thời điểm này ghi vào mục 7, kèm lý do.

## 1. Mục tiêu

1. **Kiểm giả thuyết chính trên bài toán có ngưỡng thật sự hỏng.** RBBB quá dễ (AUROC ≈ 0,99) nên không có thiết lập hỏng.
   LVH dựa vào điện thế QRS, mà điện thế phụ thuộc máy, dân số và quy trình, nên khả năng hỏng ngưỡng cao hơn.
2. **Lặp lại P1–P4 của bước 14**, cùng tiêu chí, trên bài toán thứ ba.
3. **Kiểm định xác nhận hai bản sửa công thức** (F1, F2), phát triển trên RBBB ở bước 15a.

## 2. Dữ liệu, miền, nhãn

Miền giống bước 14: D1 `ptb_cs` (PTB-XL máy CS), D2 `ptb_at` (PTB-XL máy AT), D3 `cn` (ecg-arrhythmia 1.0.0).
D1 ↔ D2 là đối chứng âm, D1/D2 ↔ D3 là đối chứng dương.

**Nhãn** (số đếm từ metadata, trước khi loại bản ghi ít nhịp):

| | dương | loại | âm |
|---|---|---|---|
| PTB-XL | mã SCP `LVH` với likelihood ≥ 50 (CS 1 088 + AT 1 044 mọi likelihood; 1 751 có likelihood ≥ 50 trên toàn bộ) | `LVH` likelihood < 50 (381); `VCLVH` (tiêu chí điện thế) không kèm `LVH` (CS 76, AT 146) | còn lại |
| cn | SNOMED 164873001 (LVH): 647/45 152 (1,4%) | SNOMED 55827005 (điện thế cao thất trái, LVHV) không kèm LVH: 5 399; giới tính Unknown: 22 | còn lại |

- Lý do loại VCLVH-only và LVHV-only: đây là nhãn "chỉ có tiêu chí điện thế", không phải chẩn đoán LVH. Nếu coi là âm thì mỏ neo
  điện thế bị phạt một cách nhân tạo, còn nếu coi là dương thì nhãn trùng với chính mỏ neo. Mã 55827005 không có trong tệp
  `ConditionNames_SNOMED-CT.csv`; ý nghĩa "left ventricular high voltage" lấy theo bảng ánh xạ của PhysioNet Challenge 2020/2021.
- Giới hạn biết trước: tỉ lệ LVH khác rất xa giữa PTB-XL (~8–12%) và cn (~1,6% sau khi loại). Đây là dịch chuyển nhãn
  có thật, thuộc về bài toán, và được giữ nguyên.

**Chuẩn hóa:** giống hệt bước 14 cho đầu vào của f (nhịp trung vị z-score). Thêm nhịp trung vị theo **mV trước z-score**
(cùng lọc, cùng căn), chỉ dùng cho mỏ neo, vì z-score xóa biên độ tuyệt đối. Code: `harmonize_lvh.py`.

## 3. Mô hình và mỏ neo (`run_lvh.py`)

- **f:** CNN 9 chuyển đạo trên nhịp z-score, đúng như bước 14 (`cbm_lib4.train_cnn`, ensemble seed 0–2). Biết trước là f
  không thấy biên độ tuyệt đối, nên có thể yếu hơn mỏ neo điện thế. Điều này được giữ nguyên vì f là quy trình đã cố định.
- Nguồn tối đa 12 000 bản ghi (phân tầng, seed 0); validation 15% phân tầng; 3 seed validation; 6 thiết lập một nguồn;
  miền đích dùng toàn bộ.
- **Mỏ neo** (ngưỡng = Youden trên validation nguồn). Biên độ đo so với đường đẳng điện = trung vị mẫu −100..−70 ms trước
  đỉnh năng lượng. R = cực đại, S = −cực tiểu trong cửa sổ QRS −80..+150 ms. aVL = (I − III)/2.
  - A1 `Sokolow`: S_V1 + max(R_V5, R_V6). Không học.
  - A2 `Cornell`: R_aVL + S_V3 + 0,8·[nữ] (tương đương ngưỡng 2,8 mV nam / 2,0 mV nữ). Không học.
  - A3 `LVH-LR`: hồi quy logistic L2 (C = 1, chuẩn hóa) trên [Sokolow, Cornell, nữ, QRS_rms, ST-T trung bình +120..+300 ms
    nhỏ nhất của V5/V6, R lớn nhất mọi chuyển đạo]. Học trên phần train của nguồn.
  - A4 `CNNalt`: CNN giống f, seed 3–5 (mỏ neo phụ thuộc, đối chứng).

## 4. Phân tích và tiêu chí (`analyze_lvh.py`)

**P1–P4: giữ nguyên định nghĩa, tham số và tiêu chí của bước 14** (analysis_v14/PLAN_vi.md mục 4), với 4 mỏ neo ở trên:
- P1 (Định lý 5a, Ṽ và Ã trên cả miền đích): trung vị dự đoán/quan sát ∈ [0,8; 1,25] và Spearman ≥ 0,7.
- P2 (Hệ quả 5c, KDE trên một mẫu không nhãn cỡ n): cùng tiêu chí.
- P3 (quy tắc dừng): ≥ 80% kế hoạch hai giai đoạn đạt SD ≤ 1,2ε, ε ∈ {0,15; 0,30}.
- P4: thiết lập hỏng ⇔ SRC < ORACLE − 0,05; bootstrap 1 000 lần.

**F1. Hệ quả 5c bản mẫu con (bản sửa, xác nhận).** Trên đúng mỗi mẫu không nhãn cỡ n của P2: lấy B = 100 mẫu con cỡ b = n/2
không hoàn lại, đo SD(t̂_b). Dự đoán SD(t̂_n) = SD(t̂_b)·[((1 − n/N)/n) / ((1 − b/n)/b)]^{1/3}. Chỉ dùng tốc độ n^{−1/3} của
Định lý 5a, không ước lượng Ṽ, Ã (`common15.sd_subsample`).
- **Tiêu chí:** trung vị dự đoán/quan sát ∈ [0,8; 1,25] và Spearman ≥ 0,7 (như P2).
- **So sánh định trước:** trung vị |log(dự đoán/quan sát)| của F1 nhỏ hơn của P2.

**F1-B. F1 trên Brugada** (`brugada_subsample_check.py`, chạy ở máy có `analysis_v11/scores`). Brugada không được dùng
để phát triển F1, nên đây là phép kiểm ngoài mẫu thứ hai, cùng tiêu chí. Báo cáo riêng cho ST_V2 (mỏ neo bị KDE dự đoán dư 1,17).

**F2. Quy tắc dừng hai giai đoạn bản mẫu con (bản sửa, xác nhận).** Giống quy tắc hai giai đoạn của P3, nhưng ở giai đoạn 2
ước lượng bằng mẫu con trên n_h điểm đã thu, rồi giải c·((1 − n/N)/n)^{1/3} = ε. n cuối = max(n_h, n₂).
- **Tiêu chí:** ≥ 80% kế hoạch khả thi đạt SD ≤ 1,2ε ở cả hai ε (như P3).
- Báo cáo thêm hệ số an toàn cần cho 80% và so với hai quy tắc của P3.

**H1. Giả thuyết chính (chỉ áp dụng nếu có ≥ 1 thiết lập hỏng).**
- Ở các thiết lập hỏng, hiệu gộp CATT(A_k) − SRC > 0 với KTC 95% bootstrap không chứa 0, cho ít nhất một mỏ neo sinh lý k ∈ {1, 2, 3}.
- Báo cáo đồng thời CATT(A_k) − Otsu và CATT(A_k) − GMM-2 (đối thủ mạnh nhất ở bước 14), đạt hay không.
- Dự đoán của Mệnh đề 1–2: khoảng cách tới oracle của CATT(A4) lớn hơn của mỏ neo sinh lý tốt nhất. Đây là tiêu chí mô tả.
- Nếu không có thiết lập hỏng: báo cáo như vậy, không đổi định nghĩa, và H1 là "không kiểm được".

## 5. Kết quả khám phá bước 15a (RBBB, KHÔNG phải bằng chứng xác nhận)

`explore_rbbb.py` trên điểm RBBB của bước 14, cùng cấu hình như P1/P2/P3:

| Ước lượng SD(t̂) không nhãn | trung vị dự đoán/quan sát | IQR | Spearman |
|---|---|---|---|
| KDE (P2 bước 14) | 0,78 | 0,56–1,02 | 0,83 |
| **Mẫu con (F1)** | **0,95** | 0,86–1,10 | **0,95** |

Quy tắc dừng hai giai đoạn bản mẫu con: 61% (ε = 0,15) và 72% (ε = 0,30) kế hoạch đạt SD ≤ 1,2ε, so với 58% và 54% của bản KDE.
Hệ số an toàn cho 80% là 3,5 và 2,5, so với 4,6 và 4,7. Vẫn dưới 80%. F2 được giữ nguyên, không chỉnh thêm, để kiểm trên LVH.

## 6. Điều sẽ báo cáo dù kết quả thế nào

Mọi tiêu chí ở mục 4, đạt hay không đạt; số bản ghi bị loại ở từng bước; AUROC của f và các mỏ neo; thời gian chạy.

## 7. Sai lệch so với kế hoạch

(chưa có)
