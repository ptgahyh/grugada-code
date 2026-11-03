# Bước 14: kế hoạch phân tích định trước — bài toán ECG thứ hai (RBBB, nhiều nguồn)

**Ghi lại và commit ngày 03/11/2026, TRƯỚC khi tải tín hiệu ECG và trước khi huấn luyện bất kỳ mô hình nào.**
Chỉ metadata (nhãn, máy đo) đã được xem để thiết kế: PTB-XL có 21.799 bản ghi, CRBBB 541 (2,5%), IRBBB 1.118.
Mọi thay đổi sau thời điểm này phải ghi vào mục "Sai lệch so với kế hoạch" ở cuối tệp, kèm lý do.

## 1. Mục tiêu

Kiểm trên một bài toán ECG khác, dữ liệu khác nước, khác máy, cỡ mẫu lớn:
1. Định lý 5a (phân phối của ngưỡng CATT) và Hệ quả 5c (ước lượng không cần nhãn).
2. Quy tắc dừng một giai đoạn và **hai giai đoạn** (chưa kiểm ở bước 13).
3. CATT so với SRC, MED, UPA, GMM-2, Otsu, ORACLE.

## 2. Dữ liệu và miền

| Miền | Nguồn | Ghi chú |
|---|---|---|
| D1 `ptb_cs` | PTB-XL 1.0.3, máy CS100 / CS-12 / CS-12 E (Schiller) | 100 Hz (`records100`) |
| D2 `ptb_at` | PTB-XL 1.0.3, máy AT-6 / AT-60 (Schiller) | 100 Hz |
| D3 `cn` | ecg-arrhythmia 1.0.0 (Chapman-Shaoxing, Ningbo; Zheng và cs. 2022) | 500 Hz, hạ về 100 Hz |

- D1 ↔ D2: chỉ đổi dòng máy, cùng trung tâm, cùng cách gán nhãn → **đối chứng âm** (giống Claris ↔ Mortara).
- D1/D2 ↔ D3: khác nước, khác bệnh viện, khác quy trình gán nhãn → **đối chứng dương** (giống HUCA ↔ Melo).
- PTB-XL: bản ghi có máy không thuộc hai nhóm trên bị loại.

**Nhãn:** dương tính = block nhánh phải hoàn toàn.
- PTB-XL: có mã SCP `CRBBB` (mọi giá trị likelihood). **Loại** bản ghi có `IRBBB` mà không có `CRBBB`, vì D3 không có nhãn IRBBB riêng.
- D3: có mã SNOMED 59118001 (RBBB) trong dòng `#Dx`.
- Giới hạn biết trước: định nghĩa RBBB của D3 có thể gồm một phần block không hoàn toàn.

**Chuẩn hóa:** giống hệt `harmonized/harmonize.py`: 9 chuyển đạo (I, II, III, V1–V6), lọc Butterworth bậc 2 0,5–40 Hz hai chiều,
đỉnh năng lượng QRS, nhịp trung vị 75 mẫu, neo mẫu 28, z-score từng chuyển đạo. D3 hạ 500 → 100 Hz bằng `resample_poly(1, 5)`
trước khi lọc. Loại bản ghi có < 3 nhịp đầy đủ.

## 3. Mô hình

- **Mô hình đích f:** CNN 9 chuyển đạo, đúng kiến trúc và quy trình huấn luyện của `cbm_lib4.train_cnn`, ensemble seed 0–2.
- **Tập huấn luyện:** mỗi miền nguồn lấy tối đa 12.000 bản ghi ngẫu nhiên phân tầng theo nhãn (seed 0) để giới hạn thời gian CPU.
  Validation 15% phân tầng theo nhãn; 3 seed validation (0, 1, 2).
- **Thiết lập:** 6 cặp một nguồn (D1→D2, D2→D1, D1→D3, D3→D1, D2→D3, D3→D2). Miền đích dùng **toàn bộ** bản ghi hợp lệ.
- **Mỏ neo** (ngưỡng = Youden trên validation nguồn):
  - A1 `QRS_rms`: thời lượng RMS của năng lượng đạo hàm 9 chuyển đạo (như `rms_duration` của `concepts5.py`, không lọc 15 Hz),
    trong cửa sổ −80 đến +140 ms quanh đỉnh. Không học. Giá trị lớn → dương.
  - A2 `RBBB-LR`: hồi quy logistic (L2, C = 1, chuẩn hóa) trên 4 phép đo của nhịp trung vị: A1; trung bình V1 ở +30 đến +70 ms
    sau đỉnh năng lượng; giá trị nhỏ nhất của I và của V6 ở cùng cửa sổ. Học trên phần train của nguồn.
  - A3 `CNNalt`: CNN 9 chuyển đạo giống f nhưng seed 3–5 (mỏ neo **phụ thuộc**, đối chứng).

## 4. Phân tích và tiêu chí (định trước)

**P1. Định lý 5a (độ phân tán của ngưỡng CATT).**
- Mỗi (thiết lập, seed validation, mỏ neo): n ∈ {500, 1000, 2000} lấy không hoàn lại từ miền đích, 300 lần; SD(t̂) quan sát.
- Dự đoán a·SD(Z), a = (4Ṽ(1 − n/N)/(Ã²n))^{1/3}, Ṽ, Ã từ KDE băng thông Scott trên cả miền đích (như `brugada_unified_v2.py`).
- **Tiêu chí:** trung vị dự đoán/quan sát trong [0,8; 1,25] và Spearman ≥ 0,7.
- Độ nhạy: băng thông ×0,5 và ×2; bỏ các điểm có < 5 ca nhãn giả dương trung bình mỗi mẫu con.

**P2. Hệ quả 5c (không cần nhãn).** Như P1 nhưng Ṽ, Ã, t̃ ước lượng từ đúng một mẫu con không nhãn cỡ n. Cùng tiêu chí.

**P3. Quy tắc dừng.** Mẫu thử m = 300; ε ∈ {0,15; 0,30} (thang logit); 6 mẫu thử mỗi cấu hình, 120 lần thu mỗi kế hoạch.
- Một giai đoạn: n = n_∞/(1 + n_∞/N), n_∞ = 0,54·Ṽ/(Ã²ε³) từ mẫu thử.
- **Hai giai đoạn (mới):** lập kế hoạch n₁ từ mẫu thử; thu ⌈n₁/2⌉ (gồm mẫu thử); ước lượng lại Ṽ, Ã trên phần đã thu;
  n cuối = max(⌈n₁/2⌉, n₂). Kế hoạch > 0,8N ghi là không khả thi.
- **Tiêu chí:** tỉ lệ đạt SD ≤ 1,2ε ≥ 80% ở quy tắc hai giai đoạn. Báo cáo hệ số an toàn cần cho 80% ở cả hai quy tắc.

**P4. CATT so với các phương pháp không nhãn** (tỉ lệ tự nhiên, trung bình 3 seed):
- Balanced accuracy của SRC, MED, UPA, GMM-2, Otsu, CATT(A1), CATT(A2), CATT(A3), ORACLE; cùng hàm như `v8_analyze.py`.
- Thiết lập "hỏng" được định nghĩa trước: SRC < ORACLE − 0,05.
- **Tiêu chí mô tả:** ở thiết lập hỏng, CATT(A1/A2) cải thiện so với SRC; CATT(A3) có khoảng cách tới oracle lớn hơn A1/A2
  (dự đoán của Mệnh đề 1–2). Bootstrap bệnh nhân 1000 lần cho hiệu CATT − phương pháp khác. Nếu không có thiết lập hỏng thì
  báo cáo như vậy, không đổi định nghĩa.

## 5. Điều sẽ báo cáo dù kết quả thế nào

Mọi tiêu chí ở mục 4, đạt hay không đạt; số bản ghi bị loại ở từng bước; thời gian chạy.

## 6. Sai lệch so với kế hoạch

(trống tại thời điểm commit)
