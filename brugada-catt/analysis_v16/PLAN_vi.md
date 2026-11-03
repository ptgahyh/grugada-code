# Bước 16: kế hoạch kiểm định định trước — CATT theo độ nhạy cố định

**Ghi lại và commit TRƯỚC khi tính bất kỳ đại lượng nào theo độ nhạy trên dữ liệu Brugada.**

Đã xem trước khi viết:
- Mô phỏng Gauss của bước 11 (ρ ∈ {0; 0,6}), dùng để dẫn công thức.
- Cỡ các cohort đích: 1 154 (596 dương), 370 (137 dương), 356 (73 dương).
- Toàn bộ kết quả theo Youden của bước 11–15.

## 1. Quy trình và ký hiệu

**CATT-độ nhạy, mục tiêu s:**
1. Gán nhãn giả ỹ = 1[g ≥ c] bằng mỏ neo; ngưỡng c là ngưỡng Youden trên validation nguồn, như mọi bước trước.
2. Chọn t̂ = phân vị (1 − s) của f trong nhóm nhãn giả dương của mẫu đích.

**Ký hiệu:**
- q = P(ỹ=1).
- Tỉ lệ báo động giả của mỏ neo: FDR_g = P(y=0 | ỹ=1).
- TPR, FPR của f tại ngưỡng t; J = TPR − FPR.
- f₁ = mật độ của f ở ca dương thật; p̃₁ = mật độ của f ở ca nhãn giả dương.

## 2. Công thức cần kiểm

**C1, độ lệch:**
- Nếu mỏ neo độc lập có điều kiện với f, tại ngưỡng quần thể t̃_s: **TPR(t̃_s) − s = FDR_g · J(t̃_s)**, chính xác.
- Hệ quả: TPR ≥ s.
- Nếu mỏ neo phụ thuộc, đẳng thức không còn đúng, và độ nhạy có thể thấp hơn s.

**C2, dao động:**
- SD(t̂) ≈ √(s(1−s)/(q·n)) / p̃₁(t̃_s).
- SD(TPR(t̂)) ≈ f₁(t̃_s) · SD(t̂).
- Lấy n mẫu không hoàn lại từ cohort N: nhân thêm √(1 − n/N), tương tự bước 13 (heuristic).

## 3. Thiết kế (`sens_brugada.py`)

- 9 thiết lập × 3 seed × 7 mỏ neo của bước 11.
  - Mỏ neo khái niệm: CBM, CBM-sd15, ST_V2.
  - Mỏ neo hộp đen: LogReg9, CNN9, ResNet, CNNalt.
- Mục tiêu s ∈ {0,80; 0,90; 0,95}.
- **Đánh giá:** mọi TPR được tính trên toàn bộ cohort đích bằng nhãn thật, chỉ để đánh giá.
- **C1:** t̃_s = phân vị (1 − s) trên toàn cohort. Đo TPR quan sát và dự đoán độc lập s + FDR_g·J(t̃_s). FDR_g và J tính bằng nhãn thật trên cohort.
- **C2:** n = 20/35/50% cohort, 300 mẫu con không hoàn lại.
  - Đo SD(t̂) và SD(TPR(t̂)).
  - Dự đoán bằng KDE (băng thông Scott) của f trong nhóm dương thật và nhóm nhãn giả dương trên toàn cohort.
- **C2-không nhãn:**
  - Trên 20 mẫu cỡ n, bootstrap B = 200 cho t̂. Với phân vị, bootstrap nhất quán, khác ngưỡng Youden.
  - Nhân √(1 − n/N); lấy trung vị 20 giá trị.
  - So với SD(t̂) quan sát.

## 4. Tiêu chí (định trước)

| Mã | Tiêu chí đạt |
|---|---|
| S1a | Trên mỏ neo khái niệm: trung vị \|TPR quan sát − dự đoán độc lập\| ≤ 0,03, và Spearman(độ vượt dự đoán, độ vượt quan sát) ≥ 0,7 |
| S1b | Trên mỏ neo khái niệm: ≥ 80% điểm (thiết lập × seed × s) có TPR quan sát ≥ s − 0,02 |
| S1c (mô tả) | Cùng các đại lượng cho mỏ neo hộp đen. Dự đoán: sai lệch lớn hơn, tỉ lệ thiếu độ nhạy cao hơn mỏ neo khái niệm |
| S2 | SD(TPR(t̂)): trung vị dự đoán/quan sát ∈ [0,8; 1,25] và Spearman ≥ 0,7 (mọi mỏ neo) |
| S3 | SD(t̂) bằng bootstrap không nhãn: trung vị dự đoán/quan sát ∈ [0,8; 1,25] và Spearman ≥ 0,7 |

**Báo cáo thêm:**
- Tỉ lệ điểm có TPR < s − 0,02 theo từng mỏ neo.
- So sánh với ngưỡng gốc chọn theo độ nhạy s trên validation nguồn (SRC-s): độ nhạy thật và độ đặc hiệu ở nguồn đích.

**Không dùng làm tiêu chí:** RBBB và LVH. Hai bộ này sẽ được kiểm sau bằng cùng script, ở máy có điểm số của chúng.

## 5. Sai lệch so với kế hoạch

(chưa có)
