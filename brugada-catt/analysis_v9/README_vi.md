# Bước 9: kiểm tra cho phản biện

Bốn phân tích, chạy 16/10/2026 trên `harmonized/*.npz` và `concepts_v5/v5_raw_*.jsonl`.

| Script | Câu hỏi | Đầu ra |
|---|---|---|
| `lead_control.py` | V1–V3 tốt hơn vì là chuyển đạo Brugada, hay chỉ vì ít đầu vào? | `lead_control_*.jsonl`, `lead_control_summary.csv` |
| `rebuttal_checks.py` (A) | CATT có còn tốt khi ngưỡng chọn trên ECG khác với ECG dùng để đánh giá? | `split_half_*.csv` |
| `rebuttal_checks.py` (B), `worst5_patient_boot.py` | Giá trị tệ nhất của CATT có chắc chắn hơn GMM-2/UPA/Otsu không? | `worst_case_ci.csv`, `worst_case_ci_5pct_patient_boot.csv` |
| `rebuttal_checks.py` (C) | ST_V2 có kém CBM khi QRS rộng không? | `stv2_vs_cbm_by_qrs_*.csv` |

Kiểm tra đầu vào: chạy lại trên toàn cohort cho đúng Bảng 12 (CATT 0.799, SRC 0.701, UPA 0.786, GMM-2 0.785, Otsu 0.783, MED 0.773, ORACLE 0.820).

## 1. Đối chứng 3 chuyển đạo ngẫu nhiên

20 bộ 3 chuyển đạo rút ngẫu nhiên (seed 0) trong 83 bộ khác V1–V3, 9 thiết lập ngoài miền, seed validation 0, ensemble 3 seed,
cùng giao thức với CNN-V1V3. Bộ (I, II, V5) mà phản biện nêu nằm trong 20 bộ.

| Đầu vào | AUROC ngoài miền, trung bình 9 thiết lập |
|---|---|
| V1–V3 | **0.861** |
| bộ ngẫu nhiên tốt nhất (V2+V3+V6) | 0.846 |
| trung vị 20 bộ ngẫu nhiên | 0.799 |
| 9 chuyển đạo (cùng split) | 0.810 |
| bộ có ≥ 1 chuyển đạo V1–V3 (n = 15) | 0.808 |
| bộ không có V1–V3 (n = 5) | 0.740 |
| I + II + V5 | 0.740 |

0/20 bộ ngẫu nhiên đạt V1–V3. Ít đầu vào hơn không tự nó làm tăng AUROC: bộ ngẫu nhiên trung vị còn kém 9 chuyển đạo.
Lợi ích đi theo chuyển đạo trước tim phải. Giới hạn: một seed validation, 20/83 bộ.

## 2. Chia đôi: chọn ngưỡng trên một nửa, đánh giá trên nửa còn lại

100 lần chia ngẫu nhiên (không dùng nhãn) × 3 seed validation, 7 thiết lập held-out.

| | CATT | UPA | GMM-2 | Otsu | MED | SRC | ORACLE |
|---|---|---|---|---|---|---|---|
| chọn A, đo B: trung bình | **0.794** | 0.786 | 0.784 | 0.784 | 0.773 | 0.702 | 0.805 |
| chọn A, đo B: tệ nhất | **0.771** | 0.762 | 0.735 | 0.735 | 0.716 | 0.548 | 0.780 |
| toàn cohort (Bảng 12): trung bình | 0.799 | 0.786 | 0.785 | 0.783 | 0.773 | 0.701 | 0.820 |

CATT gần như không đổi (−0.005). ORACLE giảm 0.015 vì Youden trên nhãn đích khớp quá mức với tập chọn ngưỡng;
khoảng cách CATT–ORACLE ngoài mẫu chỉ còn 0.011.

## 3. Khoảng tin cậy của giá trị tệ nhất (min qua 7 thiết lập held-out)

Bootstrap bệnh nhân phân tầng theo lớp; mọi ngưỡng phụ thuộc lô ước lượng lại trên mỗi mẫu.

| Tỉ lệ | Phương pháp | Tệ nhất | CI 95% | CATT − phương pháp (CI 95%) |
|---|---|---|---|---|
| tự nhiên (1000 lần) | CATT | 0.772 | 0.717–0.789 | |
| | UPA | 0.762 | 0.720–0.769 | −0.024 đến 0.035 |
| | GMM-2 | 0.751 | 0.680–0.772 | −0.016 đến 0.084 |
| | Otsu | 0.742 | 0.687–0.771 | −0.016 đến 0.076 |
| 5% (200 lần × 10 lô) | CATT | 0.739* | 0.692–0.774 | |
| | UPA | 0.720* | 0.689–0.740 | −0.017 đến 0.058 |
| | GMM-2 | 0.688* | 0.638–0.725 | **0.005 đến 0.110** |
| | Otsu | 0.691* | 0.650–0.726 | **0.004 đến 0.097** |

\* trung vị bootstrap. `worst_case_ci.csv` có thêm bản 5% chỉ lấy mẫu lại lô (bệnh nhân cố định); CI ở bản đó hẹp hơn nhiều
vì bỏ qua sai số chọn bệnh nhân, không nên dùng làm CI chính.

Kết luận: ở tỉ lệ tự nhiên, ưu thế về giá trị tệ nhất của CATT **không có ý nghĩa** so với UPA, GMM-2, Otsu.
Ở tỉ lệ 5%, CATT tệ nhất cao hơn GMM-2 và Otsu có ý nghĩa (CI loại 0, sát biên), nhưng không hơn UPA có ý nghĩa.

## 4. ST_V2 so với CBM-sd15 làm bộ gán nhãn, theo độ rộng QRS

Tam phân vị QRS_w (lớp khái niệm V1–V3, mốc khởi tạo) trong từng cohort đích; trung bình 9 thiết lập.

| Tam phân vị (QRS_w trung vị) | bacc CBM-sd15 | bacc ST_V2 | chênh (CBM tốt hơn ở) | AUROC CBM-sd15 | AUROC ST_V2 |
|---|---|---|---|---|---|
| hẹp (49 ms) | 0.741 | 0.693 | +0.049 (6/9) | 0.841 | 0.792 |
| giữa (57 ms) | 0.730 | 0.720 | +0.011 (6/9) | 0.805 | 0.821 |
| rộng (68 ms) | 0.724 | 0.701 | +0.023 (8/9) | 0.802 | 0.781 |

Lập luận "ST_V2 dễ sai khi QRS giãn rộng" **không được ủng hộ**: CBM hơn ST_V2 ở nhóm QRS rộng (+0.023) nhưng hơn nhiều hơn
ở nhóm QRS hẹp (+0.049). Không nên dùng lập luận này trong bài.
