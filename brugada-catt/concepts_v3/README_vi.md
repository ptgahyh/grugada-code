# Bước 3: stress test cho Ada, chẩn đoán Mortara, so sánh lại công bằng

## 1. Ada có bền khi tỉ lệ Brugada thay đổi không? (`stress_ada.py`)

**Cách làm:**
- Huấn luyện lại Sign (không thích nghi), SignAda (trung bình/độ lệch chuẩn) và SignAda-robust (trung vị/IQR) trên 9 thiết lập ngoài miền.
- Rút các lô 100 bệnh nhân từ miền test với tỉ lệ Brugada 5–70%, mỗi mức 30 lần.
- Ngưỡng cố định, chọn trên validation.

**Balanced accuracy trung bình trên 9 thiết lập:**

| Tỉ lệ Brugada trong lô | 5% | 10% | 20% | 35% | 50% | 70% |
|---|---|---|---|---|---|---|
| Sign | 0.726 | 0.716 | 0.721 | 0.722 | 0.720 | 0.719 |
| SignAda | 0.719 | 0.713 | 0.728 | **0.740** | **0.740** | 0.736 |
| SignAda-robust | 0.721 | 0.713 | 0.729 | 0.733 | 0.733 | 0.724 |

**Kết luận:**
- Ada chỉ có lợi khi tỉ lệ Brugada ≥ 20%. Ở 5–10%, tức tình huống sàng lọc thật, Ada **hơi kém hơn** Sign.
- Lý do: Ada đổi chỗ độ nhạy và độ đặc hiệu theo tỉ lệ bệnh. Ở tỉ lệ 5%: độ nhạy 0.86, độ đặc hiệu 0.57. Ở tỉ lệ 70%: độ nhạy 0.61, độ đặc hiệu 0.87.
- Sign (không thích nghi) giữ ổn định quanh 0.72 ở mọi tỉ lệ. AUROC hầu như không đổi.
- Chuẩn hóa bền vững (trung vị/IQR) không giúp.
- **Cỡ lô không phải vấn đề:** Ada vẫn ổn với lô chỉ 20 bệnh nhân (`stress_summary.csv`).

**Kết quả âm tính (`ada_ctrl.py`):** mình đã thử chuẩn hóa theo nhóm âm tính, ở miền mới thì ước lượng nhóm âm tính giả. Cách này **thất bại**.
- HUCA + Claris → Mortara: độ đặc hiệu chỉ 0.34 (cứng) và 0.41 (mềm), do lệch cắt cụt khi chọn nhóm âm tính giả.
- Ngay cả khi biết đúng nhóm âm tính thật, độ nhạy và độ đặc hiệu vẫn thiếu nhất quán giữa các thiết lập.
- Vì vậy cách này không được đưa vào stress test.

**Khuyến nghị cho bài báo:**
- Dùng Sign (không thích nghi) cho sàng lọc tỉ lệ thấp.
- Chỉ dùng Ada cho quần thể chuyển tuyến có tỉ lệ khoảng 20–50%.
- Báo cáo rõ hạn chế này.

## 2. Vì sao Mortara khó? (`diag_mortara.py`)

**Phát hiện lớn nhất: CNN chỉ dùng V1–V3 chuyển miền tốt hơn hẳn CNN dùng 9 chuyển đạo.**

| | CNN 9 chuyển đạo | CNN V1–V3 | CNN 9, xóa ST-T | CNN 9, chỉ QRS |
|---|---|---|---|---|
| AUROC TB ngoài miền (9) | 0.805 | **0.866** | 0.794 | 0.802 |

Từ bảng trên:
- Xóa ST-T hay chỉ giữ QRS **gần như không làm CNN 9 chuyển đạo giảm**. Nghĩa là CNN 9 chuyển đạo dựa chủ yếu vào thông tin ngoài đoạn ST, và thông tin đó không chuyển được sang miền khác.
- Giới hạn vào V1–V3, đúng vùng sinh lý của Brugada, tăng AUROC ở 8/9 thiết lập. Ví dụ Mortara → Claris: 0.733 → 0.865.

**Dịch chuyển khái niệm Claris → Mortara** (cùng bệnh viện, khác máy):
- `QRS_w` dịch chuyển rất mạnh (SMD −0.98 ở nhóm âm tính), trong khi các khái niệm khác chỉ khoảng 0.3.
- `QRS_w` lại là khái niệm có trọng số lớn nhất trong CBM. Đây là ứng viên chính giải thích vì sao CBM yếu ở Mortara.
- `QRS_w` được tính từ năng lượng của cả 9 chuyển đạo, nên nhạy với máy đo.

**Phân tích lỗi (HUCA + Claris → Mortara, 137 ca Brugada):**

| Nhóm | Số ca | Đặc điểm |
|---|---|---|
| Cả CNN và CBM bắt được | 91 | Đặc điểm Brugada rõ: J_elev V2 0.30, Corrado V2 0.62 |
| Chỉ CNN bắt được | 17 | Đặc điểm cổ điển yếu: J_elev V2 0.11 |
| Chỉ CBM bắt được | 5 | |
| Cả hai bỏ sót | 24 | Khái niệm gần như bình thường. Nhiều khả năng là Brugada ẩn, chỉ lộ ra khi test ajmaline; ECG nền không đủ thông tin |

## 3. So sánh lại công bằng (`compare_v3.py`)

9 thiết lập ngoài miền, 3 seed validation, bootstrap ghép cặp có hiệu chỉnh Holm.

| Mô hình | AUROC TB | Balanced acc TB | Độ nhạy TB | Độ đặc hiệu TB |
|---|---|---|---|---|
| ST_V2 | 0.808 | 0.729 | 0.625 | 0.832 |
| CNN 9 chuyển đạo | 0.816 | 0.642 | 0.634 | 0.650 |
| **CNN V1–V3** | **0.869** | 0.676 | 0.707 | 0.645 |
| CBM2-SignAda | 0.826 | **0.733** | 0.670 | 0.796 |
| **Hybrid** (trung bình xác suất CNN V1–V3 và CBM2-SignAda) | 0.864 | 0.729 | 0.704 | 0.753 |
| CBM2-SignAda-noShift (chỉ LODO) | 0.807 | 0.726 | — | — |

**So sánh cặp (Holm):**

| So sánh | Thắng | Thua | Hòa |
|---|---|---|---|
| CBM2-SignAda với CNN V1–V3 | 0 | 5 | 4 |
| Hybrid với CNN V1–V3 | 0 | 0 | 9 |
| Hybrid với CNN 9 chuyển đạo | 4 | 0 | 5 |

## Nhìn lại một cách trung thực

1. **Mô hình khái niệm một mình KHÔNG phải mô hình có AUROC cao nhất.** CNN V1–V3 tốt hơn có ý nghĩa ở 5/9 thiết lập. Lợi thế AUROC của CBM so với CNN 9 chuyển đạo ở bước 2 phần lớn đến từ việc **chỉ nhìn V1–V3**, không phải từ khái niệm.
2. **Thế mạnh thật của mô hình khái niệm là ngưỡng ổn định khi đổi miền**, cùng khả năng giải thích. CNN V1–V3 có AUROC cao nhưng ngưỡng sụp, ví dụ Claris → HUCA balanced accuracy 0.53.
3. **Hybrid lấy được cả hai:** AUROC ngang CNN V1–V3 (không khác biệt ở 9/9) và balanced accuracy ngang mô hình khái niệm (0.729 so với 0.676).
4. **Lọc khái niệm theo dịch chuyển giữa miền (noShift) làm kém đi**, vì loại quá nhiều khái niệm (chỉ còn 8).

## Câu chuyện bài báo đề xuất (sau bước 3)

> Mô hình học sâu trên ECG Brugada dựa vào thông tin ngoài vùng sinh lý (V1–V3, đoạn ST), và thông tin đó không chuyển được giữa bệnh viện và máy đo. Ràng buộc theo tri thức sinh lý ở hai mức — đầu vào giới hạn V1–V3 và đầu ra qua khái niệm khả vi bất biến — cho AUROC cao nhất và ngưỡng vẫn dùng được ở miền mới, kèm giải thích theo ngôn ngữ lâm sàng.

## Việc tiếp theo

- **Hybrid với Sign (không Ada)**, để dùng được cho sàng lọc tỉ lệ thấp.
- **Thay `QRS_w` toàn cục bằng `QRS_w` tính riêng trên V1–V3**, rồi kiểm tra lại trên Mortara.
- **Kiểm chứng điểm J trên LUDB** (đang chờ dữ liệu).
- **Lặp lại so sánh CNN V1–V3 với nhiều kiến trúc khác**, để chắc kết quả không phụ thuộc vào một CNN cụ thể.

## File

| File | Nội dung |
|---|---|
| `cbm_lib.py` | Thư viện dùng chung (tái lập đúng kết quả v2) |
| `stress_ada.py`, `stress_*.csv`, `stress_raw.json`, `fig_v3_prevalence.png` | Mục 1 |
| `ada_ctrl.py` | Thử nghiệm Ada theo nhóm âm tính (kết quả âm tính) |
| `diag_mortara.py`, `diag_mortara.json`, `mortara_groups.npz`, `fig_v3_cnn_restriction.png` | Mục 2 |
| `compare_v3.py`, `analyze_v3.py`, `v3_*.csv`, `v3_raw.json`, `fig_v3_compare.png` | Mục 3 |

Chạy lại (khoảng 45 phút trên CPU 2 lõi):
```
python stress_ada.py <harmonized> .
python diag_mortara.py <harmonized> <concepts_v2> .
python compare_v3.py <harmonized> <concepts_v2> .
python analyze_v3.py . <harmonized>
```
Thư mục `concepts_v3` cần nằm cạnh thư mục `concepts_v2`, vì `cbm_lib.py` nạp `concepts2.py` từ đó.
