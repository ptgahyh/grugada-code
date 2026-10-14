# Bước 8: rà soát kiểu reviewer và các phân tích bổ sung

Mục tiêu: tìm và vá những chỗ một reviewer khó tính sẽ hỏi, trước khi nộp bài. Có ba phát hiện làm **đổi cách viết bài**, và năm phân tích làm bài **chắc hơn**.

---

## A. Ba phát hiện làm đổi cách viết

### 1. Nhãn HUCA khác với cách mình mô tả trước đây (`huca_label_check.py`)

Theo trang PhysioNet:
- Cột `brugada` là **chẩn đoán hội chứng Brugada**, không phải "dạng ECG Brugada do chuyên gia đọc".
- Cột `basal_pattern` mới là "ECG nền có dạng Brugada hay không".
- Giá trị `brugada = 2` có ở 7 bản ghi nhưng **trang không định nghĩa**.

Trong 356 bản ghi mình dùng:

| | Âm tính | Dương tính |
|---|---|---|
| Không có dạng Brugada trên ECG nền | 260 | **52** |
| Có dạng Brugada trên ECG nền | **23** | 21 |

Nghĩa là:
- 71% ca dương tính ở HUCA **không** có dạng Brugada trên ECG nền.
- Câu cũ trong bài ("HUCA đọc theo dạng ECG, Melo 83% lộ ra nhờ ajmaline → khác định nghĩa nhãn") **sai về cơ sở**. Cả hai cohort đều phần lớn là Brugada "ẩn".

Đã sửa trong bản thảo:
- Mục 2.1: mô tả đúng ba cột nhãn, ghi rõ nhãn 2 không được định nghĩa.
- Mục 3.9: bỏ giải thích "khác định nghĩa nhãn", thay bằng bằng chứng từ phân rã khái niệm (xem B4).

Kiểm tra độ nhạy (`huca_subgroups.py`): bỏ 7 ca nhãn 2 làm AUROC đổi tối đa 0.011, CATT đổi tối đa 0.008. Kết luận không đổi.

### 2. CATT không còn "bỏ xa" mọi phương pháp không nhãn

Mình bổ sung các đối thủ mạnh nhất trong y văn mà bước 6 chưa có:
- **UPA** (Roschewitz và cs., *Nature Communications* 2023): khớp toàn bộ phân phối điểm đích về nguồn. Đây là công trình gần CATT nhất trong y tế; bài báo **phải** trích và so.
- **GMM-2, Otsu**: đặt ngưỡng không giám sát trên điểm đích.
- **DANN, Deep CORAL**: thích nghi miền ngay lúc huấn luyện, dùng ECG đích không nhãn.

Kết quả trên 7 thiết lập held-out (balanced accuracy, trung bình / tệ nhất):

| Phương pháp | Tỉ lệ bệnh tự nhiên | Tỉ lệ bệnh 5% |
|---|---|---|
| SRC (ngưỡng nguồn) | 0.701 / 0.548 | 0.703 / 0.548 |
| MED | 0.773 / 0.713 | 0.754 / 0.732 |
| UPA | 0.786 / 0.762 | 0.751 / 0.728 |
| GMM-2 | 0.785 / 0.751 | 0.774 / 0.686 |
| Otsu | 0.783 / 0.742 | 0.772 / 0.687 |
| AdaBN | 0.765 / 0.714 | 0.767 / 0.714 |
| TENT | 0.767 / 0.715 | 0.769 / 0.715 |
| DANN | 0.703 / 0.537 | 0.705 / 0.537 |
| Deep CORAL | 0.701 / 0.533 | 0.703 / 0.533 |
| **CATT** | **0.799 / 0.772** | **0.790 / 0.744** |
| ORACLE | 0.820 / 0.802 | 0.842 / 0.821 |

Hiệu số gộp (bootstrap theo bệnh nhân, 1000 lần, tỉ lệ bệnh tự nhiên; tất cả khoảng tin cậy đều không chứa 0):

| CATT trừ | Hiệu số [KTC 95%] |
|---|---|
| UPA | +0.013 [0.003, 0.022] |
| GMM-2 | +0.015 [0.002, 0.027] |
| Otsu | +0.016 [0.002, 0.027] |
| AdaBN / TENT | +0.032–0.034 |
| DANN / CORAL | +0.096–0.098 |

**Cách viết mới (trung thực):**
- Ở tỉ lệ bệnh tự nhiên, UPA và GMM-2 **gần bằng** CATT (kém 0.013–0.015).
- Ưu thế của CATT nằm ở hai chỗ:
  1. **Bền với tỉ lệ bệnh.** Ở 5%, UPA và MED tụt xuống 0.75, vì chúng giả định tỉ lệ bệnh không đổi.
  2. **Trường hợp tệ nhất.** GMM-2 và Otsu có trường hợp tệ nhất chỉ 0.686–0.687 ở 5%, còn CATT là 0.744.
- CATT là phương pháp duy nhất đứng cao ở cả hai tiêu chí, ở cả hai mức tỉ lệ bệnh.
- Đã viết lại toàn bộ mục 3.7 theo hướng này.

### 3. Cơ chế "độc lập có điều kiện" cần nói chính xác hơn (`sim_catt.py`)

Mô phỏng có kiểm soát cho thấy:
- Chỉ riêng việc **phụ thuộc** (ρ = 0.9) **không** làm hại, nếu ngưỡng của bộ gán nhãn còn đúng (khoảng cách tới oracle 0.004).
- Riêng việc **ngưỡng của bộ gán nhãn bị lệch** cũng chỉ thêm nhiễu mẫu, nếu bộ gán nhãn độc lập (0.009 → 0.040, và giảm khi n tăng).
- **Kết hợp cả hai** mới tạo ra **thiên lệch hệ thống**: khoảng cách 0.03 → 0.15, và **không giảm khi n tăng** (giống nhau ở n = 356 và n = 1154).
- Con số 0.15 khớp đúng với CNNalt trên dữ liệu thật (0.15–0.23). Đó là trường hợp ngưỡng nguồn của nó lệch cùng chiều với CNN, gán 80–91% âm tính HUCA thành dương.

Đã thêm vào bài: mục 2.9 (mô tả mô phỏng), mục 3.8 và Hình 6a.

---

## B. Năm phân tích làm bài chắc hơn

### 1. Không có bệnh nhân trùng giữa Claris và Mortara (`overlap_check.py`)

Hai máy cùng một trung tâm, nên có thể cùng bệnh nhân được đo hai lần, tức rò rỉ dữ liệu. Kết quả kiểm tra:
- Tương quan cao nhất giữa một nhịp Mortara và nhịp Claris giống nó nhất là **0.87**.
- Mức này bằng giữa HUCA và Claris (0.87), thấp hơn hẳn láng giềng trong cùng cohort (0.90–0.92) và thấp hơn mốc "cùng bệnh nhân" (0.96).
- Không có cặp nào vượt 0.95. Đã thêm vào mục 3.1.

### 2. Khoảng tin cậy đúng nghĩa cho CATT (`v8_core.py`, `v8_analyze.py`)

- Huấn luyện lại CNN-V1V3: tái lập điểm bước 5 với sai khác ≤ 6×10⁻⁷. Lần này lưu điểm từng bệnh nhân.
- So với ngưỡng nguồn: CATT hơn 0.17–0.25 ở cả 5 thiết lập bị hỏng (Holm p < 0.001), và không làm hại 4 thiết lập còn lại.
- Thay kiểm định Wilcoxon trên 7 thiết lập (lực thấp) bằng bootstrap theo bệnh nhân, cho từng thiết lập và gộp.

### 3. Cần bao nhiêu ECG không nhãn ở site mới?

Ngưỡng chọn trên n ECG, đánh giá trên toàn cohort:

| n | 20 | 50 | 100 | 200 | toàn bộ |
|---|---|---|---|---|---|
| CATT (phân vị 10%) | 0.766 (0.725) | 0.782 (0.753) | 0.789 (0.771) | 0.795 (0.783) | 0.799 |

Khoảng **50–100 ECG không nhãn** là đủ. Ở n rất nhỏ (20–30), UPA ngang CATT. Xem Hình 6b.

### 4. CBM học đúng hướng lâm sàng và giải thích được vì sao ngưỡng lệch (`cbm_interp.py`)

- **Trọng số ổn định ở cả 9/9 thiết lập.**
  - Dương: QRS_w (lớn nhất), J_elev V1 và V2, r′ V2, Corrado.
  - Âm: độ dốc ST, sóng T ở V2 và V3, β.
  - Đây chính là dạng type-1 "vòm": J nâng, ST dốc xuống, T âm.
- **Độ dịch logit của nhóm âm tính khi sang HUCA** (tính theo đơn vị độ tách lớp của chính mô hình):

  | Mô hình | Độ dịch |
  |---|---|
  | CNN-V1V3 | 0.93–1.06 (âm tính bị đẩy gần đúng vào chỗ dương tính nguồn) |
  | CBM-sd15 | 0.21–0.32 |
  | CBM | 0.27–0.55 |

  Đây là con số định lượng cho câu "mô hình khái niệm giữ ngưỡng ổn định hơn".
- **Phân rã chính xác theo từng khái niệm:** phần lớn độ dịch đến từ QRS_w (cao hơn ở HUCA, ở cả hai lớp) và mức ST ở V2. Kết quả này nối thẳng với phát hiện về QRS ở bước 5.
- 23 ca âm tính có dạng Brugada nền **không** phải nguyên nhân: bỏ chúng đi, độ dịch chỉ đổi 2–14%.

### 5. Phân nhóm trong HUCA (`huca_subgroups.py`)

- Brugada **ẩn** (52 ca không có dạng nền) vẫn được phát hiện tốt: AUROC 0.84–0.89. Ca có dạng nền: 0.88–0.93.
- Báo động giả tập trung ở **23 ca "giống Brugada nhưng không phải"**. Ở ngưỡng CATT, 32–64% nhóm này bị báo dương, so với 17–34% ở các ca âm tính khác. Về lâm sàng, đây cũng là những ECG bác sĩ sẽ khoanh lại.

---

## C. Rà soát y văn: công trình liên quan phải trích

- **Roschewitz và cs., Nat Commun 2023 (UPA).** Gần nhất về ý tưởng "sửa điểm làm việc không cần nhãn". Đã so trực tiếp; điểm khác là UPA giả định tỉ lệ bệnh không đổi, còn CATT thì không.
- **arXiv 2607.03009 (7/2026), "Do ECG foundation models transfer to rare cardiac diseases? Evidence from Brugada syndrome".**
  - Dùng BrSwiss và HUCA, thấy chuyển miền gần như ngẫu nhiên. Củng cố động cơ bài mình.
  - Họ **không** làm khái niệm, không giới hạn chuyển đạo, không chỉnh ngưỡng. Bài mình bổ sung đúng các chỗ đó.
- **Saleh và cs., PLOS Digit Health 2026.** Brugada với dữ liệu ít, chỉ đánh giá nội bộ.
- Mình chưa thấy công trình nào dùng **mô hình thứ hai, cấu trúc khác, để chọn lại ngưỡng Youden** dựa trên bất biến nhiễu nhãn của Menon 2015. Tính mới của CATT vẫn đứng được, nhưng phải đặt cạnh UPA và co-training.

---

## D. Tệp

| Tệp | Nội dung |
|---|---|
| `huca_label_check.py` → `huca_label_check.json`, `huca_concepts_by_label.csv` | Thành phần nhãn HUCA, RR_CV, đột tử thô và phân tầng |
| `overlap_check.py` → `overlap_check.json` | Kiểm tra trùng bệnh nhân Claris và Mortara |
| `v8_core.py` → `scores/scores_<thiết lập>_<seed>.npz` | Điểm từng bệnh nhân: CNN, AdaBN, TENT, DANN, CORAL, CBM, ST_V2 |
| `v8_analyze.py` → `v8_point_*.csv`, `v8_bootstrap_*.csv`, `v8_prev5_*.csv` | So sánh phương pháp, bootstrap, tỉ lệ bệnh 5% |
| `size_curve.py` → `v8_sample_size_*.csv` | Cỡ lô đích cần thiết |
| `cbm_interp.py` → `cbm_interp_*.json`, `cbm_weights_summary.csv`, `cbm_shift_decomposition.csv` | Trọng số CBM và phân rã độ dịch |
| `huca_subgroups.py` → `huca_subgroups.csv` | Nhóm con HUCA, loại nhãn 2 |
| `sim_catt.py` → `sim_catt*.csv`, `sim_catt_summary.json` | Mô phỏng lý thuyết |
| `../figures_en/make_fig6.py` → `fig6_simulation_samplesize.png/pdf` | Hình 6 |

Chạy lại (khoảng 45 phút trên 2 CPU):
```
python v8_core.py ../harmonized ../concepts_v5 scores 0 2 & python v8_core.py ../harmonized ../concepts_v5 scores 1 2
NBOOT=1000 NDRAW=200 python v8_analyze.py scores .
NSIZE=100 python size_curve.py scores .
python cbm_interp.py ../harmonized ../concepts_v5 .
python overlap_check.py ../harmonized ../sensitivity_v7 .
python huca_label_check.py <thư mục HUCA> ../harmonized .
python huca_subgroups.py <thư mục HUCA> ../harmonized scores .
python sim_catt.py .
```
