# Bước 7: phân tích độ nhạy — kết luận có đổi không nếu nhãn hoặc tiền xử lý khác đi?

Hai điểm yếu reviewer chắc chắn sẽ hỏi (README bước 5 và 6):
1. **Nhãn Melo là nhãn tái tạo.** Nếu một phần nhãn sai thì sao?
2. **Bất đối xứng tiền xử lý.** Nhịp trung vị HUCA do mình tính ở 100 Hz, còn của Melo do máy đo tính ở tần số cao. Liệu lệch ngưỡng HUCA ↔ Melo có phải do chính quy trình của mình gây ra?

**Trả lời ngắn: cả hai đều không làm đổi kết luận.**

---

## 1. Độ nhạy với nhãn Melo (`label_sens.py`)

**Kịch bản:**
- Nhãn "thật" của Claris/Mortara khác nhãn tái tạo ở p% bệnh nhân (lật ngẫu nhiên, 20 lần rút).
- Mô hình giữ nguyên, vì trong thực tế chúng đã được huấn luyện trên nhãn tái tạo; chỉ phép đánh giá trên miền đích Melo dùng nhãn "thật".
- Đo trên 6 thiết lập có miền đích là Melo. Các thiết lập có đích HUCA không bị ảnh hưởng.

Kết quả trung bình 6 thiết lập, CNN-V1V3:

| Tỉ lệ nhãn sai | AUROC | SRC | CATT | ORACLE | CATT − SRC |
|---|---|---|---|---|---|
| 0% (như bài) | 0.860 | 0.733 | 0.791 | 0.804 | +0.058 |
| 2% | 0.844 | 0.723 | 0.778 | 0.791 | +0.055 |
| 5% | 0.821 | 0.707 | 0.759 | 0.772 | +0.052 |
| 10% | 0.782 | 0.683 | 0.729 | 0.741 | +0.046 |

Ở hai thiết lập ngưỡng nguồn hỏng (HUCA → Claris, HUCA → Mortara), mức tăng của CATT vẫn là:
- +0.17 / +0.16 khi 2% nhãn sai.
- +0.16 / +0.15 khi 5% nhãn sai.
- +0.14 / +0.13 khi 10% nhãn sai.

Ở các thiết lập còn lại, thay đổi vẫn chỉ ±0.01.

**Đọc kết quả:**
- Nhãn sai làm **mọi con số tuyệt đối giảm đều nhau**, vì nhãn nhiễu kéo AUROC đo được về 0.5.
- **Mọi so sánh tương đối giữ nguyên**: CATT vẫn hơn SRC đúng ở các chỗ hỏng, vẫn cách oracle 0.01–0.02, và vẫn không làm hại chỗ đã ổn.
- Lưu ý: kịch bản này chỉ làm nhiễu nhãn đánh giá. Nếu có nhãn gốc, mô hình huấn luyện lại trên nhãn đúng có thể còn tốt hơn. Vì vậy đây là kịch bản thận trọng.

## 2. Độ nhạy với cách tính nhịp trung vị HUCA (`harmonize_hires.py`, `sens_huca.py`)

**Quy trình thay thế**, mô phỏng cách máy ECG tính nhịp trung vị:
1. Tăng tín hiệu lên 500 Hz.
2. Tìm đỉnh QRS ở 500 Hz.
3. Căn từng nhịp bằng tương quan chéo với khuôn mẫu (±20 ms, bước 2 ms, lặp 2 lần).
4. Lấy trung vị ở 500 Hz, rồi hạ về 100 Hz.

**Kiểm tra trước khi dùng:**
- Quy trình gốc chạy lại trên bản HUCA trong ổ D tạo ra đúng `huca.npz` đang dùng (chênh 0.0).
- Bản mới có tương quan trung vị 0.966 với bản gốc, sau khi bù lệch căn đỉnh ±1–3 mẫu ở 26% bản ghi.
- Độ lệch căn chỉnh còn lại giữa các nhịp (SD) là 0.57 ms.

Sau đó huấn luyện lại 7 thiết lập có HUCA, cùng 3 seed như bước 5. Kết quả trung bình 7 thiết lập:

| | HUCA gốc | HUCA kiểu máy đo |
|---|---|---|
| AUROC CNN-V1V3 | 0.860 | 0.862 |
| AUROC CNN 9 chuyển đạo | — | 0.819 |
| AUROC CBM5-Sign | 0.804 | 0.805 |
| Balanced accuracy, SRC | 0.633 | 0.640 |
| Balanced accuracy, CATT | 0.783 | 0.786 |
| Balanced accuracy, ORACLE | 0.802 | 0.805 |

Từng thiết lập (SRC → CATT, HUCA gốc / kiểu máy đo):

| Thiết lập | HUCA gốc | HUCA kiểu máy đo |
|---|---|---|
| HUCA → Claris | 0.590 → 0.770 | 0.593 → 0.772 |
| HUCA → Mortara | 0.571 → 0.741 | 0.572 → 0.752 |
| Claris → HUCA | 0.549 → 0.772 | 0.557 → 0.771 |
| Mortara → HUCA | 0.575 → 0.791 | 0.599 → 0.787 |
| Claris + Mortara → HUCA | 0.548 → 0.797 | 0.555 → 0.806 |
| HUCA + Mortara → Claris | 0.793 → 0.787 | 0.787 → 0.798 |
| HUCA + Claris → Mortara | 0.808 → 0.823 | 0.816 → 0.816 |

**Đọc kết quả:**
1. **Lệch ngưỡng HUCA ↔ Melo không do quy trình của mình.** Ngưỡng nguồn vẫn sụp (0.55–0.60) khi tính nhịp trung vị kiểu máy đo. Nguồn gốc nhiều khả năng là khác biệt quần thể và định nghĩa nhãn:
   - HUCA: dạng ECG Brugada do chuyên gia đọc trên ECG nền.
   - Melo: 83% ca dương tính chỉ lộ ra khi test ajmaline.

   Không có thêm dữ liệu thì chưa tách được hai yếu tố này. Bài báo cần nêu rõ.
2. **Hiệu ứng V1–V3 giữ nguyên:** CNN chỉ dùng V1–V3 vẫn hơn CNN 9 chuyển đạo (0.862 so với 0.819).
3. **CATT giữ nguyên:** vẫn sửa đúng các chỗ hỏng và vẫn sát oracle.

---

## Tệp

| Tệp | Nội dung |
|---|---|
| `label_sens.py` → `label_sens_summary.csv`, `label_sens_per_setting.csv` | Độ nhạy với nhãn Melo |
| `harmonize_hires.py` → `huca_hires.npz` | Nhịp trung vị HUCA kiểu máy đo (500 Hz, căn tương quan chéo) |
| `sens_huca.py` → `sens_huca_0.jsonl`, `sens_huca_1.jsonl` | Huấn luyện lại 7 thiết lập × 3 seed |
| `compare_sens.py` → `sens_huca_per_setting.csv`, `sens_huca_summary.csv` | So sánh với HUCA gốc |

Chạy lại (khoảng 20 phút trên 2 CPU; cần dữ liệu thô HUCA và các thư mục `../harmonized`, `../concepts_v4`, `../concepts_v5`):
```
python label_sens.py ../concepts_v5 ../harmonized .
python harmonize_hires.py <thư mục HUCA có metadata.csv và files/> ../harmonized .
python sens_huca.py ../harmonized huca_hires.npz . 0 2  &  python sens_huca.py ../harmonized huca_hires.npz . 1 2
python compare_sens.py . ../concepts_v5
```

**Bổ sung (15/10/2026):** cột "HUCA gốc" của CNN 9 chuyển đạo trong Bảng 15 được tính bằng đúng `sens_huca.py`,
thay `huca_hires.npz` bằng `harmonized/huca.npz` gốc (kết quả trong `orig_huca_run/`). CNN-V1V3 chạy lại trùng
`v5_raw` đến 4 chữ số; AUROC CNN9 trung bình 7 thiết lập = 0.817.
```bash
cd sensitivity_v7 && python sens_huca.py ../harmonized ../harmonized/huca.npz orig_huca_run 0 2   # và chunk 1
```
