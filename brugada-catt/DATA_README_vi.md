# Dữ liệu đi kèm để chạy lại phân tích

Các tệp dưới đây được đưa lên để phiên Claude Code trên cloud chạy được các phân tích còn lại mà không cần tải dữ liệu gốc.

| Tệp | Nội dung | Nguồn |
|---|---|---|
| `harmonized/huca.npz` | 356 bản ghi HUCA đã chuẩn hóa: `X` (356, 9, 75), `y`, `ids`, `nbeats`, `leads`, `fs` | Brugada-HUCA 1.0.0, PhysioNet |
| `harmonized/claris.npz` | 1.154 bản ghi Melo-Claris: `X` (1154, 9, 75), `y`, `ids`, `leads`, `fs` | Melo và cs. 2023, Dryad (doi:10.5061/dryad.s1rn8pkd9) |
| `harmonized/mortara.npz` | 370 bản ghi Melo-Mortara: `X` (370, 9, 75), `y`, `ids`, `leads`, `fs` | như trên |
| `huca_meta/metadata.csv` | Bảng nhãn gốc của HUCA (`patient_id`, `basal_pattern`, `sudden_death`, `brugada`) | Brugada-HUCA 1.0.0, PhysioNet |
| `concepts_v5/v5_raw_0.jsonl`, `v5_raw_1.jsonl` | Kết quả từng lượt (9 thiết lập × 3 seed validation) của `compare_v5.py`: điểm validation và điểm miền đích của CNN-V1V3, các CBM và Hybrid | Sinh bởi `concepts_v5/compare_v5.py` |

Định dạng chung (xem `harmonized/harmonize.py`): một nhịp đại diện mỗi bệnh nhân, 9 chuyển đạo (I, II, III, V1–V6), 100 Hz, 75 mẫu (−290 đến +460 ms quanh QRS), đỉnh năng lượng QRS ở mẫu 28, mỗi chuyển đạo z-score.

## Lưu ý

- **Nhãn Melo (`y` trong `claris.npz`, `mortara.npz`) là nhãn tái tạo**, không phải nhãn gốc của tác giả. Chúng được suy ra từ điểm dự đoán đã công bố và đường ROC đã công bố. Cách tái tạo khớp chính xác số ca mỗi lớp và số TP/TN tại ngưỡng 0.5 đã công bố. Xem `melo_processed/README_vi.md` và mục 2.1 của bản thảo.
- `huca.npz` là bản chuẩn hóa gốc dùng trong bài (chạy lại `harmonize.py` trên dữ liệu PhysioNet cho kết quả trùng khớp, chênh 0.0).
- `v5_raw_*.jsonl`: 25 lượt trong `v5_raw_0.jsonl` lấy từ lần chạy gốc; 2 lượt còn thiếu (thiết lập 8, HUCA + Claris → Mortara, seed validation 0 và 2) được chạy lại bằng đúng `compare_v5.py` và ghi vào `v5_raw_1.jsonl`.
- Khi dùng lại dữ liệu, trích dẫn nguồn gốc: Brugada-HUCA trên PhysioNet (theo giấy phép ghi trên trang dataset) và Melo và cs. 2023 (PNAS Nexus; dữ liệu trên Dryad).

## Chạy các việc còn lại (từ thư mục `brugada-catt`)

```
python analysis_v8/logit_shift.py concepts_v5 harmonized huca_meta analysis_v8
python concepts_v3/stress_ada.py harmonized concepts_v3 v3
```

CNN 9 chuyển đạo cho cột "HUCA gốc" của Bảng 15: xem ghi chú trong `sensitivity_v7/README_vi.md` và bản thảo.
