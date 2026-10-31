# Grugada: đặc trưng ECG trong hội chứng Brugada và đột tử

Phân tích bộ dữ liệu công khai **Brugada-HUCA** (363 ECG 12 chuyển đạo, 100 Hz, 12 giây; 76 ca Brugada, 11 ca đột tử) để tìm các đặc trưng ECG liên quan đến chẩn đoán Brugada và nguy cơ đột tử.

## Cách chạy

```bash
pip install -r requirements.txt
```

```bash
python "0.tai du lieu.py" brugada-huca
```

Đặt biến môi trường `BRUGADA_DIR` trỏ tới thư mục dữ liệu (Windows PowerShell: `$env:BRUGADA_DIR="D:\...\brugada-huca"`; Linux/macOS: `export BRUGADA_DIR=$PWD/brugada-huca`). Nếu không đặt, code dùng đường dẫn ổ D: cũ trong `ecg_pipeline.py`.

| Bước | Script | Việc làm | Đầu ra |
|---|---|---|---|
| 0 | `0.tai du lieu.py` | Tải dữ liệu (một lần) | thư mục `brugada-huca/` |
| 1 | `1.doc du lieu.py` | Kiểm toán nhãn và định dạng tín hiệu | in ra màn hình, `Audit_SCD_List.csv` |
| – | `ecg_pipeline.py` | Thư viện thuật toán dùng chung (lọc, dò QRS, nhịp trung vị, điểm J, đặc trưng) | – |
| 3 | `3.trung binh các nhip tim.py [ID ...]` | Vẽ nhịp trung vị + mốc đo để kiểm tra bằng mắt | `figures/qc/*.png` |
| 4 | `4.trich xuat dac trung.py` | Trích 14 đặc trưng cho 363 ca | `Final_Research_Data.csv` |
| 5 | `5.ve bieu do va thong ke.py` | Thống kê 3 phép so sánh + biểu đồ | `results/stats_*.csv`, `figures/compare_*.png` |

Kiểm thử thuật toán trên ECG giả lập: `python -m pytest tests/`

## Thay đổi so với phiên bản đầu (và vì sao)

| Vấn đề ở bản cũ | Hậu quả | Đã sửa |
|---|---|---|
| `V1_QRS = (46 - 35) * 10` là hằng số | Cả 363 ca đều bằng 110 ms, cột vô dụng | Đo QRS thật từ onset đến điểm J trên 12 chuyển đạo |
| Dò đỉnh R trên V1 bằng `find_peaks` dương | QRS ở V1 thường âm (rS) → bắt nhầm sóng T/P. Đo lại: **46/363 ca** có quá nửa số "đỉnh R" lệch > 60 ms khỏi QRS thật, **210 ca** có ít nhất một nhịp sai bị đưa vào trung bình | Dò QRS trên năng lượng đạo hàm của cả 12 chuyển đạo |
| Khoảng cách tối thiểu giữa 2 đỉnh = 0.6 s | Bỏ sót nhịp khi tim > 100 l/p | Thời gian trơ 0.25 s |
| Điểm J cố định = R + 60 ms | Sai khi QRS rộng/hẹp, đúng là chỗ cần đo nhất trong Brugada | Điểm J xác định theo vận tốc không gian, một điểm J chung cho mọi chuyển đạo |
| Chỉ số mẫu viết cứng (40, 46, 54, 55:85) | Chỉ đúng với fs = 100 Hz | Mọi mốc tính theo giây × `fs` |
| Trung bình cộng các nhịp | Một nhịp nhiễu/ngoại tâm thu kéo lệch | Căn chỉnh tương quan chéo, loại nhịp lạc, lấy trung vị |
| `except:` trống | Lỗi bị nuốt, ca lỗi thành NaN không rõ lý do | Cột `error` ghi lý do |
| Tên file "15 đặc trưng" nhưng chỉ có 4 | – | 14 đặc trưng + 2 cột chất lượng |
| `2.thuat toan.py` thiếu `import os`, `BASE_DIR` | Không chạy được | Thành module `ecg_pipeline.py` |
| Chỉ một p-value Mann-Whitney cho 11 ca đột tử | Không biết hiệu ứng lớn cỡ nào, không hiệu chỉnh nhiều phép kiểm định | AUC + KTC 95% bootstrap, q-value Benjamini-Hochberg, 3 phép so sánh riêng |

## Kết quả chính (pipeline mới, 363/363 ca xử lý thành công)

**A. Brugada (n=76) vs bình thường (n=287)**: nhiều đặc trưng khác biệt rõ, q < 0.001:

| Đặc trưng | Brugada, median [IQR] | Bình thường | AUC [KTC 95%] |
|---|---|---|---|
| QRS (ms) | 90 [80; 100] | 80 [70; 80] | 0.76 [0.69–0.82] |
| V2_J (mV) | 0.24 [0.15; 0.38] | 0.13 [0.03; 0.22] | 0.72 [0.65–0.78] |
| V1_Slope (mV/s) | −1.25 [−2.17; −0.67] | −0.37 [−1.29; 0.35] | 0.30 [0.24–0.37] (thấp hơn ở Brugada) |
| V1_J (mV) | 0.15 [0.08; 0.21] | 0.09 [0.03; 0.14] | 0.68 [0.61–0.74] |
| Sóng S ở DI (ms) | 33 [24; 42] | 24 [16; 33] | 0.67 [0.59–0.74] |

ST chênh lên ở điểm J rồi dốc xuống ở V1–V2, QRS rộng hơn và sóng S ở DI dài hơn đúng với kiểu hình Brugada type 1 và giả thuyết chậm dẫn truyền đường ra thất phải.

**B. Đột tử (n=11) vs không (n=352)**: **không đặc trưng nào có ý nghĩa sau hiệu chỉnh** (mọi q > 0.4). Tín hiệu mạnh nhất: QRS (AUC 0.68 [0.50–0.83]) và thời gian sóng S ở DI (AUC 0.64 [0.46–0.80]), cùng chiều với y văn (Calò 2016) nhưng KTC chạm 0.5. `basal_pattern` liên quan đến đột tử: OR 4.2, Fisher p = 0.038.

**C. Đột tử trong nhóm Brugada (5 vs 71)**: quá ít ca để kết luận; chỉ mô tả.

**Với kết quả cũ**: V1_Slope p = 0.156 (cũ) → p = 0.54 (mới) khi so đột tử. Không có thay đổi nào từ "không ý nghĩa" sang "có ý nghĩa". Câu hỏi đột tử cần cohort lớn hơn chứ không cần thêm đặc trưng.

Chi tiết: `results/stats_*.csv`. Tổng quan tài liệu: `docs/tong-quan-tai-lieu.md`.

## Giới hạn

- 100 Hz → độ phân giải thời gian 10 ms; QRS và sóng S chỉ đo được theo bậc 10 ms (S có nội suy).
- Không có V1–V2 ở khoang liên sườn cao (2–3), nơi type 1 hay lộ ra nhất.
- Điểm J tự động: đã kiểm tra bằng mắt trên các ca đột tử (`figures/qc/`), nhưng chưa đối chiếu với đo tay của bác sĩ.
- Ý nghĩa nhãn `brugada = 2` (7 ca) không có trong tài liệu; đang gộp vào nhóm Brugada (khớp con số 76 của PhysioNet).
- 11 ca đột tử: không đủ để xây mô hình dự báo đa biến (Riley et al., BMJ 2020).

## Trích dẫn dữ liệu

Costa Cortez N, Garcia Iglesias D. *Brugada-HUCA: 12-Lead ECG Recordings for the Study of Brugada Syndrome* (v1.0.0). PhysioNet, 2026. https://physionet.org/content/brugada-huca/1.0.0/
