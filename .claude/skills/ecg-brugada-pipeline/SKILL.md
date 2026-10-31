---
name: ecg-brugada-pipeline
description: Quy trình xử lý ECG 12 chuyển đạo cho nghiên cứu Brugada/đột tử. Dùng khi đọc file WFDB, lọc tín hiệu, dò QRS, tạo nhịp trung vị, xác định điểm J, đo ST/J/sóng S/aVR, thêm đặc trưng mới vào ecg_pipeline.py, hoặc debug khi đặc trưng ECG ra giá trị lạ.
---

# Xử lý ECG cho nghiên cứu Brugada

## Trước khi viết code
1. Đọc `ecg_pipeline.py`. Mọi thuật toán phải nằm ở đó; script đánh số chỉ gọi hàm.
2. Kiểm tra `fs` thật bằng `wfdb.rdheader` (Brugada-HUCA: 100 Hz). Không giả định.
3. Nếu chưa có dữ liệu: `python "0.tai du lieu.py" <thư_mục>` rồi đặt `BRUGADA_DIR`.

## Quy tắc cứng
| Đừng | Hãy |
|---|---|
| Viết cứng chỉ số mẫu (`beat[46]`) | `int(round(giây * fs))` |
| `find_peaks` trên riêng V1 | `ep.detect_qrs(sig12, fs)` (năng lượng 12 chuyển đạo) |
| Điểm J = R + hằng số | `ep.delineate(beat, fid, fs)` |
| Trung bình cộng các nhịp | `ep.median_beat` (căn chỉnh + loại nhịp lạc + trung vị) |
| Đường đẳng điện = đoạn cố định | `ep.baseline_correct(beat, onset, fs)` (đoạn PR) |
| `except: return NaN` | Trả về dict có `error` ghi lý do |
| Đặc trưng là hằng số với mọi ca | Kiểm tra `df[col].nunique() > 1` và `describe()` sau khi trích |

## Thêm một đặc trưng mới
1. Viết phép đo trong `features_from_beat` (đã có `onset`, `j_point`, nhịp đã trừ đường đẳng điện).
2. Thêm tên vào `FEATURE_COLUMNS` và mô tả + đơn vị vào `FEATURE_DESCRIPTIONS`.
3. Thêm test với ECG giả lập có đáp án vào `tests/test_ecg_pipeline.py`.
4. Chạy `python -m pytest tests/`, rồi bước 4 và bước 3 cho vài ca (các ca đột tử, ca giá trị cực trị) và **xem hình**.
5. Ghi nguồn y văn định nghĩa đặc trưng vào `docs/tong-quan-tai-lieu.md`.

## Định nghĩa đo chuẩn (theo y văn)
- **Brugada type 1 (coved):** J/ST chênh lên ≥ 2 mm (0.2 mV) ở ≥ 1 chuyển đạo V1–V2, ST cong xuống, T âm.
- **Type 2 (saddleback):** ST chênh lên ≥ 0.5 mm, hình yên ngựa, T dương ở V2.
- **Sóng S ở DI (Calò 2016):** thời gian ≥ 40 ms hoặc diện tích ≥ 1 mm².
- **aVR sign (Babai Bigi 2007):** R ≥ 0.3 mV hoặc R/q ≥ 0.75 ở aVR.
- **Góc β (Chevallier 2011):** > 58° giữa sườn lên S và sườn xuống r' ở V1/V2.
- **Đáy tam giác (Serra 2014):** đo ở 5 mm dưới điểm cất cánh, ≥ 4 mm (160 ms).
- Quy đổi: 1 mm = 0.1 mV (biên độ), 1 mm = 40 ms (ở 25 mm/s).

## Lọc tín hiệu
- 0.5–40 Hz Butterworth bậc 3, `filtfilt`. AHA 2007 cho phép chặn thấp ≤ 0.67 Hz **chỉ khi lọc pha bằng 0**. Không dùng `lfilter` (lệch pha làm méo ST).

## Kiểm tra chất lượng
- `n_beats` < 5 hoặc `beat_corr` < 0.9: xem hình, có thể nhiễu nặng.
- QRS < 60 ms hoặc > 160 ms: xem hình để kiểm tra điểm J.
- Giới hạn 100 Hz: không đo được fQRS; độ phân giải thời gian 10 ms.
