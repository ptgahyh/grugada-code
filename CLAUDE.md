# CLAUDE.md: bối cảnh dự án Grugada

## Người dùng và mục tiêu
- Người dùng nói tiếng Việt; trả lời, viết comment và tài liệu bằng **tiếng Việt**. Thuật ngữ y khoa/thống kê giữ tiếng Anh trong ngoặc khi cần (ví dụ: điểm J (J-point)).
- Đề tài: đặc trưng ECG liên quan đến **chẩn đoán hội chứng Brugada** và **đột tử (SCD)** trên bộ dữ liệu công khai Brugada-HUCA. Mục tiêu là báo cáo/bài báo khoa học, nên **tính đúng về phương pháp quan trọng hơn kết quả "đẹp"**.

## Dữ liệu (đã kiểm tra trực tiếp)
- 363 ECG, 12 chuyển đạo, **100 Hz, 1200 mẫu (12 s)**, đơn vị mV. Thứ tự: I, II, III, aVR, aVL, aVF, V1–V6 (V1 = cột 6).
- `metadata.csv`: `patient_id`, `basal_pattern` (0/1), `sudden_death` (0/1, **chỉ 11 ca = 1**), `brugada` (0/1/2; giá trị 2 có 7 ca, ý nghĩa chưa rõ → coi `brugada >= 1` là Brugada, tổng 76 ca khớp PhysioNet).
- Trong phiên Claude trên cloud, physionet.org và zenodo.org bị chặn; tải từ bản sao GitHub bằng `python "0.tai du lieu.py" <thư_mục>` (raw.githubusercontent.com truy cập được). Đặt `BRUGADA_DIR=<thư_mục>`. Không commit dữ liệu thô vào repo.

## Cấu trúc code
- `ecg_pipeline.py`: **mọi thuật toán nằm ở đây** (lọc, dò QRS, nhịp trung vị, onset/điểm J, đặc trưng). Các script đánh số chỉ gọi module này. Sửa thuật toán thì sửa ở đây, không chép hàm sang script khác.
- `0`→`5` là các bước chạy tuần tự; tên file có dấu cách nên phải đặt trong ngoặc kép khi chạy.
- `Final_Research_Data.csv` (đầu ra bước 4), `results/stats_*.csv` và `figures/` (đầu ra bước 3, 5) được commit để người dùng xem kết quả không cần chạy lại.
- Kiểm thử: `python -m pytest tests/` (ECG giả lập có đáp án). Chạy sau mọi thay đổi trong `ecg_pipeline.py`.

## Quy tắc phương pháp (đừng tái phạm lỗi cũ)
1. **Không viết cứng chỉ số mẫu**; mọi mốc thời gian = giây × `fs`.
2. **Không dò đỉnh R trên riêng V1** (QRS ở V1 thường âm). Dùng `ep.detect_qrs` trên cả 12 chuyển đạo.
3. Điểm J lấy từ `ep.delineate` (một điểm J chung cho mọi chuyển đạo), không dùng "R + hằng số".
4. Không dùng `except:` trống; ghi lỗi vào cột `error`.
5. Sau khi đổi thuật toán: chạy bước 3 cho các ca đột tử và vài ca lỗi, **xem hình** trước khi tin con số.
6. Thống kê: báo cáo hiệu ứng (AUC/median [IQR]) + KTC 95% + q-value BH, không chỉ p-value. Với 11 ca đột tử: **không** xây mô hình đa biến/học máy để dự báo đột tử; ghi rõ "thăm dò".
7. Ưu tiên kiểm định giả thuyết có sẵn từ y văn (S ở DI ≥ 40 ms, aVR R ≥ 0.3 mV, QRS rộng) hơn là dò tìm hàng loạt đặc trưng.

## Giới hạn đã biết
- 100 Hz → độ phân giải 10 ms; không đo được QRS phân mảnh (fQRS).
- Không có chuyển đạo V1–V2 ở khoang liên sườn cao.
- Điểm J tự động chưa được đối chiếu với đo tay của bác sĩ.

## Tài liệu
- `README.md`: cách chạy, các thay đổi so với bản đầu, kết quả chính.
- `docs/tong-quan-tai-lieu.md`: tổng quan y văn kèm nguồn. Cập nhật khi tra thêm tài liệu; **không bịa trích dẫn**, chỉ ghi nguồn đã mở được.
- Skill riêng của dự án trong `.claude/skills/`: `ecg-brugada-pipeline`, `small-sample-stats`, `lit-review`, `research-writing-vn`.

## Git
- Commit message tiếng Việt hoặc tiếng Anh ngắn gọn, mô tả "vì sao".
