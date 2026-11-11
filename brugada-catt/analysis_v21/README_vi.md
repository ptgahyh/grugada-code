# Bước 21: lệch căn thời gian có gây trôi ngưỡng không, và f nhạy với thay đổi dạng sóng nào

Kế hoạch `PLAN_vi.md` được commit và đẩy lên lúc 04:11 UTC ngày 11/11/2026 (74cb4ab), trước khi huấn luyện lại hay tính kết quả.
Không có sai lệch so với kế hoạch. Số liệu: `summary_v21.json`. Chạy trên CPU, PyTorch 2.15.

Kiểm tra trước: huấn luyện lại CNN-V1V3 cho điểm số **trùng tuyệt đối** với bước 8 (sai khác lớn nhất 0,0 ở cả 27 lần chạy).

## Chạy lại

```
python analysis_v21/train_v21.py <harmonized> <thư mục riêng>              # ~8 phút trên 1 nhân (chia 2 chunk: thêm "0 2", "1 2")
python analysis_v21/response_v21.py <harmonized> <thư mục riêng> analysis_v21
python analysis_v21/analyze_v21.py <thư mục riêng> <analysis_v8/scores> analysis_v21
python analysis_v21/make_fig_v21.py analysis_v21
```

## Kết quả theo tiêu chí

| Mã | Nội dung | Kết quả | Kết luận định trước |
|---|---|---|---|
| A1 | Dịch nhịp đích cho khớp căn (±20 ms): lấp bao nhiêu khoảng cách tới oracle | **1%** (từng thiết lập: −2% đến +3%) | ≤ 5%: **không đáng kể** |
| A2 | \|Δ\| lớp âm trước và sau khi dịch | 2,28 → 2,23 logit | mô tả |
| B1 | Huấn luyện lại với cách căn chung (đáy S của V1): còn bao nhiêu thiết lập hỏng trong 5 | **5/5**; BA ở ngưỡng nguồn 0,567 → 0,565 | **không tác dụng** |
| B2 | AUROC đổi viện, BA oracle | 0,865 → 0,864; 0,796 → 0,792 | mô tả: không mất khả năng phân biệt |
| B3 | \|Δ\| lớp âm ở 5 thiết lập hỏng | 2,28 → 2,44 logit | mô tả: không giảm |
| R1 | f tăng đơn điệu theo ST dạng vòm | Spearman = 1,00 ở cả 9 cặp mô hình × nhịp nền | **Đạt** |

**Kết luận chính: lệch căn 20 ms giữa HUCA và Melo (quan sát ở bước 19) KHÔNG phải nguyên nhân trôi ngưỡng.** Dịch nhịp trong
khoảng ±30 ms gần như không đổi balanced accuracy ở ngưỡng nguồn (thay đổi ≤ 0,03) hay AUROC (0,865–0,870). Mạng dùng gộp trung
bình toàn cục và được tăng cường dịch ±20 ms khi huấn luyện nên gần như bất biến với dịch thời gian của cả ba chuyển đạo.

## 21c: đường đáp ứng (mô tả) — hình `fig_v21_response.png`

Thay đổi logit trung bình của f ở mức biến đổi lớn nhất, trung bình trên ba bộ nhịp âm:

| Biến đổi (V1–V2) | Mô hình học từ HUCA | từ Claris | từ Mortara |
|---|---|---|---|
| P1 ST dạng vòm (+2 đơn vị z) | +1,6 | **+3,7** | **+3,0** |
| P5 lọc thông thấp (σ = 20 ms) | +1,1 | **+2,0** | +1,3 |
| P2 sóng T âm (2 đơn vị z) | +0,6 | +0,4 | +0,9 |
| P3 r′/điểm J nhô (+2) | 0,0 | +0,3 | +1,0 |
| P4 QRS rộng (×1,5) | +0,5 | +0,4 | +0,2 |
| P6 dịch riêng V1–V2 (+30 ms) | −0,2 | −0,4 | −0,9 |

- **R2.** Ba mô hình đáp ứng giống nhau về hướng: tương quan giữa các vectơ đáp ứng 0,78–0,94 (Claris với Mortara 0,88–0,94; HUCA với
  hai mô hình Ý 0,78–0,93). Mô hình học từ HUCA đáp ứng yếu hơn khoảng một nửa với ST dạng vòm.
- **R3.** Chỉ hai biến đổi tạo được độ dịch cỡ độ dịch quan sát khi đổi viện (1,3–3,4 logit): ST dạng vòm, và **làm trơn tín hiệu**
  (σ = 10–20 ms cho +1,0 đến +2,3 logit ở mô hình Ý). Dịch riêng V1–V2 so với V3 tới 30 ms đổi điểm tối đa 1,3 logit.
- Đáp ứng với cùng một biến đổi phụ thuộc bộ nhịp nền: mô hình Ý tăng +4,5 logit trên nhịp Claris nhưng chỉ +2,2 trên nhịp HUCA
  cho cùng mức ST dạng vòm.

## Đọc kết quả

1. Một giả thuyết cụ thể về nguyên nhân (lệch căn) đã bị loại bằng hai phép thử độc lập (dịch lúc chấm điểm; huấn luyện lại).
   Câu trong bài "ngưỡng hỏng không sửa được bằng chuẩn hóa đơn giản" có thêm bằng chứng.
2. Ứng viên còn lại đáng kiểm: **độ trơn/dải thông của nhịp đại diện**. Mô hình rất nhạy với làm trơn, và nhịp Melo là nhịp đại
   diện do máy tính toán (200 Hz, không qua bộ lọc 0,5–40 Hz của pipeline), còn nhịp HUCA là trung vị các nhịp đã lọc. Đây mới là
   giả thuyết; bước 8 (`device_sim.py`) đã cho thấy mô phỏng chuỗi thu nhận giải thích khoảng 1/3 độ dịch.
3. f học đúng dấu hiệu lâm sàng (ST dạng vòm làm tăng điểm ở cả ba mô hình, trên cả ba bộ nhịp).

## Hạn chế

- Phantom là phép cộng/biến đổi toán học đơn giản trên nhịp đã z-score, không phải ECG bệnh lý thật; chỉ kết luận được "mô hình
  nhạy với thay đổi nào", không kết luận được nguyên nhân của độ dịch thật.
- 21c dùng một mô hình mỗi nguồn (seed validation 0, tổ hợp 3 mạng).
- 21b chỉ thử một cách căn chung (đáy S của V1).
