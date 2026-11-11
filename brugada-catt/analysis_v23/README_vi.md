# Bước 23: dải thông của nhịp đại diện giải thích khoảng 1/5 độ trôi ngưỡng

Kế hoạch `PLAN_vi.md` được commit và đẩy lên lúc 08:57 UTC ngày 11/11/2026 (1d78b57), trước khi tính phổ hay huấn luyện.
Không có sai lệch so với kế hoạch. Số liệu: `summary_v23.json`, `spectra_v23.json`. Huấn luyện lại tái lập trùng tuyệt đối bước 8.

## Chạy lại

```
python analysis_v23/spectra_v23.py <harmonized> analysis_v23
python analysis_v23/train_v23.py <harmonized> analysis_v23/spectra_v23.json <thư mục riêng> [chunk số_chunk]   # ~15 phút trên 2 nhân
python analysis_v23/analyze_v23.py <thư mục riêng> <analysis_v8/scores> analysis_v23
python analysis_v23/make_fig_v23.py analysis_v23
```

## 23a. Phổ (mô tả, không dùng nhãn) — hình `fig_v23_spectra.png`

| Bộ | Tỉ lệ năng lượng ≥ 15 Hz, V1–V3: trung vị (IQR) | 9 chuyển đạo | Lớp âm / lớp dương |
|---|---|---|---|
| HUCA (trung vị các nhịp đã lọc 0,5–40 Hz) | **0,147** (0,104–0,214) | 0,192 | 0,166 / 0,094 |
| Claris (nhịp đại diện của máy) | 0,204 (0,132–0,289) | 0,195 | 0,253 / 0,170 |
| Mortara (nhịp đại diện của máy) | 0,230 (0,161–0,312) | 0,220 | 0,255 / 0,202 |

- HUCA trơn nhất và tụt mạnh từ khoảng 40 Hz: dấu vết của bộ lọc 0,5–40 Hz chỉ áp cho HUCA trong `harmonize.py`.
- Làm trơn Gauss **σ = 6 ms** cho Claris và Mortara thì phổ khớp HUCA (HF 0,141 và 0,159).
- Ở cả ba bộ, ca Brugada có ít năng lượng cao tần hơn ca chứng: độ trơn liên quan đến nhãn, nên nó có thể ảnh hưởng điểm số.

## Kết quả theo tiêu chí

| Mã | Nội dung | Kết quả | Kết luận định trước |
|---|---|---|---|
| A1 | Làm trơn nhịp Ý σ = 6 ms lúc chấm điểm (mô hình giữ nguyên): lấp bao nhiêu khoảng cách tới oracle | **20,5%** (từng thiết lập 15–32%) | ≥ 20%: **dải thông đóng góp đáng kể** (vừa chạm ngưỡng) |
| A2 | \|Δ\| lớp âm | 2,28 → 1,73 logit | mô tả |
| B1 | Huấn luyện lại sau khi đồng bộ dải thông: còn bao nhiêu thiết lập hỏng | **5/5**; BA ở ngưỡng nguồn 0,567 → **0,620** (+0,053) | **một phần** |
| B2 | AUROC đổi viện; BA oracle | 0,865 → 0,869; 0,796 → 0,805 | không mất khả năng phân biệt |
| B3 | \|Δ\| lớp âm | 2,28 → 1,77 logit | mô tả |
| 23d | Làm trơn cả ba bộ σ = 20 ms rồi huấn luyện lại | 4/5 hỏng; BA nguồn 0,600; **AUROC 0,817, BA oracle 0,748** | đổi lấy mất khả năng phân biệt |

**A3, liều–đáp ứng (lúc chấm điểm, nhân σ = 6 ms):**

| Hệ số | 0 | 0,5 | 1 (khớp phổ) | 1,5 | 2 |
|---|---|---|---|---|---|
| Khoảng cách được lấp | 0% | 1% | 21% | 40% | 57% |
| \|Δ\| lớp âm | 2,28 | 2,27 | 1,73 | 1,15 | 0,72 |
| AUROC (9 thiết lập) | 0,865 | 0,865 | 0,861 | 0,848 | 0,827 |

Hệ số 1,5 và 2 làm trơn **quá** mức khớp phổ; chúng cho thấy hướng tác động, không phải phép đồng bộ dải thông.

## Đọc kết quả

1. **Dải thông là một nguyên nhân thật nhưng không phải nguyên nhân chính.** Đồng bộ dải thông lấp khoảng 20–23% khoảng cách
   (cả khi chấm điểm lại và khi huấn luyện lại), không mất AUROC. Cả 5 thiết lập vẫn hỏng.
2. Con số này cùng cỡ với kết quả bước 8 (mô phỏng chuỗi thu nhận giải thích khoảng 1/3 độ dịch). Cùng với bước 19 (độ dịch nằm
   trong cùng dạng sóng) và bước 21 (lệch căn không đáng kể), bức tranh là: **chuẩn hóa tiền xử lý sửa được một phần nhỏ; phần
   lớn độ trôi còn lại cần đặt lại ngưỡng (CATT)**.
3. Hệ quả cho pipeline: nên áp **cùng một bộ lọc cho mọi nguồn**. Hiện chỉ HUCA được lọc 0,5–40 Hz. Đây là một cải thiện tiền xử lý
   hợp lý cho bài, nhưng đổi pipeline thì mọi con số chính phải chạy lại.
4. Làm trơn mạnh hơn nữa giảm độ dịch thêm nhưng mất khả năng phân biệt (AUROC 0,817 ở σ = 20 ms): không phải lối thoát.

## Hạn chế

- Bộ lọc là Gauss trên nhịp đã z-score ở 100 Hz, không phải tái tạo đúng bộ lọc của máy. σ khớp phổ chọn trên lưới bước 2 ms.
- Chỉ đồng bộ theo hướng làm trơn bộ sắc hơn; không khử trơn HUCA.
- Chưa kiểm CATT trên điểm số sau đồng bộ (cần huấn luyện lại mỏ neo); chưa có kế hoạch cho phần này.
