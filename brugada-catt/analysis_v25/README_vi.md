# Bước 25: ngưỡng làm trơn vẫn đứng vững ngoài mô hình Gauss và với 5 mỏ neo

Kế hoạch `PLAN_vi.md` được commit và đẩy lên lúc 09:27 UTC ngày 11/11/2026 (9e5f46a), trước mọi phép chạy. Không có sai lệch so
với kế hoạch. Số liệu: `summary_v25.json`.

## Chạy lại

```
python analysis_v25/nongauss25.py analysis_v25                            # ~10 phút
python analysis_v25/brugada25.py <analysis_v8/scores> analysis_v25        # ~5 phút
python analysis_v25/analyze25.py analysis_v25
```

## 25a. Ngoài mô hình Gauss (4 kịch bản × 16 ô)

| Kịch bản | N1: RMSE làm trơn / thường | N2: tổn thất BA làm trơn / thường | N3: công thức SD / quan sát | Số ECG cần: thường / làm trơn (ε = 0,15; 0,30) |
|---|---|---|---|---|
| Lệch (Gamma) | 0,70 (0,56–1,00) | 0,44 | 0,95 (0,85–1,01) | 6,8; 4,8 lần |
| Đuôi dày (t₃) | 0,65 (0,49–1,18) | 0,53 | 0,98 (0,69–1,02) | 6,1; 4,3 lần |
| Hai đỉnh | 0,73 (0,44–1,12) | 0,77 | 0,96 (0,82–1,13) | 5,5; 3,9 lần |
| Rời rạc (bước 0,25) | 0,57 (0,30–0,97) | 0,69 | 1,00 (0,94–1,07) | 4,6; 3,2 lần |
| **Tiêu chí** | **Đạt** (gộp 0,67) | **Đạt** | **Đạt** | mô tả |

- Ngưỡng làm trơn tốt hơn ngưỡng thường ở cả bốn kịch bản, cả về sai số của ngưỡng lẫn balanced accuracy.
- **Điểm số rời rạc:** lý thuyết căn bậc ba của ngưỡng thường không áp dụng được (bước 13 thấy nó sai ở đây), còn công thức của
  ngưỡng làm trơn vẫn khớp (0,94–1,07). Đây là một ưu điểm thêm của bản làm trơn.
- Có vài ô ngưỡng làm trơn kém hơn (RMSE tới 1,18 ở đuôi dày, 1,12 ở hai đỉnh), và một ô đuôi dày công thức dự đoán thấp (0,69).
- N4 (mô tả): ước lượng không nhãn từ một mẫu có sai số 0,20–0,25 ở mọi kịch bản, giống mô hình Gauss (bước 24, T2).

## 25b. Brugada với 5 mỏ neo

| Mỏ neo | SD ngưỡng làm trơn / thường | BA với 50 ECG: thường → làm trơn | BA toàn cohort: thường → làm trơn | G2 từng mẫu: tỉ số, \|log\| | Số ECG cần ε = 0,30: thường → làm trơn |
|---|---|---|---|---|---|
| CBM5 (mỏ neo chính) | 0,52 | 0,783 → 0,789 | 0,799 → 0,796 | 0,93; 0,15 | 431 → 71 |
| CBM3 | 0,50 | 0,777 → 0,785 | 0,793 → 0,789 | 0,93; 0,16 | 332 → 52 |
| ST_V2 | 0,57 | 0,776 → 0,787 | 0,789 → 0,790 | 0,93; 0,16 | 462 → 90 |
| DANN (mỏ neo phụ thuộc) | 0,63 | 0,717 → 0,743 | 0,720 → 0,739 | 0,94; 0,10 | 66 → 8 |
| CORAL (mỏ neo phụ thuộc) | 0,57 | 0,709 → 0,737 | 0,714 → 0,733 | 0,93; 0,12 | 77 → 10 |

| Mã | Nội dung | Kết quả | Đạt? |
|---|---|---|---|
| M1 | SD làm trơn / thường | gộp 0,56; mọi mỏ neo 0,50–0,63 | **Đạt** |
| M2 | BA với 50 / 100 ECG, làm trơn − thường (7 thiết lập giữ lại) | +0,016 / +0,013 | **Đạt** |
| M3 | BA toàn cohort, làm trơn − thường | +0,006 | **Đạt** |
| M4 | G2 từng mẫu đơn | tỉ số 0,93; \|log\| 0,135 | **Đạt** |

Không lần chạy nào bị bỏ vì lớp nhãn giả quá nhỏ.

## Đọc kết quả

1. Kết luận của bước 20 và 24 không phụ thuộc vào giả định Gauss hay vào riêng mỏ neo CBM5: ngưỡng làm trơn **giảm độ dao động
   khoảng một nửa với mọi mỏ neo**, không làm giảm balanced accuracy (với ít ECG còn tăng 0,006–0,028), và cần ít ECG hơn 4–8 lần.
2. Với hai mỏ neo phụ thuộc (DANN, CORAL), CATT vẫn kém hơn mỏ neo khái niệm (BA khoảng 0,73 so với 0,79), như Mệnh đề 1–2 dự đoán.
   Số ECG cần rất nhỏ ở hai mỏ neo này chỉ có nghĩa ngưỡng ổn định, không có nghĩa ngưỡng đúng.
3. Trên Brugada, ước lượng sandwich G2 từ một mẫu đạt sai số 0,10–0,16 ở mọi mỏ neo, tốt hơn F1 cho ngưỡng thường (0,23, bước 18c).
   Trong mô phỏng thì vẫn 0,20–0,25; khác biệt có thể do cỡ mẫu và hình dạng điểm số Brugada.

## Hạn chế

- Bốn kịch bản mô phỏng vẫn dùng copula Gauss giữa f và mỏ neo, π = 0,2.
- 5 mỏ neo cùng một bộ điểm số bước 8 (CNN-V1V3). RBBB, LVH chưa kiểm: máy cloud không tải được dữ liệu từ PhysioNet.
