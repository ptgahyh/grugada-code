# Bước 22: LVH với mô hình đích nhận nhịp mV. Kế hoạch (viết và commit TRƯỚC khi chạy)

Ngày viết: 11/11/2026. **Chưa chạy.** Bước này cần dữ liệu PTB-XL và Chapman/Ningbo đã chuẩn hóa của bước 15 (`ptb_cs.npz`,
`ptb_at.npz`, `cn.npz`, có cả `X` z-score và `M` mV), không có ở máy cloud. Người chạy: Phương, trên máy có dữ liệu.

## Lý do

Ở bước 15, f nhận nhịp z-score từng chuyển đạo nên không thấy điện thế tuyệt đối, trong khi LVH được định nghĩa chủ yếu bằng
điện thế. Hệ quả: mỏ neo LVH-LR xếp hạng tốt hơn f ở mọi thiết lập (AUROC 0,90–0,98 so với 0,83–0,94), và không thiết lập nào
"hỏng" nên giả thuyết chính không kiểm được. Câu hỏi: nếu f nhận nhịp mV thì (1) f có tốt lên không, (2) ngưỡng nguồn có hỏng
khi đổi máy/đổi viện không, (3) CATT có sửa được không.

## Thay đổi duy nhất so với bước 15

`analysis_v22/run_lvh_mv.py` = `analysis_v15/run_lvh.py` với **một dòng** đổi: đầu vào của f và CNNalt là `M` (mV) thay cho `X`.
Kiến trúc, tăng cường (gồm nhiễu Gauss σ = 0,05, nay tính bằng mV), giới hạn 12 000 bản ghi, cách chia, mỏ neo A1–A3, hạt giống:
giữ nguyên. Phân tích dùng lại `analysis_v15/analyze_lvh.py` (các giai đoạn p1, p3, p4) trên thư mục điểm số mới.

## Tiêu chí (định trước)

| Mã | Nội dung | Đạt khi |
|---|---|---|
| M1 | f nhận mV xếp hạng tốt hơn f z-score | AUROC đích tăng ≥ 0,02 ở ít nhất 4/6 thiết lập (so với bảng bước 15) |
| M2 | Có thiết lập hỏng không (BA ở ngưỡng nguồn thấp hơn oracle quá 0,05, trung bình 3 seed) | mô tả: số thiết lập hỏng trong 6 |
| M3 = H1 | CATT(A3) − SRC ở các thiết lập hỏng, gộp, bootstrap bệnh nhân như bước 15 | hiệu > 0 và KTC 95% không chứa 0; nếu M2 = 0 thì ghi "không kiểm được" |
| M4 | Định lý 1 (P1) và ước lượng không nhãn trên điểm số mới | trung vị dự đoán/quan sát ∈ [0,80; 1,25], Spearman ≥ 0,7 (như bước 15) |
| M5 | CATT có hại ở thiết lập không hỏng không | mô tả: CATT(A3) − SRC gộp trên các thiết lập không hỏng, kèm KTC |

Dự đoán trước khi chạy (để đối chiếu, không phải tiêu chí): f và các mỏ neo điện thế nay dùng chung thông tin biên độ, nên phụ
thuộc giữa chúng tăng; theo Mệnh đề 2, CATT với mỏ neo điện thế có thể lệch nhiều hơn bước 15 ở đích có tỉ lệ bệnh thấp (PTB→cn).

Mọi kết quả đều báo cáo, kể cả khi M1 không đạt hoặc CATT kém hơn ngưỡng nguồn. Kết quả bước 15 giữ nguyên trong bài; bước 22
được trình bày là phân tích bổ sung có kế hoạch trước.

## Lệnh chạy

```
git pull
python analysis_v22/run_lvh_mv.py <thư mục npz bước 15> <thư mục điểm mV> 0 1
python analysis_v15/analyze_lvh.py <thư mục điểm mV> analysis_v22/results p1
python analysis_v15/analyze_lvh.py <thư mục điểm mV> analysis_v22/results p3
python analysis_v15/analyze_lvh.py <thư mục điểm mV> analysis_v22/results p4
```

Nếu chưa có ba tệp npz: chạy lại `analysis_v15/harmonize_lvh.py` (cần tải PTB-XL và bộ Chapman/Ningbo từ PhysioNet).
