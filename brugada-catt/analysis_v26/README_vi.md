# Bước 26: CATT có cổng, và CATT tương đương bao nhiêu ECG có nhãn

Kế hoạch `PLAN_vi.md` được commit và đẩy lên lúc 10:48 UTC ngày 14/11/2026 (474f20c), trước mọi phép chạy. Không có sai lệch so
với kế hoạch. Phần Brugada đã chạy; **phần RBBB/LVH (G3, G4) chưa chạy**, chờ điểm số bước 14–15. Số liệu: `summary_v26_brugada.json`.

## Chạy lại

```
python analysis_v26/brugada26.py <analysis_v8/scores> analysis_v26     # ~5 phút
```

## 26a. CATT có cổng (Brugada)

Cổng: chỉ đổi sang ngưỡng CATT khi nó cách ngưỡng nguồn hơn z lần độ dao động ước lượng không nhãn của chính nó (Hệ quả 2 cho bản
thường, G2 sandwich cho bản làm trơn).

| | Giữ lại phần lợi ích ở 5 thiết lập hỏng | Cổng mở ở thiết lập hỏng (/15) | Cổng mở ở thiết lập không hỏng (/12) | BA held-out |
|---|---|---|---|---|
| CATT thường, không cổng | 100% | 15 | 12 | 0,799 |
| Thường, z = 1 | 100% | 15 | 7 | 0,798 |
| **Thường, z = 2 (chính)** | **100%** | 15 | **3** | 0,798 |
| Thường, z = 3 | 100% | 15 | 0 | 0,800 |
| Làm trơn, z = 2 (chính) | 100% | 15 | 8 | 0,795 |

**G1 đạt.** Ở 5 thiết lập hỏng, ngưỡng CATT cách ngưỡng nguồn 6,6–24 lần SD ước lượng (bản thường), nên cổng luôn mở. Ở 4 thiết lập
không hỏng, khoảng cách chỉ 1,2–1,7 SD: độ lệch ở đó cỡ nhiễu lấy mẫu, và cổng z = 2 đóng ở 9/12 lần.

- Kết quả này đúng như mong đợi về cơ chế: cổng tách được "ngưỡng hỏng thật" khỏi "dao động ngẫu nhiên" mà không cần nhãn.
- Bản làm trơn có SD ước lượng nhỏ hơn nhiều, nên cổng mở thường hơn ở thiết lập không hỏng (8/12). Với bản làm trơn có lẽ cần z
  lớn hơn; chưa kiểm, và không được chọn z sau khi xem kết quả.
- Trên Brugada CATT vốn không gây hại, nên phép thử thật của cổng là RBBB/LVH (G3), nơi CATT gây hại nhẹ. Dự đoán đã ghi trước: cổng
  chỉ chặn được độ lệch cỡ nhiễu, không chặn được độ lệch do mỏ neo phụ thuộc.

## 26b. CATT tương đương bao nhiêu ECG có nhãn (mô tả)

Ngưỡng Youden đặt trên k ECG đích có nhãn thật (rút ngẫu nhiên, 300 lần), so với CATT dùng toàn bộ cohort đích không nhãn.
Trung bình 7 thiết lập held-out.

**Tỉ lệ bệnh tự nhiên (20–52%)**, CATT làm trơn 0,796; CATT thường 0,799:

| k ECG có nhãn | 10 | 20 | 30 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|
| Số ca dương kỳ vọng | 3 | 7 | 10 | 17 | 34 | 68 |
| BA, Youden thường | 0,743 | 0,776 | 0,789 | **0,799** | 0,808 | 0,814 |
| BA, Youden làm trơn | 0,762 | 0,790 | **0,798** | 0,803 | 0,807 | 0,809 |
| Phân vị 10%, làm trơn | 0,660 | 0,757 | 0,779 | 0,789 | 0,796 | 0,802 |

**Tỉ lệ bệnh 5%** (giữ mọi ca âm, rút ca dương), CATT làm trơn 0,797 (phân vị 10%: 0,741):

| k ECG có nhãn | 10 | 20 | 30 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|
| Số ca dương kỳ vọng | 0,5 | 1 | 1,5 | 2,5 | 5 | 10 |
| BA, Youden thường | 0,705 | 0,718 | 0,734 | 0,761 | **0,800** | 0,828 |
| BA, Youden làm trơn | 0,707 | 0,724 | 0,741 | 0,772 | **0,802** | 0,814 |
| Phân vị 10%, làm trơn | 0,595 | 0,586 | 0,589 | 0,646 | 0,733 | 0,753 |

## Đọc kết quả

1. **Ở tỉ lệ bệnh tự nhiên, CATT tương đương khoảng 30–50 ECG có nhãn. Ở tỉ lệ bệnh 5%, khoảng 100 ECG có nhãn** (tức khoảng 5 ca
   Brugada). Với bệnh hiếm, lợi thế của cách không nhãn rõ hơn, vì phải đọc rất nhiều ECG mới gặp đủ ca dương.
2. Cũng phải nói thẳng: nếu viện mới gán nhãn được hơn 50–100 ECG thì ngưỡng có nhãn tốt hơn CATT. CATT là lựa chọn khi chưa có
   nhãn hoặc nhãn đắt, không phải thay thế cho nhãn.
3. Cổng z = 2 giữ nguyên lợi ích trên Brugada và gần như không động vào chỗ đã ổn. Đây là ứng viên tốt cho câu "khi nào không nên
   đặt lại ngưỡng", nhưng cần kết quả RBBB/LVH mới kết luận được.

## Hạn chế

- Một bộ điểm số (CNN-V1V3, mỏ neo CBM5), 9 thiết lập Brugada; pipeline tiền xử lý cũ.
- Cohort 5% được tạo bằng cách rút bớt ca dương, không phải quần thể sàng lọc thật.
- Ngưỡng có nhãn dùng k ECG rút ngẫu nhiên; ngoài đời có thể chọn có chủ đích (ví dụ làm giàu ca dương), sẽ cần ít ECG hơn.
