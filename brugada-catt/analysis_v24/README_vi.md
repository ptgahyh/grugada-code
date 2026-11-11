# Bước 24: công thức độ dao động và cỡ mẫu cho ngưỡng CATT làm trơn

Kế hoạch `PLAN_vi.md` được commit và đẩy lên lúc 08:57 UTC ngày 11/11/2026 (1d78b57), trước mọi phép chạy. Không có sai lệch so
với kế hoạch. Số liệu: `summary_v24.json`.

## Kết quả lý thuyết (khai triển delta chuẩn của ước lượng kernel; không phải định lý mới)

Ngưỡng làm trơn t̂_h là nghiệm của p̂₁,h(t) = p̂₀,h(t) (hai mật độ kernel của f trong hai lớp nhãn giả cắt nhau). Khi đó

  **SD(t̂_h) ≈ √(R·Ṽ / (n·h)) / Ã,  R = 1/(2√π) ≈ 0,282,**

với **cùng hai hằng số Ṽ, Ã của Định lý 1**. Với độ rộng h = c·(n·m)^(−1/5) (c = 0,9·min(SD, IQR/1,34) của f, m = min(q, 1 − q)):

  **muốn SD ≤ ε cần n ≥ [R·Ṽ·m^(1/5) / (c·Ã²·ε²)]^(5/4)**, so với n ≥ 0,54·Ṽ/(Ã²·ε³) cho ngưỡng thường.

Dạng chính xác (h cố định) dùng hàm ảnh hưởng ψ, xem `PLAN_vi.md`.

## Chạy lại

```
python analysis_v24/sim24.py analysis_v24          # T1, T2
python analysis_v24/fpc24.py analysis_v24          # T3
python analysis_v24/plan24.py analysis_v24         # T4
python analysis_v24/brugada24.py <analysis_v8/scores> analysis_v24   # T5, T6
python analysis_v24/analyze24.py analysis_v24
```

## Kết quả theo tiêu chí

| Mã | Nội dung | Kết quả | Đạt? |
|---|---|---|---|
| T1 | Công thức (dạng chính xác, đại lượng quần thể) so với SD mô phỏng | trung vị **0,98**; theo n = 100/300/1000/3000: 0,93 / 0,98 / 0,99 / 1,00 | **Đạt** |
| T1 phụ | Dạng đóng √(RṼ/(nh))/Ã | trung vị 1,06 (0,98–1,07 theo n) | mô tả |
| T1 phụ | Độ dốc log SD theo log n | −0,42 (lý thuyết −0,40) | mô tả |
| T2 | Ước lượng không nhãn từ một mẫu (mô phỏng): tốt nhất G1–G3 | G1 (dạng đóng + KDE): \|log\| **0,204** (0,28 → 0,15 khi n tăng); G2 sandwich 0,236; bootstrap 0,210 | **Không đạt** (cần ≤ 0,15) |
| T3 | Số mũ hệ số quần thể hữu hạn | γ = **0,445** (KTC 95% 0,41–0,48) | **Đạt** (gần 1/2 hơn nhiều so với 0,24 của ngưỡng thường; KTC chưa chứa 0,5) |
| T4 | Quy tắc cỡ mẫu từ mẫu thử 300 ECG: tỉ lệ đạt SD ≤ 1,2ε | ε = 0,15: 74%; ε = 0,30: 77% (mẫu thử 100: 69%, 74%) | **Không đạt** (cần 80%) |
| T5 | Brugada, từng mẫu đơn | G2 sandwich: tỉ số 0,92 (IQR 0,79–1,05), \|log\| **0,158**; G1: 1,09, 0,188 | **Đạt** |

Độ lệch (bậc h², không nằm trong SD): trong mô phỏng từ −0,03 đến +0,24 lần SD.

**T4, hệ số an toàn (mô tả):** nhân n_plan với 1,25 thì tỉ lệ đạt 83% (mẫu thử 300, cả hai ε); với 1,5 thì 81–86% cả khi mẫu thử
chỉ 100 ECG.

## T6 và đối chiếu: cần bao nhiêu ECG không nhãn

| | ε = 0,15 logit | ε = 0,30 logit |
|---|---|---|
| Brugada (trung vị 27 lần chạy, ước lượng không nhãn trên toàn cohort): ngưỡng thường | 3449 | 431 |
| Brugada: ngưỡng làm trơn | **399** | **70** |
| Brugada: tỉ số (trung vị) | **7,4 lần** | **5,2 lần** |
| Mô phỏng (hằng số quần thể, 4 cấu hình): tỉ số | 2,9–5,8 lần | 2,0–4,1 lần |

## Đọc kết quả

1. **Công thức đúng.** SD dự đoán khớp mô phỏng (0,93–1,09 ở mọi ô) và dạng đóng chỉ cao hơn 6%. Nó dùng lại đúng Ṽ, Ã của Định lý 1,
   nên bài có thể trình bày hai ngưỡng (thường, làm trơn) trong cùng một khung.
2. **Lợi ích thực tế lớn:** cùng độ chính xác, ngưỡng làm trơn cần ít ECG không nhãn hơn 2–7 lần; ở Brugada với ε = 0,30, 70 ECG
   thay cho 431. Đổi lại có độ lệch nhỏ (≤ 0,24 SD trong mô phỏng); bước 20 cho thấy balanced accuracy không giảm.
3. **Hệ số quần thể hữu hạn gần với lý thuyết cổ điển** (0,45 so với 1/2), khác hẳn ngưỡng thường (0,24), đúng như dự đoán vì ước lượng
   làm trơn tuyến tính tiệm cận.
4. **Điểm yếu còn lại:** ước lượng không nhãn từ một mẫu vẫn dao động khoảng ±20% ở n ≤ 1000 (nhưng giảm theo n, khác F1), và quy tắc
   cỡ mẫu một giai đoạn cần hệ số an toàn khoảng 1,25–1,5 để đạt 80%.

## Hạn chế

- Mô phỏng chỉ có mô hình Gauss. Brugada chỉ một mỏ neo. RBBB, LVH chưa kiểm (cần điểm số bước 14–15).
- SD không gồm độ lệch; quy tắc cỡ mẫu chỉ kiểm soát độ dao động.
- Hệ số an toàn được chọn sau khi xem kết quả (T4 mô tả); muốn dùng phải kiểm lại trên dữ liệu khác với kế hoạch trước.
