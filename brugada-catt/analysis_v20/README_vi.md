# Bước 20: ngưỡng Youden làm trơn

Kế hoạch `PLAN_vi.md` được commit và đẩy lên lúc 04:11 UTC ngày 11/11/2026 (74cb4ab), trước mọi phép chạy. Không có sai lệch
so với kế hoạch. Số liệu: `summary_v20.json`.

Ngưỡng làm trơn: thay hàm bậc thang trong tiêu chí Youden theo nhãn giả bằng hàm Φ với độ rộng
h = 0,9·min(SD, IQR/1,34)·n_min^(−1/5). Ngưỡng là điểm hai mật độ kernel của hai lớp nhãn giả cắt nhau.

## Chạy lại

```
python analysis_v20/smooth_sim.py analysis_v20                              # ~3 phút
python analysis_v20/smooth_brugada.py <analysis_v8/scores> analysis_v20     # ~2 phút
```

## Kết quả theo tiêu chí

| Mã | Nội dung | Kết quả | Đạt? |
|---|---|---|---|
| S1a | Mô phỏng: RMSE của ngưỡng, làm trơn / thường | trung vị 16 ô **0,65** (0,47–0,93); không ô nào > 1 | **Đạt** |
| S1b | Mô phỏng: tổn thất balanced accuracy, làm trơn / thường | trung vị **0,67** (lớn nhất 0,97) | **Đạt** |
| S1c | Độ dốc log SD theo log n | thường −0,31; làm trơn **−0,41** (lý thuyết −0,33 và −0,40) | mô tả |
| S1d | Bootstrap thường ước lượng SD của ngưỡng làm trơn | trung vị 0,95 (theo n: 0,91–0,97, đạt); nhưng \|log\| gộp **0,221** > 0,15 (theo n: 0,31 → 0,16) | **Không đạt** |
| S2a | Brugada: SD của ngưỡng, làm trơn / thường | trung vị 81 điểm **0,54** (IQR 0,40–0,65) | **Đạt** |
| S2b | Brugada: BA với 50 / 100 ECG không nhãn (7 thiết lập giữ lại) | n = 50: 0,7895 so với 0,7824; n = 100: 0,7921 so với 0,7900 | **Đạt** |
| S2c | Brugada: BA với toàn bộ cohort đích | 0,7956 so với 0,7993 (−0,004) | **Đạt** |
| S2d | Phân vị 10% của BA | n = 50: **0,770** so với 0,741; n = 100: 0,779 so với 0,764 | mô tả |

Đối chiếu: ngưỡng thường ở đây tái lập đúng số của bài (0,799 toàn cohort; 0,782 và 0,790 với 50 và 100 ECG). Ngưỡng nguồn 0,701,
oracle 0,820.

## Chi tiết

**Mô phỏng, theo n (trung vị 4 cấu hình mỏ neo):**

| n | 100 | 300 | 1000 | 3000 |
|---|---|---|---|---|
| RMSE làm trơn / thường | 0,74 | 0,65 | 0,57 | 0,53 |
| Tổn thất BA làm trơn / thường | 0,74 | 0,67 | 0,68 | 0,68 |

**Độ nhạy theo độ rộng h:**

| | h × 0,5 | h × 1 (chính) | h × 2 |
|---|---|---|---|
| Mô phỏng: RMSE làm trơn / thường | 0,84 | 0,65 | 0,58 |
| Mô phỏng: \|độ lệch\| / SD lớn nhất | 0,15 | 0,24 | **1,09** |
| Brugada: SD làm trơn / thường | 0,68 | 0,54 | 0,44 |
| Brugada: BA toàn cohort | 0,7953 | 0,7956 | 0,7939 |
| Brugada: BA với 50 ECG | 0,7879 | 0,7895 | 0,7880 |

## Đọc kết quả

1. **Làm trơn giảm độ dao động của ngưỡng khoảng một nửa trên Brugada** (0,54) và 35% trong mô phỏng, lợi thế tăng theo n đúng
   như lý thuyết (tốc độ n^(−2/5) thay cho n^(−1/3)).
2. **Về độ chính xác, lợi ích nhỏ nhưng nằm ở chỗ cần:** với ít ECG (n = 50), trường hợp xấu (phân vị 10%) tăng từ 0,741 lên
   0,770. Với toàn bộ cohort, hai ngưỡng gần như bằng nhau (−0,004, trong giới hạn định trước).
3. **Cái giá là độ lệch.** Với h × 2, độ lệch đã lớn bằng cả độ dao động trong mô phỏng. Độ rộng định trước (h × 1) giữ độ lệch
   dưới 1/4 SD.
4. **Bootstrap cho ngưỡng làm trơn chưa đạt mức đặt ra** (sai số từng mẫu 0,22, yêu cầu 0,15), dù trung vị đúng và sai số giảm
   theo n (khác với ngưỡng thường, ở đó sai số bootstrap đứng yên 0,24). Vậy "làm trơn thì bootstrap dùng được" đúng về xu hướng,
   chưa đủ tốt ở n ≤ 1000.

## Hạn chế

- Mô phỏng chỉ có mô hình Gauss, nơi mật độ trơn nên làm trơn có lợi thế. Chưa thử điểm số rời rạc, nhiều đỉnh.
- Chỉ mới kiểm trên Brugada với một mỏ neo (CBM-sd15). RBBB và LVH chưa kiểm (cần điểm số bước 14–15).
- Chưa có công thức SD và công thức cỡ mẫu cho ngưỡng làm trơn (tương ứng Định lý 1); đây là việc lý thuyết tiếp theo nếu chọn
  hướng này.
- Ý tưởng làm trơn tiêu chí bậc thang đã có trong y văn (Horowitz 1992); đóng góp ở đây chỉ là áp dụng và kiểm chứng.
