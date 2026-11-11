# Bước 18: hệ số quần thể hữu hạn và ước lượng nửa mẫu F1 so với các ước lượng có chứng minh

Kế hoạch `PLAN_vi.md` được commit và đẩy lên lúc 03:50 UTC ngày 11/11/2026 (d7b3549), trước mọi phép chạy.
Bổ sung 18c được viết **sau** khi thấy kết quả 18a/18b và commit (fd7bbd8, 03:54 UTC) trước khi chạy 18c.
Không có sai lệch so với kế hoạch. Mọi con số dưới đây nằm trong `summary_v18.json`.

## Chạy lại

```
python analysis_v18/fpc_sim.py analysis_v18                                  # ~2 phút
python analysis_v18/est_sim.py analysis_v18                                  # ~2 phút
python analysis_v18/est_brugada.py <analysis_v8/scores> analysis_v18         # vài giây
python analysis_v18/est_brugada_single.py <analysis_v8/scores> analysis_v18  # ~1 phút
```

Điểm số bước 8 có nhãn từng bệnh nhân nên không nằm trong kho. Kho chỉ chứa số tổng hợp theo thiết lập.

## Kết quả theo tiêu chí

| Mã | Nội dung | Kết quả | Đạt? |
|---|---|---|---|
| A1 | (1 − n/N)^(1/3) khớp tỉ số SD_kho / SD_iid | trung vị 60 ô: 1,04 (đạt); theo tỉ lệ n/N = 0,1 / 0,2 / 0,35 / 0,5 / 0,8: 0,99 / 1,02 / 1,04 / 1,07 / **1,16** (vượt 1,15) | **Không đạt** (sát ngưỡng, chỉ ở n/N = 0,8) |
| A2 | Số mũ thực nghiệm | γ = **0,242** (KTC 95% 0,234–0,250); không chứa 1/3, không chứa 1/2 | mô tả |
| B1 | Từng ước lượng nằm trong dải [0,80; 1,25] | cả 6 ước lượng đạt (kể cả F1: 0,94; theo n: 0,91–0,95) | **Đạt** |
| B2 | Sai số từng mẫu của F1 ≤ ước lượng có chứng minh tốt nhất + 0,03 | F1 0,244; tốt nhất (bootstrap số, ε = n^(−1/3)) 0,125 | **Không đạt** |
| C1 | Như B2, trên Brugada, từng mẫu đơn | F1 0,231; tốt nhất 0,207 | **Đạt** (dự đoán trước khi chạy là không đạt: dự đoán sai) |

## 18a: hệ số quần thể hữu hạn

| n/N | 0,1 | 0,2 | 0,35 | 0,5 | 0,8 |
|---|---|---|---|---|---|
| R quan sát (trung vị) | 0,958 | 0,942 | 0,902 | 0,850 | 0,680 |
| (1 − n/N)^(1/3) | 0,965 | 0,928 | 0,866 | 0,794 | 0,585 |
| (1 − n/N)^0,242 | 0,975 | 0,947 | 0,901 | 0,846 | 0,677 |

- Công thức của bài dự đoán độ dao động **thấp hơn** thực tế: 4% ở n/N = 0,35, 7% ở 0,5, 16% ở 0,8.
- Kết quả giống nhau ở N = 400, 1200, 4000 và ở cả bốn cấu hình mỏ neo, nên không phải hiệu ứng mẫu nhỏ.
- Hệ quả cho bài: ở Brugada và UCI (n/N tới 0,5) tỉ số dự đoán/quan sát thật ra cao hơn con số đã báo cáo vài phần trăm.
  Ở RBBB và LVH (n/N nhỏ, cao nhất khoảng 0,25) ảnh hưởng khoảng 2% trở xuống.
- Số mũ 0,24 là kết quả thực nghiệm trong một mô hình mô phỏng; chưa có chứng minh.

## 18b: F1 trong mô phỏng biết SD thật (3200 mẫu)

| Ước lượng | Trung vị ước lượng / thật | Khoảng tứ phân vị | Trung vị \|log\| | \|log\| ở n = 100 → 3000 |
|---|---|---|---|---|
| E0 plug-in KDE | 1,04 | 0,91–1,20 | 0,142 | 0,20 → 0,10 |
| **E1 F1 nửa mẫu** | 0,94 | 0,72–1,21 | **0,244** | 0,26 → **0,24** |
| E2 mẫu con b = n^(2/3) | 0,88 | 0,78–0,97 | 0,140 | 0,22 → 0,10 |
| E3 bootstrap định hình lại | 0,94 | 0,80–1,07 | 0,142 | 0,20 → 0,11 |
| E4 bootstrap số, ε = n^(−1/3) | 0,91 | 0,82–1,01 | **0,125** | 0,19 → 0,10 |
| E5 bootstrap thường | 0,95 | 0,72–1,20 | 0,245 | 0,27 → 0,24 |

- **F1 hành xử gần như trùng bootstrap thường** (E5): trung vị đúng, nhưng sai số của từng mẫu không giảm khi n tăng.
  Đây là dấu hiệu của tính không nhất quán đã biết ở ước lượng căn bậc ba.
- Ba ước lượng có chứng minh (E2–E4) có sai số giảm theo n, đổi lại lệch thấp 3–18%.
- Độ nhạy của E4 theo bước ε (B3): ε = n^(−1/4): 0,87, \|log\| 0,145; ε = n^(−0,4): 0,94, \|log\| 0,149.

## 18c: từng mẫu đơn trên Brugada (27 lần chạy × 3 cỡ mẫu × 20 mẫu = 1620 điểm, n = 71–577)

| Ước lượng | Từng mẫu: trung vị (IQR) | Từng mẫu: \|log\| | Trung vị 20 mẫu: tỉ số | Trung vị 20 mẫu: \|log\| |
|---|---|---|---|---|
| E0 plug-in KDE | 1,08 (0,90–1,38) | 0,201 | 1,08 | 0,128 |
| **E1 F1** | 0,99 (0,78–1,24) | 0,231 | 1,01 | 0,109 |
| E2 | 0,91 (0,77–1,15) | 0,229 | 0,88 | 0,207 |
| E3 | 0,95 (0,79–1,20) | 0,214 | 0,93 | 0,158 |
| E4, ε = n^(−1/3) | 0,97 (0,82–1,25) | 0,207 | 0,94 | 0,168 |
| E5 | 0,97 (0,77–1,22) | 0,228 | 0,97 | 0,107 |

- Cột "trung vị 20 mẫu" tái lập bước 15c (F1: 0,98 và 0,114 ở đó; 1,01 và 0,109 ở đây, khác bộ điểm số).
- **Sai số của một mẫu đơn lớn gấp đôi con số trong bài** (0,23 so với 0,11). Bài viết "từ một mẫu con" nhưng code bước 13–15
  lấy trung vị 20 mẫu con. Bản thảo v4 đã sửa câu này.
- Ở cỡ mẫu Brugada (n nhỏ), các ước lượng có chứng minh không hơn F1 rõ rệt. Lợi thế của chúng trong mô phỏng xuất hiện
  từ n khoảng 300 trở lên.
- Dùng số mũ 0,242 thay 1/3 không đổi kết luận (F1 0,234; E4 0,199).

## Đọc kết quả

1. Câu trong bài "F1 gần như không lệch nhưng dao động theo từng mẫu" vẫn đúng. Điều mới: dao động đó **không giảm theo n**,
   vì F1 về bản chất tương đương bootstrap thường.
2. Với đích có n lớn (RBBB, LVH: 500–2000), nên dùng bootstrap số hoặc mẫu con b = n^(2/3). Đây có thể là lý do quy tắc
   dừng hai giai đoạn dùng F1 (F2) không đạt 80%. **Chưa kiểm**: cần điểm số bước 14–15 và một kế hoạch riêng.
3. Hệ số quần thể hữu hạn nên viết với số mũ thực nghiệm khoảng 1/4, ghi rõ là heuristic.

## Hạn chế

- Một mô hình mô phỏng (Gauss, π = 0,2), bốn cấu hình mỏ neo. Chưa thử phân phối lệch, đuôi dày.
- E3 và E4 cài theo ý tưởng của Cattaneo, Jansson, Nagasawa (2020) và Hong, Li (2020); chi tiết thủ tục chưa đối chiếu
  từng bước với bài gốc. E3 vẫn cần Ã từ KDE.
- 18c chỉ có một mỏ neo (CBM-sd15) và được lên kế hoạch sau khi thấy 18b.
