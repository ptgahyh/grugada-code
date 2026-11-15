# Bước 29: chạy lại các phân tích Brugada trên pipeline lọc chung (điểm số bước 27). Kế hoạch (viết và commit TRƯỚC khi chạy)

Ngày viết: 15/11/2026. Lúc viết: đang tạo lại điểm số bước 27 trên máy mới (đúng chuỗi của `analysis_v27/PLAN_vi.md`); chưa chạy
phân tích nào dưới đây trên điểm số đó. Kết quả cũ (pipeline gốc) của mọi bước đã biết, nên đây là **kiểm độ vững**, không phải
kiểm định mới: câu hỏi duy nhất là kết luận cũ có đổi khi mọi nguồn qua cùng bộ lọc thông thấp 40 Hz hay không.

## Nguyên tắc

- Script **không sửa**; chỉ đổi thư mục đầu vào (`harmonized_v27`, điểm số v5/v8/v11 tạo lại từ nó) và thư mục ra `analysis_v29/<bước>/`.
- Tiêu chí đạt/không đạt: **đúng tiêu chí đã ghi trong kế hoạch gốc của từng bước**, cùng ngưỡng. Không thêm, không bỏ.
- Báo cáo bảng "cũ → mới" cho mọi tiêu chí và con số chính, kể cả khi trạng thái đạt/không đạt đổi chiều.
- Danh sách thiết lập "hỏng" dùng trong bước 26 giữ như script (0, 1, 2, 4, 6); bước 27 đã cho thấy danh sách này không đổi.

## 29.0 Tái lập bước 27 trên máy này

`harmonize_v27.py` → `compare_v5.py` → `v8_core.py` → `v8_analyze.py`. So với `analysis_v27/results/` đã commit: BA held-out của
SRC/CATT/ORACLE và hiệu CATT − SRC lệch ≤ 0,005 thì coi là tái lập được (phiên bản PyTorch có thể khác nên không đòi trùng tuyệt đối).
Nếu lệch hơn: báo cáo, và mọi phần dưới dùng điểm số mới.

## Các bước chạy lại

| Bước | Script | Tiêu chí gốc được đối chiếu |
|---|---|---|
| 11 (đầu vào) | `labeller_scores.py` trên `harmonized_v27` | không có (tạo điểm 7 mỏ neo) |
| 13 | `brugada_unified_v2.py`, `stopping_rule.py` | tỉ số dự đoán/quan sát của Định lý 1 (cũ 1,09) và bản không nhãn (cũ 0,98), Spearman; tỉ lệ đạt của quy tắc dừng |
| 16 | `sens_brugada.py` | các kiểm độ nhạy C1, C2 |
| 18c | `est_brugada.py`, `est_brugada_single.py` | sai số từng mẫu đơn của F1 và các ước lượng khác (cũ 0,23; 0,20–0,23) |
| 19 | `cluster_v19.py`, `realign_posthoc.py` | D1–D4 (cũ: 97% độ dịch trong cụm; sửa thành phần lấp 4%) |
| 20 | `smooth_brugada.py` | S2a–c (cũ: SD còn 0,54 lần; BA n = 50: 0,790 so với 0,782) |
| 24 | `brugada24.py` | T5 (cũ: sai số 0,16), T6 |
| 25 | `brugada25.py` | M1–M4 (5 mỏ neo) |
| 26 | `brugada26.py` | G1, G2; L1–L3 (cũ: cổng giữ 100%, đóng 9/12; CATT ≈ 30–50 ECG có nhãn, ≈ 100 ở 5%) |

Không chạy lại: bước 17 (code không còn trong repo, cần viết lại, để riêng); bước 21 và 23 (chính là các phân tích về căn chỉnh và
dải thông trên pipeline gốc; bước 27 thay thế vai trò của 23); mọi mô phỏng (không phụ thuộc dữ liệu).

## Diễn giải định trước

- Một kết luận "vững" nếu trạng thái đạt/không đạt của tiêu chí gốc không đổi. Con số lệch nhỏ được báo cáo nhưng không diễn giải.
- Nếu tiêu chí nào đổi từ đạt sang không đạt: ghi rõ trong README và trong kế hoạch viết bài; phần tương ứng của bản thảo phải nêu
  kết quả ở cả hai pipeline.
- Bước này không quyết định pipeline nào là chính (quyết định của tác giả); nó cung cấp bảng số để quyết định.

## Tệp sẽ tạo

`analysis_v29/<bước>/` (chỉ bảng tổng hợp), `analysis_v29/compare29.py` (bảng cũ → mới), `summary_v29.json`, `README_vi.md`.
Điểm số từng bệnh nhân ở thư mục riêng, không đẩy lên repo.
