# Bước 29: các phân tích Brugada chạy lại trên pipeline lọc chung (bước 27)

Ngày chạy: 15/11/2026. Kế hoạch: `PLAN_vi.md` (commit 0d0b161, trước khi chạy). Script không sửa; chỉ đổi dữ liệu đầu vào.

## Kết luận

**Không tiêu chí nào đổi trạng thái đạt/không đạt** (580 giá trị đối chiếu trong `compare29_all.csv`, 0 lần đổi chiều).
Mọi kết luận của bước 13, 16, 18c, 19, 20, 24, 25, 26 trên Brugada giữ nguyên khi mọi nguồn qua cùng bộ lọc 40 Hz.

**29.0 Tái lập bước 27:** bảng bootstrap gộp và bảng tỉ lệ 5% tạo lại trên máy mới **trùng tuyệt đối** bản đã commit (sai khác 0,0).

## Bảng cũ → mới (pipeline gốc → lọc chung)

| Bước | Đại lượng | Cũ | Mới | Tiêu chí |
|---|---|---|---|---|
| 13 | Định lý 1: trung vị dự đoán/quan sát (SD) | 1,09 | 1,09 | giữ |
| 13 | Spearman dự đoán với quan sát | 0,915 | 0,916 | giữ |
| 13 | Bản không nhãn (plug-in một mẫu): trung vị tỉ số | 1,08 | 1,10 | giữ |
| 13 | Quy tắc dừng (mẫu thử 100, KDE): tỉ lệ đạt đích, ε = 0,2 / 0,3 | 62% / 65% | 60% / 69% | vẫn < 80% |
| 16 | S1a, S1b | không đạt | không đạt | giữ |
| 18c | F1 (nửa mẫu): trung vị \|log\| sai số từng mẫu đơn | 0,26 | 0,27 | giữ |
| 18c | Các ước lượng khác | 0,19–0,25 | 0,23–0,31 | giữ (đều kém hơn chút) |
| 19 | Phần độ dịch logit lớp âm nằm trong cùng cụm (K = 5) | 97% | 98% | giữ |
| 19 | Sửa thành phần lấp khoảng cách tới oracle / CATT lấp | 4% / 91% | 10% / 90% | giữ |
| 19 | TV dạng sóng lớp âm: HUCA–Claris / Claris–Mortara | 0,67 / 0,24 | 0,70 / 0,16 | giữ |
| 20 | S2a: SD ngưỡng làm trơn / thường | 0,54 | 0,56 | đạt (≤ 0,85) |
| 20 | BA với 50 ECG: thường → làm trơn | 0,782 → 0,790 | 0,789 → 0,796 | đạt |
| 20 | Phân vị 10% với 50 ECG: thường → làm trơn | 0,741 → 0,770 | 0,751 → 0,774 | |
| 20 | Toàn cohort: thường → làm trơn | 0,799 → 0,796 | 0,800 → 0,801 | đạt |
| 24 | T5: G2 từng mẫu đơn, trung vị tỉ số / \|log\| | 0,92 / 0,16 | 0,90 / 0,16 | đạt |
| 24 | T6: số ECG cần, làm trơn so với thường (ε = 0,30) | 70 / 431 | 69 / 411 | |
| 24 | T6: (ε = 0,15) | 399 / 3449 | 389 / 3290 | |
| 25 | M1–M4 (5 mỏ neo) | đạt cả 4 | đạt cả 4 | giữ |
| 26 | Cổng z = 2 (thường): giữ lợi ích ở thiết lập hỏng | 100% | 100% | G1 đạt |
| 26 | Cổng z = 2 (thường): số lần mở ở thiết lập không hỏng (trên 12) | 3 | 4 | |
| 26 | Cổng z = 2 (làm trơn): số lần mở ở thiết lập không hỏng | 8 | 7 | |
| 26 | CATT tương đương k ECG có nhãn: tự nhiên (thường / làm trơn) | 50 / 30 | 50 / 50 | |
| 26 | CATT tương đương k ECG có nhãn ở tỉ lệ 5% | 100 | 100 | |

Ghi chú:
- Cột "cũ" và "mới" được tính bằng **cùng một hàm** trên CSV/JSON cũ và mới (`compare29.py`). Vài con số "cũ" vì thế khác chút so với
  con số đã ghi trong README gốc khi README dùng thống kê khác (ví dụ 18c: README gốc ghi 0,23; hàm ở đây cho 0,26 trên cùng CSV cũ).
- Bước 13 mới có 558 điểm thay vì 561 (3 điểm bị script loại do điều kiện sẵn có của nó).
- Không chạy lại: bước 17 (code không còn), 21, 23, các mô phỏng.

## Chạy lại

Xem `/home/claude/work/chain27.sh`, `chain29.sh` trong nhật ký phiên; tóm tắt:
```
python analysis_v27/harmonize_v27.py harmonized <melo_processed> <h27>
python concepts_v5/compare_v5.py <h27> <v5> <chunk> 2 ; python analysis_v8/v8_core.py <h27> <v5> <v8> <chunk> 2
python analysis_v11/labeller_scores.py <h27> <v5> <v11> <chunk> 2
python analysis_v13/brugada_unified_v2.py <v11> analysis_v29/v13 ; python analysis_v13/stopping_rule.py <v11> analysis_v29/v13
python analysis_v16/sens_brugada.py <v11> analysis_v29/v16
python analysis_v18/est_brugada.py <v8> analysis_v29/v18 ; python analysis_v18/est_brugada_single.py <v8> analysis_v29/v18
python analysis_v19/cluster_v19.py <h27> <v8> huca_meta/metadata.csv analysis_v29/v19 <thư mục riêng>
python analysis_v20/smooth_brugada.py <v8> analysis_v29/v20 ; python analysis_v24/brugada24.py <v8> analysis_v29/v24
python analysis_v25/brugada25.py <v8> analysis_v29/v25 ; python analysis_v26/brugada26.py <v8> analysis_v29/v26
python analysis_v29/compare29.py
```
Bước 24, 25: `summary_v2x.json` tạo bằng `analyze2x.py` trên thư mục tạm gồm CSV mô phỏng cũ (không phụ thuộc dữ liệu) và CSV Brugada mới.
