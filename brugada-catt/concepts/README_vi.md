# Lớp khái niệm Brugada khả vi — thí nghiệm đầu tiên

## File

| File | Nội dung |
|---|---|
| `concepts.py` | Lớp khái niệm `BrugadaConcepts` và mô hình `ConceptBottleneck` (CBM) |
| `phantom_test.py` / `phantom_report.json` | Kiểm chứng trên ECG giả lập (6 bài kiểm tra) |
| `run_concepts.py` | So sánh 5 mô hình trên 3 miền, cùng cách chia dữ liệu với `harmonized/baselines.py` |
| `concept_results.json` | Toàn bộ kết quả: AUROC + CI, so sánh cặp, dấu vân tay miền, AUROC đơn biến, trọng số |
| `concept_scores.npy` | Điểm dự đoán từng bệnh nhân (để tự tính lại so sánh cặp) |
| `fig_delta_auroc_ood.png` | Chênh lệch AUROC của CBM so với CNN và ST_V2 ngoài miền huấn luyện |
| `fig_concept_consistency.png` | AUROC đơn biến của 25 khái niệm theo từng miền |

Chạy lại:
```
python phantom_test.py .
python run_concepts.py <thư mục harmonized> .
```
(khoảng 30 phút trên CPU 2 lõi)

## 25 khái niệm

Với mỗi chuyển đạo V1, V2, V3 có 8 khái niệm:
- `J_elev`, `ST80`
- `Corrado` (dạng ổn định: (STJ−ST80)/(|STJ|+|ST80|+ε))
- `slope`, `curv` (khớp bậc hai trên J..J+120 ms)
- `r_prime`
- `tri_w` (độ rộng tại độ cao tương đối h)
- `T`

Thêm một khái niệm toàn cục là `QRS_w`. Tất cả là tỉ số hoặc thời gian, nên bất biến với biến đổi afin theo từng chuyển đạo.

**Tham số mốc học được:**
- ngưỡng J θ;
- độ lệch J;
- tâm đoạn nền;
- vị trí ST80;
- độ cao tam giác h.

## Kiểm chứng giả lập (không cần bác sĩ)

| Kiểm tra | Kết quả |
|---|---|
| Bất biến afin (mV thô và z-score) | sai khác tối đa 1.5e-5 |
| Gradient (autograd so với sai phân hữu hạn) | 0.970 so với 0.967 |
| J_elev V2 so với giá trị thật (nhiễu + trôi nền + lệch ±20 ms) | Spearman 0.955 |
| ST80 V2 (cùng điều kiện) | Spearman 0.87 |
| Vòm và yên ngựa: Corrado V2 | AUROC 0.94 |
| Vòm và yên ngựa: curv V2 | AUROC 0.77 |
| Đo dưới mức một mẫu so với đo cứng (biên độ J) | 0.947 so với 0.944: **không có lợi** |

## Kết quả chính (AUROC)

| | ST_V2 | LogReg | CNN | ConceptLR (mốc cố định) | **CBM** |
|---|---|---|---|---|---|
| Trung bình trong miền (3) | 0.808 | 0.856 | **0.877** | 0.842 | 0.866 |
| Trung bình ngoài miền (9) | 0.808 | 0.793 | 0.809 | 0.798 | **0.821** |
| Balanced accuracy ngoài miền (ngưỡng chọn trên train) | **0.730** | 0.681 | 0.645 | 0.692 | 0.716 |

So sánh cặp ngoài miền (95% CI không chứa 0), trên 9 cặp:

| So sánh | Thắng | Thua | Hòa |
|---|---|---|---|
| CBM với CNN | 2 | 1 | 6 |
| CBM với ST_V2 | 4 | 1 | 4 |
| CBM với ConceptLR | 4 | 0 | 5 |

Dấu vân tay miền (hồi quy logistic phân biệt 3 miền):

| Đầu vào | Độ chính xác 3 lớp | Claris vs Mortara | HUCA vs Melo |
|---|---|---|---|
| Nhịp thô | 0.955 | AUC 0.98 | AUC 0.998 |
| 25 khái niệm | 0.763 | AUC 0.75 | AUC 0.96 |

## Đọc kết quả một cách trung thực

1. **CBM cạnh tranh được với CNN ngoài miền, nhưng chưa vượt trội đều.** Trung bình cao hơn 0.012. Thắng rõ khi test trên Claris, thua rõ ở Mortara → HUCA.
2. **Mốc học được (CBM) luôn bằng hoặc tốt hơn mốc cố định (ConceptLR).** Đây là bằng chứng rằng tính khả vi có ích. Lợi ích đến từ việc học vị trí mốc, không phải từ độ chính xác dưới mẫu.
3. **Khái niệm làm giảm mạnh dấu vân tay thiết bị** (Claris vs Mortara: 0.98 → 0.75). HUCA vẫn khác Melo rất rõ.
4. **Ngưỡng của CBM ổn định hơn CNN khi đổi miền** (balanced accuracy 0.716 so với 0.645), nhưng vẫn kém quy tắc ST_V2.
5. **Nhất quán sinh lý ở cả 3 miền:**
   - J_elev V1/V2 tăng; Corrado > 0 (ST đi xuống); slope âm; T V1/V2 âm; QRS rộng hơn. Tất cả đúng với y văn Brugada.
   - **Không nhất quán:** `curv` và `tri_w` (đổi chiều giữa HUCA và Melo), cùng các khái niệm ở V3. Đây là lý do khả dĩ khiến Mortara → HUCA thua.
6. **Trọng số đa biến có hiện tượng triệt tiêu do cộng tuyến**, ví dụ `r_prime_V1` đơn biến dương nhưng trọng số âm. Chưa nên diễn giải từng trọng số riêng lẻ.
7. **Tham số mốc học được trên cả 3 miền:**
   - J lùi khoảng +10 ms;
   - nền ở −52 ms;
   - ST đo ở 60 ms sau điểm cất cánh;
   - h ≈ 0.21, chạm biên dưới 0.2, nên cần nới khoảng.

## Việc tiếp theo đề xuất

- **Chọn khái niệm theo tính bất biến giữa các miền huấn luyện**, như một ràng buộc kiểu IRM, rồi kiểm tra trên miền giữ lại.
- **Thiết kế lại `curv` và `tri_w`:** cửa sổ ngắn hơn, góc β tương đối.
- **Lặp lại với nhiều cách chia và nhiều seed**, hiệu chỉnh so sánh bội.
- **Kiểm chứng điểm J trên LUDB** (cần tải về máy).
