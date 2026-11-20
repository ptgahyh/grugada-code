# Bước 30: độ dao động do huấn luyện lại (5 lần). Kết quả

Kế hoạch: `PLAN_vi.md` (commit b31d0f4, trước khi chạy). Mã: commit 2de33b9. Brugada chạy trên máy cloud 17/11 (pipeline lọc chung,
lần 0 = bước 27, lần 1–4 với SEED_OFFSET = 1000·r cho mọi thành phần huấn luyện). RBBB/LVH/LVH mV chạy trên máy tác giả 18–20/11 (`chay_nhieu_hat_giong.py`, khoảng 10 giờ; một lần dừng vì
hết RAM ở lần 1 LVH, chạy tiếp được từ lượt đã xong).

Kiểm an toàn: ở cả 4 lần × 3 bài toán RBBB/LVH, điểm số khác lần 0, nhãn và cách chia trùng lần 0 (kiểm tự động trong script).
Brugada: lần 1–4 khác lần 0 (AUROC từng mô hình khác nhau), cùng cách chia validation.

## Brugada: 5 lần huấn luyện

| Hiệu (CATT −) | 5 lần | Trung bình | SD giữa các lần | Nửa KTC bootstrap (trung vị) | Cùng dấu | V2 (định trước) | Khoảng gộp (bổ sung sau) |
|---|---|---|---|---|---|---|---|
| SRC, held-out | 0,097 0,098 0,093 0,094 0,083 | **0,093** | 0,006 | 0,012 | 5/5 | **tốt hơn** | 0,080–0,106 → tốt hơn |
| GMM-2, held-out | 0,006 0,009 0,009 0,019 0,003 | 0,009 | 0,006 | 0,010 | 5/5 | tốt hơn | −0,002–0,020 → **không phân biệt được** |
| Otsu, held-out | 0,004 0,009 0,009 0,016 0,003 | 0,008 | 0,005 | 0,009 | 5/5 | tốt hơn | −0,002–0,019 → **không phân biệt được** |
| SRC, tỉ lệ 5% | 0,099 0,098 0,087 0,091 0,085 | 0,092 | 0,006 | 0,019 | 5/5 | tốt hơn | 0,073–0,112 → tốt hơn |
| GMM-2, tỉ lệ 5% | 0,018 0,020 0,019 0,022 0,022 | 0,020 | 0,002 | 0,014 | 5/5 | tốt hơn | 0,006–0,034 → tốt hơn |
| Otsu, tỉ lệ 5% | 0,021 0,023 0,024 0,024 0,026 | 0,024 | 0,002 | 0,015 | 5/5 | tốt hơn | 0,009–0,038 → tốt hơn |

- **V1:** với CATT − SRC, SD giữa các lần ≈ SE bootstrap (tỉ số 0,99 ở held-out, 0,66 ở tỉ lệ 5%): bootstrap không đánh giá thấp
  rõ rệt ở Brugada. Với CATT − GMM-2/Otsu ở held-out, tỉ số 1,1–1,3: độ dao động huấn luyện cỡ bằng độ dao động do bệnh nhân.
- **V3: kết luận chính của Brugada đứng vững.** CATT hơn ngưỡng nguồn 0,083–0,098 ở cả 5 lần.
- **V4:** đúng 5 thiết lập hỏng (cùng 5 thiết lập) ở cả 5 lần.

### Bổ sung sau (không thuộc kế hoạch): quy tắc V2 có lỗ hổng

Khoảng t của V2 chỉ tính độ dao động **giữa các lần huấn luyện**. Cả 5 lần dùng **cùng** bệnh nhân, nên lấy trung bình 5 lần không làm
giảm độ không chắc chắn do mẫu bệnh nhân. Quy tắc V2 vì thế quá dễ dãi: ở held-out, mỗi lần riêng lẻ CATT − GMM-2 có KTC chứa 0, nhưng
V2 vẫn kết luận "tốt hơn". Khoảng gộp SE² ≈ (h/1,96)² + s²/5 tính cả hai nguồn. Theo khoảng gộp, **CATT ngang GMM-2/Otsu ở tỉ lệ bệnh
tự nhiên** (khớp bước 27), và hơn rõ ở tỉ lệ 5%. **Đề xuất viết bài theo khoảng gộp**, ghi rõ là thay đổi sau khi xem kết quả và
nêu cả kết quả theo V2.

## RBBB, LVH, LVH mV: 5 lần huấn luyện (gộp 6 thiết lập trừ khi ghi khác)

| Hiệu (CATT −) | 5 lần | Trung bình | SD giữa các lần | Nửa KTC bootstrap | Tỉ số s/SE | Cùng dấu | V2 (định trước) | Khoảng gộp (bổ sung sau) |
|---|---|---|---|---|---|---|---|---|
| RBBB: SRC | +0,007 +0,011 +0,011 +0,013 +0,012 | +0,011 | 0,002 | 0,003 | 1,25 | 5/5 | tốt hơn | 0,007–0,015 → tốt hơn |
| RBBB: GMM-2 | | +0,007 | 0,003 | 0,002 | 2,30 | 5/5 | tốt hơn | 0,004–0,011 → tốt hơn |
| RBBB: Otsu | | −0,008 | 0,002 | 0,002 | 1,84 | 5/5 | kém hơn | −0,011 – −0,005 → kém hơn |
| LVH: SRC | −0,010 −0,014 −0,012 −0,018 −0,010 | −0,013 | 0,003 | 0,004 | 1,49 | 5/5 | kém hơn | −0,018 – −0,007 → kém hơn |
| LVH: GMM-2 | | +0,031 | 0,004 | 0,004 | 1,92 | 5/5 | tốt hơn | 0,026–0,036 → tốt hơn |
| LVH: Otsu | | +0,037 | 0,004 | 0,004 | 1,88 | 5/5 | tốt hơn | 0,032–0,043 → tốt hơn |
| LVH mV, 2 thiết lập hỏng: SRC | +0,128 +0,141 +0,143 +0,135 +0,123 | **+0,134** | 0,009 | 0,012 | 1,44 | 5/5 | tốt hơn | 0,120–0,148 → tốt hơn |
| LVH mV, hỏng: GMM-2 | | −0,004 | 0,007 | 0,009 | 1,48 | 4/5 | không phân biệt được | −0,015–0,007 → không phân biệt được |
| LVH mV, hỏng: Otsu | | −0,007 | 0,010 | 0,009 | 2,15 | 4/5 | không phân biệt được | −0,019–0,005 → không phân biệt được |
| LVH mV, 4 không hỏng: SRC | −0,037 −0,039 −0,035 −0,037 −0,042 | **−0,038** | 0,003 | 0,004 | 1,35 | 5/5 | kém hơn | −0,042 – −0,034 → kém hơn |
| LVH mV, không hỏng: GMM-2 / Otsu | | +0,023 / +0,025 | | | | 5/5 | tốt hơn | tốt hơn |
| LVH mV, cả 6: SRC | | +0,019 | 0,004 | 0,005 | 1,68 | 5/5 | tốt hơn | 0,013–0,025 → tốt hơn |

Số thiết lập hỏng (V4): RBBB 0 ở 4/5 lần, 1 ở lần 2 (ptb_cs→cn); LVH 0 ở cả 5; LVH mV đúng 2 (cùng hai thiết lập) ở cả 5;
Brugada đúng 5 (cùng năm thiết lập) ở cả 5.

## Kết luận theo tiêu chí

- **V1 (bootstrap đánh giá thấp?):** **có** ở RBBB (tỉ số 1,25), LVH (1,49), LVH mV (1,35–1,68); **không** ở Brugada (0,99; 0,66).
  Với các so sánh CATT − GMM-2/Otsu, tỉ số 1,8–2,8: độ dao động huấn luyện lớn gần gấp đôi độ dao động do bệnh nhân.
  → Mọi KTC trong bài nên là khoảng gộp (bootstrap bệnh nhân + dao động huấn luyện), hoặc nêu rõ chỉ tính bệnh nhân.
- **V2 / khoảng gộp:** hai cách cho cùng kết luận ở RBBB, LVH, LVH mV (chỉ khác ở Brugada so với GMM-2/Otsu tỉ lệ tự nhiên, xem trên).
- **V3:** kết luận chính Brugada đứng.
- Dự đoán trước khi chạy: "V1 có ở RBBB" đúng; "Brugada và LVH mV thiết lập hỏng đứng vững" đúng; "hiệu nhỏ có thể thành không
  phân biệt được": chỉ đúng một phần (Brugada so với GMM-2/Otsu, LVH mV hỏng so với GMM-2/Otsu); nhiều hiệu cỡ 0,01 ở RBBB/LVH vẫn
  ổn định qua 5 lần.

## Đọc kết quả

1. **Hình ảnh chung rõ hơn trước:** CATT sửa lớn khi ngưỡng nguồn hỏng (Brugada +0,093; LVH mV +0,134), nhưng **làm kém đi** khi ngưỡng
   nguồn vốn đúng (LVH −0,013; LVH mV không hỏng −0,038). RBBB (không hỏng) +0,011 là ngoại lệ nhỏ.
2. **Ở thiết lập hỏng, CATT ngang GMM-2/Otsu** (Brugada tỉ lệ tự nhiên, LVH mV). CATT hơn rõ GMM-2/Otsu khi tỉ lệ bệnh thấp
   (Brugada 5%; LVH, tỉ lệ 1,7–10%: +0,031/+0,037).
3. **Câu hỏi còn mở (quan sát sau):** RBBB lần 11/11 cho −0,008, nằm ngoài khoảng của cả 5 lần mới (+0,007 đến +0,013). Khác biệt
   không do hạt giống mà do môi trường chạy (máy, phiên bản thư viện, 23 bản ghi ít hơn, hoặc bản sao bản ghi đọc từ zip HuggingFace).
   Bài nên dùng 5 lần mới, nêu lần 11/11 là chạy trên dữ liệu PhysioNet gốc với kết quả khác dấu, và không diễn giải hiệu RBBB.

## Tệp

`results/per_rep.csv`, `results/summary_table.csv`, `results/broken_per_rep.csv`, `summary_v30.json`.
Đầu vào: `results_brugada_raw/` (Brugada lần 1–4), `results_rbbb_lvh_raw/` (p4_point từng lần; p4_boot ở máy tác giả,
`ket_qua_15_11\\v30`). Chạy lại: `python analysis_v30/aggregate30.py <ket_qua_15_11/v30> analysis_v30/results_brugada_raw`.
