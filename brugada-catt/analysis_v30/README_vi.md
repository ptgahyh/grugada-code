# Bước 30: độ dao động do huấn luyện lại (5 lần). Kết quả phần 1: Brugada

Kế hoạch: `PLAN_vi.md` (commit b31d0f4, trước khi chạy). Mã: commit 2de33b9. Brugada chạy trên máy cloud 17/11 (pipeline lọc chung,
lần 0 = bước 27, lần 1–4 với SEED_OFFSET = 1000·r cho mọi thành phần huấn luyện). **RBBB/LVH/LVH mV chưa chạy** (máy tác giả,
`chay_nhieu_hat_giong.py`).

Kiểm an toàn: lần 1–4 khác lần 0 (AUROC từng mô hình khác nhau), cùng nhãn và cùng cách chia validation (đã kiểm `y_val`).

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

## Tệp

`results/per_rep.csv`, `results/summary_table.csv`, `results/broken_per_rep.csv`, `summary_v30.json` (sẽ chạy lại khi có RBBB/LVH).
