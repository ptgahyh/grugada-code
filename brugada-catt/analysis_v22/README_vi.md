# Bước 22: LVH với mô hình đích nhận nhịp mV. Kết quả

Kế hoạch: `PLAN_vi.md` (commit 11/11, trước khi chạy). Chạy 16–17/11/2026 trên máy của tác giả bằng `analysis_v28/chay_rbbb_lvh.py`
(cùng dữ liệu chuẩn hóa với lần chạy lại bước 15, xem `analysis_v28/README_vi.md` mục "Dữ liệu"). Không sai lệch so với kế hoạch.
**Chưa đủ:** M1 cần AUROC đích từng thiết lập, nằm trong nhật ký huấn luyện trên máy tác giả, chưa lấy về được (máy mất kết nối).

## Kết quả theo tiêu chí

| Mã | Nội dung | Kết quả | Đạt? |
|---|---|---|---|
| M1 | AUROC đích tăng ≥ 0,02 ở ≥ 4/6 thiết lập | chưa có số | **chờ** |
| M2 | Số thiết lập hỏng (SRC thấp hơn oracle > 0,05) | **2/6**: cn→PTB-CS (0,728 so với 0,872), cn→PTB-AT (0,699 so với 0,845) | mô tả |
| M3 = H1 | CATT(A3) − SRC ở thiết lập hỏng, gộp | **+0,128 (0,116–0,140)** | **Đạt** |
| M4 | Định lý 1: dự đoán/quan sát ∈ [0,80; 1,25], Spearman ≥ 0,7 | 1,07, Spearman 0,97; không nhãn 1,15, Spearman 0,94 | **Đạt** |
| M5 | CATT(A3) − SRC ở 4 thiết lập không hỏng | **−0,037 (−0,041 đến −0,033)** | mô tả: CATT có hại |

So sánh khác (gộp, bootstrap bệnh nhân):

| Hiệu | Thiết lập hỏng | Tất cả 6 |
|---|---|---|
| CATT(A3) − SRC | +0,128 (0,116; 0,140) | +0,018 (0,014; 0,023) |
| CATT(A3) − GMM-2 | −0,004 (−0,013; 0,006) | +0,014 (0,011; 0,019) |
| CATT(A3) − Otsu | −0,009 (−0,018; −0,000) | +0,016 (0,013; 0,021) |
| CATT(A1) − SRC | +0,141 (0,130; 0,151) | −0,006 (−0,011; −0,002) |

Ở 4 thiết lập không hỏng, GMM-2 và Otsu còn hại hơn CATT(A3) (−0,060 và −0,066 so với SRC).

## Đọc kết quả

1. **LVH trở thành ví dụ thứ hai có ngưỡng hỏng khi đổi viện.** Khi f nhận điện thế mV, ngưỡng học ở Chapman/Ningbo (Trung Quốc)
   hỏng khi đem sang PTB-XL (Đức), cả hai máy. Ở bước 15 (f z-score) không thiết lập nào hỏng.
2. **CATT sửa được phần lớn** (+0,128; oracle cách SRC khoảng 0,145), nhưng ở thiết lập hỏng CATT chỉ ngang GMM-2/Otsu,
   giống Brugada ở tỉ lệ bệnh tự nhiên (bước 27).
3. **Ở thiết lập không hỏng CATT có hại** (−0,037). Đây đúng là trường hợp cổng (bước 26, 28c) được thiết kế để chặn; kết quả G3 trên
   điểm số mV ở `analysis_v28/` (chờ lấy về).
4. Dự đoán trước khi chạy (CATT với mỏ neo điện thế lệch nhiều hơn khi đích có tỉ lệ bệnh thấp, PTB→cn) phù hợp: ở PTB→cn, CATT(A3)
   thấp hơn SRC 0,048–0,068.

## Tệp

`results/` (chỉ số liệu tổng hợp): `p1_spread.csv`, `p3_stopping.csv`, `p4_point.csv`, `p4_boot.csv`, `summary.json`, `harmonize_log.json`.
