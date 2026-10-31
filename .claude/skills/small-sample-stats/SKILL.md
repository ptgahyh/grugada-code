---
name: small-sample-stats
description: "Grugada project statistics rules for tiny event groups (11 SCD cases): AUC with bootstrap CI, BH q-values, Fisher exact, no multivariable/ML models for SCD. Use whenever comparing groups or interpreting significance in this repo; general test selection lives in statistical-analysis."
---

# Thống kê với cỡ mẫu nhỏ

## Bước 0: đếm trước khi tính
In ra số ca mỗi nhóm. Cỡ mẫu hiệu dụng của kết cục nhị phân ≈ **min(số ca có biến cố, số ca không)**. Trong dự án Grugada: đột tử = 11 (toàn cohort), 5 (trong nhóm Brugada).

## Chọn kiểm định
| Tình huống | Dùng |
|---|---|
| Biến liên tục, 2 nhóm, phân phối lệch/mẫu nhỏ | Mann-Whitney U |
| Bảng 2×2 có ô < 5 | Fisher exact (không dùng χ²) |
| Nhiều đặc trưng cùng một câu hỏi | Hiệu chỉnh Benjamini-Hochberg (FDR) → báo q-value |
| Một giả thuyết đặt trước từ y văn | Kiểm định riêng, không cần hiệu chỉnh cho cả họ; ghi rõ là giả thuyết đặt trước |

## Báo cáo bắt buộc cho mỗi đặc trưng
1. n mỗi nhóm.
2. Median [IQR] mỗi nhóm.
3. **Cỡ hiệu ứng + KTC 95%**: AUC = U/(n1·n0) (0.5 = không phân biệt), KTC bằng bootstrap phân tầng ≥ 2000 lần, đặt seed cố định.
4. p và q (BH).
5. Nếu n nhóm nhỏ < 20: ghi "thăm dò, cần cohort độc lập xác nhận".

Hàm sẵn có: `auc_with_ci`, `bh_qvalues` trong `5.ve bieu do va thong ke.py`.

## Không được làm
- Xây hồi quy đa biến hay mô hình học máy dự báo đột tử với 11 ca. Theo Riley et al. (BMJ 2020) cần đủ số biến cố cho mỗi tham số ứng viên; 11 ca không đủ dù chỉ 1–2 biến. Nếu người dùng muốn, giải thích lý do và đề xuất: (a) kiểm định giả thuyết đơn biến đặt trước, (b) gộp dữ liệu ngoài, (c) chuyển sang câu hỏi chẩn đoán Brugada (76 ca).
- Chọn ngưỡng tối ưu (Youden) rồi báo độ nhạy/đặc hiệu trên cùng dữ liệu; kết quả sẽ lạc quan quá mức.
- Viết "cực kỳ ý nghĩa" vì p < 0.05; viết "khác biệt có ý nghĩa thống kê (q = …), AUC … [KTC …]".
- Chạy nhiều phân tích rồi chỉ báo phân tích có p < 0.05 (p-hacking). Báo cáo hết các phân tích đã chạy.

## Khi làm mô hình chẩn đoán Brugada (76 vs 287)
- Kiểm định chéo phân tầng lặp lại (ví dụ 5-fold × 20) hoặc bootstrap optimism correction; mọi bước chọn biến/chuẩn hoá nằm **bên trong** vòng CV.
- Báo AUC, calibration (calibration plot, slope), độ nhạy/đặc hiệu tại ngưỡng đặt trước.
- Checklist báo cáo: TRIPOD+AI (2024).
