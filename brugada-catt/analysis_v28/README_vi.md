# Bước 28: RBBB/LVH chạy lại, ngưỡng làm trơn, cỡ mẫu, cổng. Kết quả (phần 1)

Kế hoạch: `PLAN_vi.md` (commit trước khi có dữ liệu). Chạy 16–17/11/2026 trên máy của tác giả: `python chay_rbbb_lvh.py`
(16 nhân, RAM 7 GB, 3 tiến trình; khoảng 3,7 giờ). **Phần 2 (28a–28c) chưa có:** các tệp `v28/*` còn trên máy tác giả.

## Dữ liệu

PTB-XL 1.0.3: khớp đúng số cũ (RBBB: CS 12.403/297, AT 8.278/244; LVH: CS 12.843/941, AT 8.352/810).

ecg-arrhythmia 1.0.0 lấy từ **bản sao HuggingFace**, không phải PhysioNet: zip có 89.960 header cho 45.152 mã bản ghi (gần như mọi
bản ghi lặp hai lần) và một số tệp `.mat` hỏng. Script chuẩn hóa (bản 3) gom mọi bản sao của một mã bản ghi và thử lần lượt:

| | Số |
|---|---|
| Mã bản ghi | 45.152 (đúng bản gốc) |
| Đọc được ở bản sao đầu | 43.073 |
| Phải dùng bản sao khác | 2.054 |
| Hỏng ở mọi bản sao | 24 |
| Header sai định dạng | 1 (bản gốc: 2) |
| RBBB: số bản ghi dùng (dương tính) | **43.380 (1.733)**; lần 11/11: 43.403 (1.734) |
| LVH: số bản ghi dùng (dương tính) | **38.221 (637)**; lần 11/11: 38.242 (637) |

Mất 23 bản ghi RBBB (1 dương tính) và 21 bản ghi LVH (0 dương tính) so với bản PhysioNet. Ghi là hạn chế nhỏ.

## Kiểm việc chạy lại bước 14–15 (tiêu chí của PLAN: P4 lệch ≤ 0,005 mỗi ô)

**Không đạt.** RBBB: 25/54 ô lệch > 0,005; LVH: 27/60 ô. ORACLE khớp (lệch ≤ 0,004), nên dữ liệu đích gần như trùng; phần lệch đến
từ việc huấn luyện lại mô hình (máy, phiên bản PyTorch, số tiến trình khác; 23 bản ghi ít hơn). Theo kế hoạch, mọi phần sau dùng
điểm số mới.

| Hiệu gộp (6 thiết lập, KTC 95% bootstrap bệnh nhân) | Lần 11/11 | Lần 16/11 |
|---|---|---|
| RBBB: CATT(A2) − SRC | −0,008 (−0,013; −0,005) | **+0,007 (0,004; 0,010)** |
| RBBB: CATT(A2) − GMM-2 | −0,008 | +0,005 |
| RBBB: CATT(A2) − Otsu | −0,023 | −0,006 |
| LVH: CATT(A3) − SRC | −0,010 (−0,012; −0,005) | −0,010 (−0,013; −0,004) |
| LVH: CATT(A3) − GMM-2 | +0,039 | +0,033 |
| LVH: CATT(A3) − Otsu | +0,046 | +0,039 |

**Phát hiện quan trọng (quan sát sau):** với RBBB, dấu của CATT(A2) − SRC **đổi chiều** giữa hai lần huấn luyện, dù cả hai KTC đều
không chứa 0. KTC bootstrap bệnh nhân không tính độ dao động do huấn luyện lại mô hình. Phần lệch nằm ở hai thiết lập cn→PTB
(CATT(A2): 0,925 → 0,964 và 0,927 → 0,968). Hệ quả cho bài: hiệu nhỏ cỡ ±0,01 trên RBBB **không được diễn giải là có hay không có
lợi**; muốn kết luận cần nhiều lần huấn luyện (ví dụ 5 hạt giống huấn luyện) và báo cáo độ dao động giữa các lần. LVH ổn định.

Định lý 1 (P1) giữ: RBBB dự đoán/quan sát 0,81 (cũ 0,80), Spearman 0,80 (cũ 0,87); LVH 1,07 (cũ 1,06), Spearman 0,99.
Quy tắc dừng (P3): RBBB vẫn không đạt (48–62% đạt đích ±20%); LVH bản một giai đoạn tụt (77% so với 83–87%), hai giai đoạn
81–87% (cũ 89–90%).

## Tệp

`rerun_v14/`, `rerun_v15/`: số liệu tổng hợp của lần chạy lại (cùng định dạng `analysis_v14/results`, `analysis_v15/results`).
Bước 22: `analysis_v22/results/`, `analysis_v22/README_vi.md`.
