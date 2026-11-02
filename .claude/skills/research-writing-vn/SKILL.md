---
name: research-writing-vn
description: Grugada project rules for writing the Vietnamese/English report or paper (IMRaD, STROBE/TRIPOD+AI) from results/ and figures/. Use for abstract, methods, results, discussion or submission prep; general manuscript method lives in scientific-writing.
---

# Viết báo cáo khoa học

## Nguồn số liệu
- Chỉ lấy số từ `results/stats_*.csv`, `Final_Research_Data.csv` và đầu ra script. Không làm tròn khác với bảng và không tự tính lại bằng tay.
- Nếu code vừa đổi: chạy lại bước 4 và 5 trước khi viết.

## Cấu trúc IMRaD
1. **Đặt vấn đề**: Brugada, gánh nặng đột tử, khoảng trống (dấu hiệu ECG tiên lượng còn tranh cãi, xem `docs/tong-quan-tai-lieu.md`), mục tiêu cụ thể.
2. **Phương pháp**: dữ liệu (Brugada-HUCA, trích dẫn PhysioNet), tiêu chí chọn, xử lý tín hiệu (lọc 0.5–40 Hz pha bằng 0, dò QRS 12 chuyển đạo, nhịp trung vị, điểm J theo vận tốc không gian), định nghĩa từng đặc trưng + đơn vị, thống kê (Mann-Whitney, AUC bootstrap, BH-FDR, Fisher), phần mềm và phiên bản, link mã nguồn.
3. **Kết quả**: bảng đặc điểm nhóm (Table 1), bảng so sánh có median [IQR], AUC [KTC 95%], q. Hình: `figures/compare_*.png`, ví dụ nhịp trung vị từ `figures/qc/`.
4. **Bàn luận**: so với y văn (Calò 2016, Babai Bigi 2007, BRUGADA-RISK…), cơ chế (chậm dẫn truyền RVOT), **giới hạn** (100 Hz, một trung tâm, 11 ca đột tử, điểm J tự động chưa đối chiếu đo tay, nhãn `brugada=2` chưa rõ), hướng tiếp theo.
5. **Kết luận**: một hai câu, không vượt quá dữ liệu.

## Văn phong
- Kết quả đột tử: "gợi ý", "cần xác nhận", không viết "chứng minh", "dự báo được".
- Báo cáo q-value và KTC cùng p. Nêu cả kết quả âm tính.
- Thuật ngữ: giữ tiếng Anh trong ngoặc lần đầu, ví dụ "nhịp trung vị (median beat)".

## Checklist trước khi nộp
- STROBE (nghiên cứu quan sát) hoặc TRIPOD+AI (nếu có mô hình dự báo/chẩn đoán).
- Mọi trích dẫn đã kiểm tra theo skill `lit-review`.
- Code và kết quả đã commit; ghi commit hash trong phần phương pháp để tái lập.
