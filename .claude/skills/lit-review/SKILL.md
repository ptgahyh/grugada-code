---
name: lit-review
description: Grugada project rules for updating docs/tong-quan-tai-lieu.md: verified sources only, table format, link each finding to project features. Use when adding or checking literature for this repo; for searching use paper-lookup and literature-review.
---

# Tổng quan y văn có kiểm chứng

## Quy tắc số 1: không bịa trích dẫn
- Chỉ ghi một bài báo nếu đã thấy nó trong kết quả tìm kiếm hoặc mở được trang. Ghi kèm URL.
- Con số (HR, AUC, n, độ nhạy) phải lấy từ nguồn đã đọc. Nếu chỉ đọc tóm tắt hoặc bài tổng hợp, ghi rõ *(từ tóm tắt)*.
- Không chắc tên tác giả/năm/tạp chí → bỏ chi tiết đó, đừng đoán.

## Quy trình
1. Tách câu hỏi thành 3–6 truy vấn cụ thể (tên tác giả + năm + từ khoá nếu đã biết). Gửi các truy vấn song song.
2. Ưu tiên nguồn: hướng dẫn/đồng thuận (ESC, AHA/HRS, J-wave consensus) → phân tích gộp → cohort đa trung tâm lớn → nghiên cứu đơn trung tâm → tóm tắt hội nghị/luận văn.
3. Với mỗi nghiên cứu ghi: thiết kế, n, số biến cố, thời gian theo dõi, kết quả chính kèm cỡ hiệu ứng, giới hạn.
4. Liên hệ với dự án: đặc trưng nào đã có trong `ecg_pipeline.FEATURE_COLUMNS`, đặc trưng nào có thể thêm, kết quả dự án khớp hay ngược y văn.
5. Cập nhật `docs/tong-quan-tai-lieu.md` (giữ cấu trúc bảng hiện có, thêm nguồn vào cuối file).

## Lưu ý về mạng trong phiên cloud
- physionet.org, zenodo.org có thể bị chặn; WebSearch vẫn dùng được, PMC (`pmc.ncbi.nlm.nih.gov`) thường mở được qua kết quả tìm kiếm.
- Connector PubMed/bioRxiv của plugin bio-research cần mạng cho phép; nếu không kết nối được, báo người dùng thay vì bỏ qua im lặng.

## Mẫu một mục trong bảng
| Nghiên cứu | Thiết kế | Kết quả | Có trong dự án? |
|---|---|---|---|
| Tác giả et al. Tạp chí Năm | n BN, số trung tâm, theo dõi | con số chính *(từ tóm tắt)* | ✅ tên_cột / ❌ lý do |
