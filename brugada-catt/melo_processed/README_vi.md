# Bộ Melo et al. (San Raffaele) đã xử lý

Nguồn: Melo L. et al., *PNAS Nexus* 2023; dữ liệu Dryad doi:10.5061/dryad.s1rn8pkd9 (CC0).
README gốc đề nghị liên hệ tác giả (Prof. Edward R. Grant, UBC) khi dùng lại dữ liệu.

## Nội dung

| File | Nội dung |
|---|---|
| `melo_claris.npz` | `X` (1154, 9, 150), `y`, `leads`, `fs=200`. Máy Claris, tập train gốc. 596 BrS / 558 âm tính |
| `melo_mortara.npz` | `X` (370, 12, 150), `y`. Máy Mortara, tập validation gốc. 137 BrS / 233 âm tính |
| `labels_claris.csv`, `labels_mortara.csv` | Nhãn từng dòng, kèm điểm dự đoán trung bình của mô hình gốc |
| `reconstruct_labels.py` | Script khôi phục nhãn (chạy lại được) |

## Dữ liệu này là gì

- Mỗi bệnh nhân là **một nhịp trung bình**, dài 150 mẫu ở 200 Hz (750 ms). Đỉnh R nằm ở khoảng mẫu 60.
- Thứ tự chuyển đạo của Claris: I, II, III, V1–V6. Thứ tự này được suy ra từ tương quan giữa các đoạn và độ phân tách theo nhãn. V2 (đoạn 4) phân biệt tốt nhất, khớp với sinh lý Brugada.
- Mortara có 12 đoạn. Đoạn 0–8 khớp với Claris (I, II, III, V1–V6; đoạn 6 hơi lệch). Đoạn 9–11 chưa xác định được, có thể là aVR, aVL, aVF.
- **Mỗi chuyển đạo đã được chuẩn hóa z-score** (trung bình 0, độ lệch chuẩn 1). Biên độ tính bằng mV đã mất.
- ECG đều là ECG nền, chưa dùng ajmaline. Chẩn đoán được xác nhận bằng test ajmaline.

## Nhãn được khôi phục thế nào

Bản phát hành **không có file nhãn**. Nhãn được suy ra từ đường ROC đã công bố cùng với điểm dự đoán của từng bệnh nhân. Ba cách kiểm tra:

1. Số ca dương khớp: 596/1154 và 137/370.
2. Số TP và TN ở ngưỡng 0.5 khớp **chính xác** với số tác giả báo cáo: Claris 509/460, Mortara 102/207.
3. Ở Mortara, nhãn suy từ mô hình CNN và nhãn suy từ mô hình DNN trùng nhau 100% (361/361 ca so sánh được).

Nên viết email cho tác giả để xác nhận và ghi rõ cách lấy nhãn trong bài báo.

## Lưu ý chất lượng

- Claris có 3 cặp nhịp gần như trùng hệt nhau (tương quan > 0.999): dòng (565, 626), (596, 800), (729, 739). Mỗi cặp cùng nhãn. Cần gom cùng nhóm khi chia fold.
- Không có tuổi, giới hay ID gốc theo từng bệnh nhân, dù bài báo có phân tích theo nhóm.
