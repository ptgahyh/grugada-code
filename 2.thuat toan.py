"""Các thuật toán xử lý nằm trong brugada_utils.py (dùng chung cho mọi bước):

- apply_standard_filter:  Butterworth bậc 3, 0.5-40 Hz
- detect_r_peaks:         tìm đỉnh R trên chuyển đạo II
- median_beat:            nhịp trung vị 12 chuyển đạo, loại nhịp lạc (tương quan < 0.9),
                          hiệu chỉnh đẳng điện theo đoạn PR
- upsample:               nội suy nhịp trung bình 100 Hz -> 1000 Hz
- qrs_bounds:             điểm bắt đầu QRS và điểm J (tốc độ biến thiên không gian)
- t_wave:                 đỉnh T và kết thúc T (phương pháp tiếp tuyến)
- features_from_analysis: các đặc trưng (xem README.md)

Thống kê nằm trong brugada_stats.py (AUC + CI bootstrap, hồi quy logistic Firth, FDR).

Chạy file này để thử thuật toán trên một bệnh nhân:
    python "2.thuat toan.py" 428031
"""
import sys

from brugada_utils import extract_features

if __name__ == '__main__':
    pid = sys.argv[1] if len(sys.argv) > 1 else 428031
    for k, v in extract_features(pid).items():
        print(f"{k:12s} {v}")
