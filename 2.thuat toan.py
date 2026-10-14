"""Các thuật toán xử lý nằm trong brugada_utils.py (dùng chung cho mọi bước):

- apply_standard_filter: Butterworth bậc 3, 0.5-40 Hz
- detect_r_peaks:        tìm đỉnh R trên chuyển đạo II
- average_beats:         nhịp trung bình 12 chuyển đạo, hiệu chỉnh đẳng điện theo đoạn PR
- qrs_bounds:            điểm bắt đầu QRS và điểm J
- extract_features:      đặc trưng V1 (J, độ dốc ST, ST-T thấp nhất, QRS)

Chạy file này để thử thuật toán trên một bệnh nhân.
"""
import sys

from brugada_utils import extract_features

if __name__ == '__main__':
    pid = sys.argv[1] if len(sys.argv) > 1 else 428031
    print(f"ID {pid}: {extract_features(pid)}")
