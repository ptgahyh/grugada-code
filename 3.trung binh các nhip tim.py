import matplotlib.pyplot as plt

from brugada_utils import (LEAD_II, LEAD_V1, average_beats, detect_r_peaks,
                           load_clean_record, qrs_bounds)


def run_verification(pid, title_ext):
    try:
        print(f"🔄 Đang xử lý ca {pid}...")
        signal, fs = load_clean_record(pid)
        peaks = detect_r_peaks(signal[:, LEAD_II], fs)
        beats, r_idx, n = average_beats(signal, peaks, fs)

        if beats is None:
            print(f"❌ Không tìm thấy đủ nhịp tim cho ID {pid}")
            return

        onset, j_idx = qrs_bounds(beats, r_idx, fs)
        v1 = beats[:, LEAD_V1]
        t_ms = [(i - r_idx) * 1000 / fs for i in range(len(v1))]

        plt.figure(figsize=(10, 5))
        plt.plot(t_ms, v1, color='blue' if 'Brugada' in title_ext else 'red', linewidth=2)
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)  # Đường đẳng điện
        plt.axvspan(t_ms[onset], t_ms[j_idx], color='gray', alpha=0.15, label='QRS')
        plt.scatter(t_ms[j_idx], v1[j_idx], color='black', zorder=5, label='J-point')

        plt.title(f"Nhịp tim trung bình V1 - ID {pid} ({title_ext}, {n} nhịp)")
        plt.xlabel("Thời gian so với đỉnh R (ms)")
        plt.ylabel("Biên độ (mV)")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.show()
    except Exception as e:
        print(f"❌ Lỗi khi xử lý ID {pid}: {e}")


if __name__ == '__main__':
    # Chạy xác minh cho 2 ca quan trọng nhất
    run_verification(428031, "Brugada Đột tử điển hình")
    run_verification(1142915, "Đột tử ẩn - Nhãn Bình thường")
