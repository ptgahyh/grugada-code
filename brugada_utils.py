"""Các hàm xử lý dùng chung cho toàn bộ pipeline Brugada-HUCA.

Đường dẫn dữ liệu lấy từ biến môi trường BRUGADA_DIR (nếu có),
nếu không thì dùng đường dẫn mặc định bên dưới.
"""
import os

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, find_peaks

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN (chỉ sửa ở đây)
# ==========================================
BASE_DIR = os.environ.get(
    'BRUGADA_DIR',
    r'D:\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0',
)
METADATA_PATH = os.path.join(BASE_DIR, 'metadata.csv')
FILES_DIR = os.path.join(BASE_DIR, 'files')

# Thứ tự chuyển đạo: I, II, III, aVR, aVL, aVF, V1..V6
LEAD_II = 1
LEAD_V1 = 6

# Cửa sổ cắt nhịp quanh đỉnh R (giây)
PRE_R = 0.4
POST_R = 0.6

FEATURE_COLUMNS = ['V1_J', 'V1_Slope', 'V1_T_Amp', 'V1_QRS', 'HR', 'n_beats']


def apply_standard_filter(data, fs):
    """Bộ lọc Butterworth bậc 3 (0.5-40 Hz) theo Protocol."""
    nyq = 0.5 * fs
    b, a = butter(3, [0.5 / nyq, min(40, 0.9 * nyq) / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)


def load_clean_record(pid):
    """Đọc bản ghi 12 chuyển đạo và lọc tất cả các chuyển đạo."""
    path = os.path.join(FILES_DIR, str(pid), str(pid))
    record = wfdb.rdrecord(path)
    return apply_standard_filter(record.p_signal, record.fs), record.fs


def detect_r_peaks(lead_ii, fs):
    """Tìm đỉnh R trên chuyển đạo II (QRS ở V1 thường âm, dễ bắt nhầm sóng T).

    Nếu QRS ở II chủ yếu âm (lệch trục), đảo dấu trước khi tìm đỉnh.
    """
    sig = lead_ii if np.max(lead_ii) >= -np.min(lead_ii) else -lead_ii
    peaks, _ = find_peaks(sig, distance=int(0.35 * fs),
                          height=0.4 * np.percentile(sig, 99))
    return peaks


def average_beats(signal, peaks, fs):
    """Nhịp tim trung bình cho mọi chuyển đạo, đã hiệu chỉnh đường đẳng điện.

    Trả về (beats [mẫu x chuyển đạo], chỉ số R trong cửa sổ, số nhịp dùng).
    """
    pre, post = int(PRE_R * fs), int(POST_R * fs)
    # Đường đẳng điện: đoạn PR, 100 ms -> 50 ms trước R
    b0, b1 = pre - int(0.10 * fs), pre - int(0.05 * fs)
    beats = []
    for r in peaks:
        start, end = r - pre, r + post
        if start >= 0 and end <= len(signal):
            beat = signal[start:end]
            beats.append(beat - beat[b0:b1].mean(axis=0))
    if not beats:
        return None, pre, 0
    return np.mean(beats, axis=0), pre, len(beats)


def qrs_bounds(beats, r_idx, fs):
    """Ước lượng điểm bắt đầu QRS và điểm J (kết thúc QRS).

    Dùng tổng |đạo hàm| của cả 12 chuyển đạo: QRS là vùng biến thiên nhanh nhất.
    """
    vel = np.abs(np.diff(beats, axis=0)).sum(axis=1)
    lo = max(r_idx - int(0.12 * fs), 0)
    hi = min(r_idx + int(0.16 * fs), len(vel))
    thr = 0.2 * vel[lo:hi].max()
    active = np.where(vel[lo:hi] >= thr)[0] + lo
    return int(active[0]), int(active[-1]) + 1


def extract_features(pid):
    """Trích đặc trưng V1. Luôn trả về dict đủ cột (NaN nếu lỗi)."""
    result = dict.fromkeys(FEATURE_COLUMNS, np.nan)
    try:
        signal, fs = load_clean_record(pid)
        peaks = detect_r_peaks(signal[:, LEAD_II], fs)
        beats, r_idx, n = average_beats(signal, peaks, fs)
        if beats is None:
            print(f"⚠️ ID {pid}: không tìm được nhịp hợp lệ")
            return result

        onset, j_idx = qrs_bounds(beats, r_idx, fs)
        st80_idx = j_idx + int(round(0.08 * fs))
        t_end = r_idx + int(0.45 * fs)
        v1 = beats[:, LEAD_V1]

        result.update({
            # 1. Chênh J (mV)
            'V1_J': round(v1[j_idx], 4),
            # 2. Độ dốc ST từ J đến J+80ms (mV/s)
            'V1_Slope': round((v1[st80_idx] - v1[j_idx]) / 0.08, 4),
            # 3. Giá trị thấp nhất vùng ST-T (mV)
            'V1_T_Amp': round(np.min(v1[st80_idx:t_end]), 4),
            # 4. Độ rộng QRS (ms)
            'V1_QRS': round((j_idx - onset) * 1000 / fs),
            'HR': round(60 * fs / np.median(np.diff(peaks)), 1) if len(peaks) > 1 else np.nan,
            'n_beats': n,
        })
    except Exception as e:
        print(f"❌ Lỗi khi xử lý ID {pid}: {e}")
    return result
