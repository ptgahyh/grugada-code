import wfdb
from scipy.signal import butter, filtfilt

def apply_standard_filter(data, fs=100):
    """Bộ lọc Butterworth Bậc 3 theo Protocol"""
    nyq = 0.5 * fs
    b, a = butter(3, [0.5/nyq, 40/nyq], btype='band')
    return filtfilt(b, a, data)

def get_cleaned_signal(pid):
    """Hàm đọc và làm sạch tín hiệu V1 cho từng bệnh nhân"""
    path = os.path.join(BASE_DIR, 'files', str(pid), str(pid))
    record = wfdb.rdrecord(path)
    # Lấy chuyển đạo V1 (Index 6)
    v1_raw = record.p_signal[:, 6]
    v1_clean = apply_standard_filter(v1_raw, record.fs)
    return v1_clean, record.fs