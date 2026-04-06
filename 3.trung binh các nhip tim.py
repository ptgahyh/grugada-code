import pandas as pd
import os
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (Thay đổi cho đúng máy bạn)
# ==========================================
BASE_DIR = r'D:\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0'
FILES_DIR = os.path.join(BASE_DIR, 'files')

# ==========================================
# 2. ĐỊNH NGHĨA CÁC HÀM (BỘ MÁY XỬ LÝ)
# ==========================================

def apply_standard_filter(data, fs=100):
    """Bộ lọc Butterworth Bậc 3 (0.5Hz - 40Hz) theo Protocol"""
    nyq = 0.5 * fs
    b, a = butter(3, [0.5/nyq, 40/nyq], btype='band')
    return filtfilt(b, a, data)

def get_cleaned_signal(pid):
    """Hàm đọc file thô và làm sạch tín hiệu chuyển đạo V1"""
    path = os.path.join(FILES_DIR, str(pid), str(pid))
    record = wfdb.rdrecord(path)
    # Lấy chuyển đạo V1 (Index 6)
    v1_raw = record.p_signal[:, 6]
    v1_clean = apply_standard_filter(v1_raw, record.fs)
    return v1_clean, record.fs

def extract_average_beat(signal, fs=100):
    """Tính toán nhịp tim trung bình để khử nhiễu tuyệt đối"""
    # Tìm đỉnh R (khoảng cách tối thiểu 0.6s)
    peaks, _ = find_peaks(signal, distance=int(fs*0.6), height=np.mean(signal))
    
    beats = []
    for r in peaks:
        # Cửa sổ 1000ms: 40 mẫu trước R (400ms), 60 mẫu sau R (600ms)
        start, end = r - 40, r + 60
        if start > 0 and end < len(signal):
            beat = signal[start:end]
            # Baseline Correction dựa trên đoạn PR (mẫu 30-35 trong cửa sổ)
            baseline = np.mean(beat[30:35])
            beats.append(beat - baseline)
            
    return np.mean(beats, axis=0) if len(beats) > 0 else None

# ==========================================
# 3. THỰC THI VÀ VẼ HÌNH XÁC MINH
# ==========================================

def run_verification(pid, title_ext):
    try:
        print(f"🔄 Đang xử lý ca {pid}...")
        sig_clean, fs = get_cleaned_signal(pid)
        avg_beat = extract_average_beat(sig_clean, fs)
        
        if avg_beat is not None:
            plt.figure(figsize=(10, 5))
            plt.plot(avg_beat, color='blue' if 'Brugada' in title_ext else 'red', linewidth=2)
            plt.axhline(0, color='black', linestyle='--', alpha=0.5) # Đường đẳng điện
            
            # Đánh dấu các mốc quan trọng để kiểm tra mắt
            r_idx = 40 # Đỉnh R luôn nằm ở mẫu 40 do ta cắt 40 trước R
            j_idx = r_idx + 6 # J-point (60ms sau R)
            
            plt.scatter(j_idx, avg_beat[j_idx], color='black', zorder=5, label='J-point Candidate')
            
            plt.title(f"Xác minh Nhịp tim trung bình - ID {pid} ({title_ext})")
            plt.xlabel("Mẫu (10ms mỗi mẫu)")
            plt.ylabel("Biên độ (mV)")
            plt.legend()
            plt.grid(True, alpha=0.2)
            plt.show()
        else:
            print(f"❌ Không tìm thấy đủ nhịp tim cho ID {pid}")
    except Exception as e:
        print(f"❌ Lỗi khi xử lý ID {pid}: {e}")

# Chạy xác minh cho 2 ca quan trọng nhất
run_verification(428031, "Brugada Đột tử điển hình")
run_verification(1142915, "Đột tử ẩn - Nhãn Bình thường")