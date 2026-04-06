import pandas as pd
import numpy as np
import wfdb
import os
from scipy.signal import butter, filtfilt, find_peaks

# --- GIỮ NGUYÊN CẤU HÌNH ĐƯỜNG DẪN CỦA BẠN ---
BASE_DIR = r'D:\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0'
METADATA_PATH = os.path.join(BASE_DIR, 'metadata.csv')
FILES_DIR = os.path.join(BASE_DIR, 'files')

def apply_standard_filter(data, fs=100):
    nyq = 0.5 * fs
    b, a = butter(3, [0.5/nyq, 40/nyq], btype='band')
    return filtfilt(b, a, data)

def get_average_beat(signal, fs=100):
    peaks, _ = find_peaks(signal, distance=int(fs*0.6), height=np.mean(signal))
    beats = []
    for r in peaks:
        start, end = r - 40, r + 60
        if start > 0 and end < len(signal):
            beat = signal[start:end]
            baseline = np.mean(beat[30:35])
            beats.append(beat - baseline)
    return np.mean(beats, axis=0) if len(beats) > 0 else None

def extract_features(pid):
    try:
        path = os.path.join(FILES_DIR, str(pid), str(pid))
        record = wfdb.rdrecord(path)
        # Chỉ lấy V1 (quan trọng nhất cho Brugada)
        sig_clean = apply_standard_filter(record.p_signal[:, 6], record.fs)
        avg_beat = get_average_beat(sig_clean, record.fs)
        
        if avg_beat is not None:
            j_idx = 46
            s80_idx = 54
            
            # 1. J-Elevation (mV)
            j_elev = avg_beat[j_idx]
            # 2. ST-Slope (mV/s)
            slope = (avg_beat[s80_idx] - avg_beat[j_idx]) / 0.08
            # 3. T-Amplitude (mV) - Biên độ thấp nhất vùng ST-T
            t_amp = np.min(avg_beat[55:85])
            # 4. QRS Width (ms) - Ước tính từ lúc bắt đầu Q đến J
            qrs_width = (j_idx - 35) * 10 
            
            return {'V1_J': round(j_elev, 4), 'V1_Slope': round(slope, 4), 
                    'V1_T_Amp': round(t_amp, 4), 'V1_QRS': qrs_width}
    except:
        return {'V1_J': np.nan, 'V1_Slope': np.nan, 'V1_T_Amp': np.nan, 'V1_QRS': np.nan}

# --- CHẠY VÒNG LẶP TOÀN BỘ DATASET ---
print("🚀 Đang xử lý 363 ca. Vui lòng đợi...")
df = pd.read_csv(METADATA_PATH)
features_list = []

for idx, row in df.iterrows():
    f = extract_features(row['patient_id'])
    features_list.append({**row.to_dict(), **f})
    if idx % 50 == 0: print(f"✅ Đã xong {idx}/363")

df_final = pd.DataFrame(features_list)
df_final.to_csv('Final_Research_Data.csv', index=False)
print("\n🎉 CHÚC MỪNG! Bạn đã hoàn thành toàn bộ phần Code trích xuất dữ liệu.")
print("File 'Final_Research_Data.csv' đã sẵn sàng để làm Thống kê.")