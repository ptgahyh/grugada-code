"""Xử lý tín hiệu và trích đặc trưng ECG cho bộ dữ liệu Brugada-HUCA.

Đường dẫn dữ liệu (theo thứ tự ưu tiên):
1. Biến môi trường BRUGADA_DIR
2. Thư mục data/ cạnh file này (do "0.tai du lieu.py" tải về)
3. Đường dẫn ổ D:\\ cũ
"""
import os

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, find_peaks, resample_poly

# ==========================================
# CẤU HÌNH
# ==========================================
_HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATA_DIR = os.path.join(_HERE, 'data')
_OLD_DATA_DIR = r'D:\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0'


def _default_base_dir():
    if os.environ.get('BRUGADA_DIR'):
        return os.environ['BRUGADA_DIR']
    if os.path.exists(os.path.join(LOCAL_DATA_DIR, 'metadata.csv')) or not os.path.exists(_OLD_DATA_DIR):
        return LOCAL_DATA_DIR
    return _OLD_DATA_DIR


BASE_DIR = _default_base_dir()
METADATA_PATH = os.path.join(BASE_DIR, 'metadata.csv')
FILES_DIR = os.path.join(BASE_DIR, 'files')

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
LEADS = {name: i for i, name in enumerate(LEAD_NAMES)}
LEAD_II = LEADS['II']
LEAD_V1 = LEADS['V1']

PRE_R = 0.4             # cửa sổ cắt nhịp: 400 ms trước R ...
POST_R = 0.6            # ... và 600 ms sau R
TARGET_FS = 1000        # nội suy nhịp trung bình lên 1000 Hz (1 ms/mẫu)
MIN_CORR = 0.9          # nhịp có tương quan QRS < 0.9 với nhịp mẫu bị loại (ngoại tâm thu, nhiễu)
QRS_THRESHOLD = 0.08    # ngưỡng (tỉ lệ so với đỉnh) của tốc độ biến thiên không gian để xác định QRS
MIN_BEATS = 6           # tiêu chí chất lượng: ít nhất 6 nhịp hợp lệ
MAX_RR_CV = 0.15        # tiêu chí chất lượng: nhịp đều (hệ số biến thiên RR <= 0.15)
MIN_USED_FRAC = 0.7     # tiêu chí chất lượng: >= 70% số nhịp giống nhịp mẫu (ít nhiễu/nhịp lạc)

CONTINUOUS_FEATURES = [
    'V1_J', 'V1_Slope', 'V1_T_Amp', 'V2_J', 'V2_Slope', 'V2_T_Amp',
    'QRS', 'aVR_R', 'S_I_Amp', 'S_I_Dur', 'fQRS_Count', 'TpTe', 'QTc', 'HR',
]
BINARY_FEATURES = ['Type1_Like', 'aVR_Sign', 'S_I_Sign', 'fQRS', 'ER']
QC_COLUMNS = ['RR_CV', 'n_beats', 'n_used', 'quality_ok']
FEATURE_COLUMNS = CONTINUOUS_FEATURES + BINARY_FEATURES + QC_COLUMNS


def ms(t_ms, fs):
    """Đổi mili-giây sang số mẫu."""
    return int(round(t_ms * fs / 1000))


# ==========================================
# TIỀN XỬ LÝ
# ==========================================

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


def median_beat(signal, peaks, fs):
    """Nhịp trung vị 12 chuyển đạo sau khi loại nhịp lạc.

    Mỗi nhịp được hiệu chỉnh đường đẳng điện theo đoạn PR (100 -> 50 ms trước R),
    so với nhịp mẫu trong vùng QRS; nhịp có tương quan < MIN_CORR bị loại.
    Trả về (beat, chỉ số R, số nhịp cắt được, số nhịp được dùng) hoặc None.
    """
    pre, post = int(PRE_R * fs), int(POST_R * fs)
    b0, b1 = pre - ms(100, fs), pre - ms(50, fs)
    segs = []
    for r in peaks:
        start, end = r - pre, r + post
        if start >= 0 and end <= len(signal):
            seg = signal[start:end]
            segs.append(seg - seg[b0:b1].mean(axis=0))
    if not segs:
        return None
    segs = np.stack(segs)

    template = np.median(segs, axis=0)
    w = slice(pre - ms(100, fs), pre + ms(150, fs))
    ref = template[w].ravel()
    corr = np.array([np.corrcoef(s[w].ravel(), ref)[0, 1] for s in segs])
    keep = corr >= MIN_CORR
    if keep.sum() < 3:  # quá ít nhịp đạt: giữ 3 nhịp giống mẫu nhất
        keep = corr >= np.sort(corr)[-min(3, len(corr))]
    return np.median(segs[keep], axis=0), pre, len(segs), int(keep.sum())


def upsample(beat, fs):
    """Nội suy giới hạn băng tần lên TARGET_FS (hợp lệ vì tín hiệu đã lọc < 40 Hz)."""
    up = max(1, int(round(TARGET_FS / fs)))
    return resample_poly(beat, up, 1, axis=0), fs * up, up


# ==========================================
# CÁC MỐC TRÊN NHỊP TRUNG BÌNH
# ==========================================

def qrs_bounds(beat, r_idx, fs):
    """Điểm bắt đầu QRS và điểm J (kết thúc QRS) chung cho 12 chuyển đạo.

    Dùng tốc độ biến thiên không gian (tổng |đạo hàm| 12 chuyển đạo). Mở rộng từ
    điểm biến thiên lớn nhất sang hai phía khi còn trên ngưỡng, cho phép hụt ngưỡng
    tối đa 10 ms (vd tại đáy sóng S), để không lan sang sóng P/T.
    """
    vel = np.abs(np.diff(beat, axis=0)).sum(axis=1)
    lo = max(r_idx - ms(120, fs), 0)
    hi = min(r_idx + ms(160, fs), len(vel))
    peak = lo + int(np.argmax(vel[lo:hi]))
    above = vel >= QRS_THRESHOLD * vel[peak]
    gap = ms(10, fs)

    i = last = peak
    while i > lo:
        i -= 1
        if above[i]:
            last = i
        elif last - i > gap:
            break
    onset = last

    i = last = peak
    while i < hi - 1:
        i += 1
        if above[i]:
            last = i
        elif i - last > gap:
            break
    return onset, last + 1


def t_wave(x, start, end, fs):
    """Đỉnh T và điểm kết thúc T (phương pháp tiếp tuyến) trên một chuyển đạo.

    Trả về (t_peak, t_end) theo chỉ số mẫu; NaN nếu sóng T quá dẹt (< 0.1 mV).
    """
    start, end = int(start), int(min(end, len(x) - 1))
    if end - start < ms(60, fs):
        return np.nan, np.nan
    tp = start + int(np.argmax(np.abs(x[start:end])))
    if abs(x[tp]) < 0.1:
        return np.nan, np.nan
    sgn = np.sign(x[tp])
    d = np.diff(x[tp:end + 1])
    k = int(np.argmax(-sgn * d))  # điểm dốc nhất trên nhánh xuống
    slope = d[k]
    if slope * sgn >= 0:
        return tp, np.nan
    t = tp + k
    # Tiếp tuyến cắt mức nền sau sóng T (không giả định nền đúng bằng 0)
    after = x[t:end + 1]
    base = np.percentile(after, 25 if sgn > 0 else 75)
    te = t + (base - x[t]) / slope
    if not tp < te < end - ms(5, fs):  # tiếp tuyến không cắt nền trong cửa sổ: không đo được
        return tp, np.nan
    return tp, float(te)


def s_wave_lead_i(x, onset, offset, fs):
    """Biên độ (mV) và thời gian (ms) sóng S ở DI."""
    r = onset + int(np.argmax(x[onset:offset]))
    if offset - r < 2:
        return 0.0, 0.0
    s = r + int(np.argmin(x[r:offset]))
    if x[s] >= 0:
        return 0.0, 0.0
    a = s
    while a > r and x[a - 1] < 0:
        a -= 1
    b = s
    while b < offset - 1 and x[b + 1] < 0:
        b += 1
    return float(-x[s]), (b - a + 1) * 1000 / fs


def count_qrs_spikes(x, onset, offset, fs, prominence=0.05):
    """Số đỉnh + đáy trong QRS (phức bộ rS bình thường có 2)."""
    seg = x[max(onset - ms(10, fs), 0):offset + ms(10, fs)]
    n_peaks = len(find_peaks(seg, prominence=prominence)[0])
    n_troughs = len(find_peaks(-seg, prominence=prominence)[0])
    return n_peaks + n_troughs


# ==========================================
# TRÍCH ĐẶC TRƯNG
# ==========================================

def analyze_record(pid):
    """Chạy toàn bộ pipeline cho một bệnh nhân.

    Trả về dict chứa nhịp trung bình đã nội suy và các mốc (dùng cho vẽ hình
    và trích đặc trưng), hoặc None nếu không có nhịp hợp lệ.
    """
    signal, fs = load_clean_record(pid)
    peaks = detect_r_peaks(signal[:, LEAD_II], fs)
    res = median_beat(signal, peaks, fs)
    if res is None or len(peaks) < 2:
        return None
    beat, r_idx, n_beats, n_used = res
    beat, fs_up, up = upsample(beat, fs)
    r_idx *= up

    rr = np.diff(peaks) / fs  # giây
    rr_med = float(np.median(rr))
    onset, j = qrs_bounds(beat, r_idx, fs_up)
    # Đo mọi biên độ so với mức đẳng điện ngay trước QRS (đúng cả khi QRS rộng, bắt đầu sớm)
    beat = beat - beat[max(onset - ms(20, fs_up), 0):max(onset - ms(5, fs_up), 1)].mean(axis=0)
    # Sóng T phải kết thúc trước sóng P của nhịp sau và trước mép cửa sổ
    t_limit = min(len(beat) - 1 - ms(20, fs_up), r_idx + ms(rr_med * 1000 - 200, fs_up))
    t_start = j + ms(80, fs_up)
    return {
        'beat': beat, 'fs': fs_up, 'r': r_idx, 'onset': onset, 'j': j,
        't_II': t_wave(beat[:, LEADS['II']], t_start, t_limit, fs_up),
        't_V5': t_wave(beat[:, LEADS['V5']], t_start, t_limit, fs_up),
        't_limit': t_limit, 'rr': rr, 'rr_med': rr_med,
        'n_beats': n_beats, 'n_used': n_used,
    }


def _st_features(x, j, r, t_limit, fs):
    st20, st80 = j + ms(20, fs), j + ms(80, fs)
    j_amp = x[j]
    slope = (x[st80] - j_amp) / 0.08
    t_min = np.min(x[st80:r + ms(450, fs)])
    # Dạng type 1 (coved): ST tại J+20ms >= 0.2 mV, ST đi xuống, sóng T âm rõ (<= -0.1 mV)
    # và từ J+80ms đi thẳng xuống đáy T, không có sóng T dương xen giữa.
    if t_limit <= st80 + 1:  # nhịp quá nhanh, không đủ chỗ cho sóng T
        return j_amp, slope, t_min, False
    t_neg = st80 + int(np.argmin(x[st80:t_limit]))
    t_hump = np.max(x[st80:t_neg + 1]) - x[st80]
    coved = x[st20] >= 0.2 and x[st80] < x[st20] and x[t_neg] <= -0.1 and t_hump < 0.05
    return j_amp, slope, t_min, coved


def features_from_analysis(a):
    """Tính các đặc trưng từ kết quả analyze_record."""
    beat, fs, r, onset, j = a['beat'], a['fs'], a['r'], a['onset'], a['j']
    lead = lambda name: beat[:, LEADS[name]]
    f = {}

    # 1-6. ST-T ở V1, V2: chênh J (mV), độ dốc ST J->J+80ms (mV/s), giá trị thấp nhất vùng ST-T (mV)
    type1 = False
    for name in ['V1', 'V2']:
        j_amp, slope, t_min, coved = _st_features(lead(name), j, r, a['t_limit'], fs)
        f[f'{name}_J'], f[f'{name}_Slope'], f[f'{name}_T_Amp'] = j_amp, slope, t_min
        type1 |= bool(coved)
    f['Type1_Like'] = int(type1)

    # 7. Độ rộng QRS toàn bộ (ms)
    f['QRS'] = (j - onset) * 1000 / fs

    # 8. Dấu aVR: sóng R cuối (nửa sau QRS) ở aVR >= 0.3 mV
    mid = (onset + j) // 2
    f['aVR_R'] = max(0.0, float(np.max(lead('aVR')[mid:j])))
    f['aVR_Sign'] = int(f['aVR_R'] >= 0.3)

    # 9. Sóng S ở DI: >= 0.1 mV và/hoặc >= 40 ms
    f['S_I_Amp'], f['S_I_Dur'] = s_wave_lead_i(lead('I'), onset, j, fs)
    f['S_I_Sign'] = int(f['S_I_Amp'] >= 0.1 or f['S_I_Dur'] >= 40)

    # 10. QRS phân mảnh ở V1-V3: >= 4 đỉnh/đáy trong một chuyển đạo
    f['fQRS_Count'] = max(count_qrs_spikes(lead(n), onset, j, fs) for n in ['V1', 'V2', 'V3'])
    f['fQRS'] = int(f['fQRS_Count'] >= 4)

    # 11. Tpeak-Tend ở V5 (ms)
    tp, te = a['t_V5']
    f['TpTe'] = (te - tp) * 1000 / fs

    # 12. QTc Bazett (ms), QT đo ở chuyển đạo có sóng T rõ hơn giữa DII và V5
    valid = [(abs(beat[int(tp_), LEADS[n]]), te_) for n, (tp_, te_) in
             [('II', a['t_II']), ('V5', a['t_V5'])] if not np.isnan(te_)]
    te_qt = max(valid)[1] if valid else np.nan
    f['QTc'] = (te_qt - onset) / fs / np.sqrt(a['rr_med']) * 1000

    # 13. Tái cực sớm: J >= 0.1 mV ở >= 2 chuyển đạo dưới hoặc >= 2 chuyển đạo bên
    j_amps = beat[j]
    inferior = sum(j_amps[LEADS[n]] >= 0.1 for n in ['II', 'III', 'aVF'])
    lateral = sum(j_amps[LEADS[n]] >= 0.1 for n in ['I', 'aVL', 'V4', 'V5', 'V6'])
    f['ER'] = int(inferior >= 2 or lateral >= 2)

    # 14. Nhịp tim và chỉ số chất lượng
    f['HR'] = 60 / a['rr_med']
    f['RR_CV'] = float(np.std(a['rr']) / np.mean(a['rr']))
    f['n_beats'], f['n_used'] = a['n_beats'], a['n_used']
    f['quality_ok'] = int(f['n_used'] >= MIN_BEATS and f['RR_CV'] <= MAX_RR_CV
                          and f['n_used'] >= MIN_USED_FRAC * f['n_beats'])

    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in f.items()}


def extract_features(pid):
    """Trích đặc trưng cho một bệnh nhân. Luôn trả về dict đủ cột (NaN nếu lỗi)."""
    result = dict.fromkeys(FEATURE_COLUMNS, np.nan)
    try:
        a = analyze_record(pid)
        if a is None:
            print(f"⚠️ ID {pid}: không tìm được nhịp hợp lệ")
            return result
        result.update(features_from_analysis(a))
    except Exception as e:
        print(f"❌ Lỗi khi xử lý ID {pid}: {e}")
    return result
