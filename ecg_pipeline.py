"""
BỘ MÁY XỬ LÝ ECG DÙNG CHUNG CHO TOÀN BỘ DỰ ÁN BRUGADA-HUCA
==========================================================
Các script 1, 3, 4, 5 đều `import ecg_pipeline` để mọi bước dùng cùng một thuật toán.

Quy trình cho mỗi bệnh nhân:
  1. Đọc bản ghi WFDB 12 chuyển đạo (100 Hz, 12 giây).
  2. Lọc thông dải Butterworth bậc 3, 0.5-40 Hz, filtfilt (pha bằng 0).
  3. Phát hiện phức bộ QRS trên NĂNG LƯỢNG 12 CHUYỂN ĐẠO (không dùng riêng V1,
     vì QRS ở V1 thường âm - dạng rS - nên tìm đỉnh dương ở V1 hay bắt nhầm sóng T).
  4. Cắt từng nhịp, căn chỉnh lại bằng tương quan chéo, loại nhịp lạc (ngoại tâm thu/nhiễu),
     lấy NHỊP TRUNG VỊ (median beat) - bền với nhiễu hơn trung bình cộng.
  5. Xác định QRS onset và điểm J trên vận tốc không gian (spatial velocity) của cả 12 chuyển đạo
     -> một điểm J chung cho mọi chuyển đạo, giống cách máy ECG thương mại làm.
  6. Đo đặc trưng theo đơn vị thời gian thật (ms) qua `fs`, không viết cứng chỉ số mẫu.

Giới hạn cần nhớ: ở 100 Hz mỗi mẫu = 10 ms, nên mọi phép đo thời gian có độ phân giải ~10 ms.
"""
import os

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, find_peaks

# ==========================================
# 1. CẤU HÌNH
# ==========================================
# Đặt biến môi trường BRUGADA_DIR để trỏ tới thư mục dữ liệu (chứa metadata.csv và files/).
# Nếu không đặt, dùng đường dẫn mặc định trên máy Windows của bạn.
DEFAULT_BASE_DIR = r'D:\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0'
BASE_DIR = os.environ.get('BRUGADA_DIR', DEFAULT_BASE_DIR)
METADATA_PATH = os.path.join(BASE_DIR, 'metadata.csv')
FILES_DIR = os.path.join(BASE_DIR, 'files')

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
LEAD_IDX = {name: i for i, name in enumerate(LEADS)}

# Tham số thuật toán (đơn vị giây / mV) - đổi ở đây, không sửa rải rác trong code
FILTER_BAND = (0.5, 40.0)   # Hz. AHA 2007: chặn thấp <= 0.67 Hz chấp nhận được khi lọc pha bằng 0
BEAT_PRE = 0.30             # s trước điểm mốc QRS
BEAT_POST = 0.60            # s sau điểm mốc QRS
REFRACTORY = 0.25           # s - khoảng cách tối thiểu giữa 2 QRS (cho phép nhịp tới 240 l/p)
MIN_BEAT_CORR = 0.80        # nhịp có tương quan với nhịp trung vị thấp hơn ngưỡng này bị loại
VELOCITY_THRESHOLD = 0.15   # tỷ lệ so với vận tốc cực đại để xác định onset / điểm J
ST_OFFSET = 0.08            # s - đo ST tại J + 80 ms

FEATURE_COLUMNS = [
    'HR_bpm', 'RR_sd_ms', 'QRS_ms',
    'V1_J', 'V1_ST80', 'V1_Slope', 'V1_T',
    'V2_J', 'V2_ST80', 'V2_Slope', 'V2_T',
    'I_S_amp', 'I_S_ms', 'aVR_R_amp',
]
QUALITY_COLUMNS = ['n_beats', 'beat_corr']

FEATURE_DESCRIPTIONS = {
    'HR_bpm': 'Tần số tim (nhịp/phút)',
    'RR_sd_ms': 'Độ lệch chuẩn khoảng RR (ms)',
    'QRS_ms': 'Thời gian QRS toàn cục, onset -> J (ms)',
    'V1_J': 'Biên độ tại điểm J ở V1 (mV)',
    'V1_ST80': 'Biên độ ST tại J+80 ms ở V1 (mV)',
    'V1_Slope': 'Độ dốc ST từ J đến J+80 ms ở V1 (mV/s)',
    'V1_T': 'Biên độ sóng T có dấu ở V1 (mV)',
    'V2_J': 'Biên độ tại điểm J ở V2 (mV)',
    'V2_ST80': 'Biên độ ST tại J+80 ms ở V2 (mV)',
    'V2_Slope': 'Độ dốc ST từ J đến J+80 ms ở V2 (mV/s)',
    'V2_T': 'Biên độ sóng T có dấu ở V2 (mV)',
    'I_S_amp': 'Độ sâu sóng S ở DI (mV, số dương)',
    'I_S_ms': 'Thời gian sóng S ở DI (ms)',
    'aVR_R_amp': 'Biên độ sóng R cuối ở aVR (mV)',
}


# ==========================================
# 2. ĐỌC VÀ LỌC TÍN HIỆU
# ==========================================
def load_record(pid, files_dir=None):
    """Đọc bản ghi 12 chuyển đạo. Trả về (mảng [n_mẫu, 12] đơn vị mV, fs)."""
    path = os.path.join(files_dir or FILES_DIR, str(pid), str(pid))
    record = wfdb.rdrecord(path)
    if list(record.sig_name) != LEADS:
        raise ValueError(f'ID {pid}: thứ tự chuyển đạo lạ {record.sig_name}')
    return record.p_signal, record.fs


def bandpass(signal, fs, band=FILTER_BAND, order=3):
    """Butterworth bậc 3, filtfilt (pha bằng 0) theo từng cột/chuyển đạo."""
    nyq = 0.5 * fs
    b, a = butter(order, [band[0] / nyq, band[1] / nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)


# ==========================================
# 3. PHÁT HIỆN QRS VÀ TẠO NHỊP TRUNG VỊ
# ==========================================
def detect_qrs(sig12, fs):
    """Tìm điểm mốc QRS từ năng lượng đạo hàm cộng trên 12 chuyển đạo."""
    energy = np.sum(np.diff(sig12, axis=0) ** 2, axis=1)
    win = max(1, int(round(0.08 * fs)))
    energy = np.convolve(energy, np.ones(win) / win, mode='same')
    height = 0.3 * np.percentile(energy, 99)
    peaks, _ = find_peaks(energy, distance=int(REFRACTORY * fs), height=height)
    return peaks


def _segment(sig12, centers, pre, post):
    return np.array([sig12[c - pre:c + post] for c in centers])


def median_beat(sig12, fs, qrs=None):
    """
    Tạo nhịp trung vị 12 chuyển đạo.
    Trả về dict: beat [n_mẫu_cửa_sổ, 12], fid (vị trí điểm mốc trong cửa sổ), fs, rr (giây),
    n_beats, beat_corr - hoặc None nếu không đủ nhịp tốt.
    """
    if qrs is None:
        qrs = detect_qrs(sig12, fs)
    pre, post = int(round(BEAT_PRE * fs)), int(round(BEAT_POST * fs))
    shift_max = int(round(0.03 * fs))
    lo, hi = pre + shift_max, len(sig12) - post - shift_max
    centers = np.array([c for c in qrs if lo <= c < hi])
    if len(centers) < 3:
        return None

    # Chỉ so sánh vùng lõi QRS-ST-T (-100..+350 ms) để nhịp kế tiếp lọt vào cửa sổ
    # (khi tim nhanh) không làm hỏng việc căn chỉnh và loại nhịp
    core_pre, core_post = int(round(0.10 * fs)), int(round(0.35 * fs))

    # Căn chỉnh lại từng nhịp theo tương quan chéo với nhịp trung vị ban đầu (±30 ms)
    template = np.median(_segment(sig12, centers, core_pre, core_post), axis=0)
    aligned = []
    for c in centers:
        scores = [np.sum(sig12[c + s - core_pre:c + s + core_post] * template)
                  for s in range(-shift_max, shift_max + 1)]
        aligned.append(c + int(np.argmax(scores)) - shift_max)
    centers = np.array(aligned)
    beats = _segment(sig12, centers, pre, post)

    # Loại nhịp lạc: tương quan (12 chuyển đạo ghép nối, vùng lõi) với nhịp trung vị
    cores = beats[:, pre - core_pre:pre + core_post].reshape(len(beats), -1)
    template = np.median(cores, axis=0)
    corrs = np.array([np.corrcoef(b, template)[0, 1] for b in cores])
    good = corrs >= MIN_BEAT_CORR
    if good.sum() < 3:
        return None

    return {
        'beat': np.median(beats[good], axis=0),
        'fid': pre,
        'fs': fs,
        'rr': np.diff(qrs) / fs,  # RR lấy từ MỌI QRS phát hiện được, kể cả nhịp bị loại
        'n_beats': int(good.sum()),
        'beat_corr': float(np.mean(corrs[good])),
    }


# ==========================================
# 4. XÁC ĐỊNH QRS ONSET VÀ ĐIỂM J
# ==========================================
def delineate(beat, fid, fs):
    """
    Xác định QRS onset và điểm J trên vận tốc không gian của nhịp trung vị.
    v[i] = độ lớn thay đổi giữa mẫu i và i+1 (gộp 12 chuyển đạo).
    Onset/J = chỗ v giảm dưới VELOCITY_THRESHOLD * v_max và giữ phẳng 2 mẫu liền.
    """
    v = np.sqrt(np.sum(np.diff(beat, axis=0) ** 2, axis=1))
    search = int(round(0.06 * fs))
    v_max = v[max(0, fid - search):fid + search].max()
    th = VELOCITY_THRESHOLD * v_max
    flat = (v < th) & (np.roll(v, 1) < th)  # phẳng ở cả 2 khoảng liên tiếp

    # Tìm onset tối đa 120 ms trước mốc, điểm J tối đa 180 ms sau mốc (QRS rộng kiểu block nhánh phải)
    min_gap, max_back, max_fwd = int(round(0.02 * fs)), int(round(0.12 * fs)), int(round(0.18 * fs))
    onset = None
    for i in range(fid - min_gap, max(1, fid - max_back), -1):
        if flat[i]:
            onset = i + 1
            break
    j_point = None
    for i in range(fid + min_gap, min(len(v) - 1, fid + max_fwd)):
        if flat[i]:
            j_point = i - 1
            break
    if onset is None or j_point is None:
        return None
    return onset, j_point


# ==========================================
# 5. TRÍCH XUẤT ĐẶC TRƯNG
# ==========================================
def baseline_correct(beat, onset, fs):
    """Trừ đường đẳng điện: trung bình đoạn PR 10-30 ms trước QRS onset, từng chuyển đạo."""
    b0, b1 = onset - int(round(0.03 * fs)), onset - int(round(0.01 * fs))
    return beat - beat[b0:b1 + 1].mean(axis=0)


def _s_wave_lead_i(x, onset, j_point, fs):
    """Sóng S ở DI: phần âm (dưới đường đẳng điện) sau đỉnh R, trong QRS. Trả về (độ sâu mV, thời gian ms)."""
    qrs = x[onset:j_point + 1]
    r = int(np.argmax(qrs))
    tail = qrs[r:]
    if tail.min() >= 0:
        return 0.0, 0.0
    # Thời gian dưới 0: nội suy tuyến tính điểm cắt 0 để có độ phân giải dưới 10 ms
    t = np.arange(len(tail)) / fs
    neg = tail < 0
    first = int(np.argmax(neg))
    last = len(tail) - 1 - int(np.argmax(neg[::-1]))
    t_start = t[first] if first == 0 else np.interp(0, [tail[first], tail[first - 1]], [t[first], t[first - 1]])
    if last + 1 < len(tail):
        t_end = np.interp(0, [tail[last], tail[last + 1]], [t[last], t[last + 1]])
    else:
        t_end = t[last]
    return float(-tail.min()), float((t_end - t_start) * 1000)


def features_from_beat(beat, fid, fs):
    """Tính FEATURE_COLUMNS (trừ HR/RR) từ nhịp trung vị. Trả về dict hoặc None."""
    marks = delineate(beat, fid, fs)
    if marks is None:
        return None
    onset, j_point = marks
    beat = baseline_correct(beat, onset, fs)

    st80 = j_point + int(round(ST_OFFSET * fs))
    t_lo, t_hi = st80, min(len(beat), j_point + int(round(0.40 * fs)))
    feats = {'QRS_ms': (j_point - onset) / fs * 1000}
    for lead in ('V1', 'V2'):
        x = beat[:, LEAD_IDX[lead]]
        t_seg = x[t_lo:t_hi]
        feats[f'{lead}_J'] = x[j_point]
        feats[f'{lead}_ST80'] = x[st80]
        feats[f'{lead}_Slope'] = (x[st80] - x[j_point]) / ST_OFFSET
        feats[f'{lead}_T'] = t_seg[np.argmax(np.abs(t_seg))]

    feats['I_S_amp'], feats['I_S_ms'] = _s_wave_lead_i(beat[:, LEAD_IDX['I']], onset, j_point, fs)
    feats['aVR_R_amp'] = max(0.0, beat[onset:j_point + 1, LEAD_IDX['aVR']].max())
    feats['_onset'], feats['_j'] = onset, j_point
    return feats


def process_patient(pid, files_dir=None):
    """
    Toàn bộ quy trình cho 1 bệnh nhân.
    Trả về (features dict, nhịp trung vị dict). Nếu thất bại, features có cột 'error' ghi lý do -
    KHÔNG nuốt lỗi im lặng như `except: pass`.
    """
    empty = {c: np.nan for c in FEATURE_COLUMNS + QUALITY_COLUMNS}
    try:
        raw, fs = load_record(pid, files_dir)
        sig = bandpass(raw, fs)
        mb = median_beat(sig, fs)
        if mb is None:
            return {**empty, 'error': 'không đủ nhịp tốt'}, None
        feats = features_from_beat(mb['beat'], mb['fid'], fs)
        if feats is None:
            return {**empty, 'error': 'không xác định được onset/điểm J'}, mb
    except (OSError, ValueError) as e:
        return {**empty, 'error': str(e)}, None

    mb['onset'], mb['j'] = feats.pop('_onset'), feats.pop('_j')
    feats['HR_bpm'] = 60.0 / np.mean(mb['rr'])
    feats['RR_sd_ms'] = np.std(mb['rr']) * 1000 if len(mb['rr']) > 1 else np.nan
    feats['n_beats'], feats['beat_corr'] = mb['n_beats'], mb['beat_corr']
    out = {c: feats[c] for c in FEATURE_COLUMNS + QUALITY_COLUMNS}
    out['error'] = ''
    return out, mb
