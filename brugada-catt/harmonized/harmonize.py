"""Đưa 3 miền dữ liệu Brugada về cùng một định dạng.

Định dạng chung (cho cả HUCA, Melo-Claris, Melo-Mortara):
  - 9 chuyển đạo: I, II, III, V1..V6
  - 1 nhịp trung vị / bệnh nhân, 100 Hz, 75 mẫu (~-290 ms .. +460 ms quanh QRS)
  - căn chỉnh: đỉnh năng lượng QRS (tổng |đạo hàm| trên 9 chuyển đạo) đặt tại mẫu 28
  - z-score theo từng chuyển đạo (giống Melo, vì Melo đã mất biên độ mV)

Lý do chọn 100 Hz: HUCA chỉ có 100 Hz. Nếu giữ Melo ở 200 Hz thì tần số lấy mẫu
sẽ trở thành một "shortcut" để phân biệt miền.

Chạy:
  python harmonize.py <thư mục HUCA chứa metadata.csv> <thư mục melo_processed> <thư mục output>
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, resample_poly

HUCA_DIR, MELO_DIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)

FS = 100
PRE, POST = 29, 46          # 29 mẫu trước + 46 mẫu sau = 75
ANCHOR = 28                 # vị trí đỉnh năng lượng QRS sau căn chỉnh
LEADS = ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6"]
HUCA_IDX = [0, 1, 2, 6, 7, 8, 9, 10, 11]   # vị trí 9 chuyển đạo trong file 12 kênh của HUCA

# Bản ghi HUCA bị loại (xem phần kiểm tra chất lượng):
#  - bản trùng tuyệt đối: giữ bản đầu, bỏ bản sau
#  - có đoạn tín hiệu đứng yên >= 2 giây
HUCA_DUP_DROP = [844915, 1220716, 1366050, 1307673]
HUCA_FLAT_DROP = [267630, 903802, 1230482]


def qrs_energy(beat):
    """beat: (9, T) -> chỉ số đỉnh năng lượng QRS."""
    e = np.abs(np.diff(beat, axis=1)).sum(0)
    return int(np.argmax(e))


def align(beat, anchor=ANCHOR):
    """Dịch nhịp sao cho đỉnh năng lượng QRS nằm ở `anchor` (đệm bằng giá trị biên)."""
    shift = anchor - qrs_energy(beat)
    if shift == 0:
        return beat
    out = np.empty_like(beat)
    if shift > 0:
        out[:, shift:] = beat[:, :-shift]
        out[:, :shift] = beat[:, :1]
    else:
        out[:, :shift] = beat[:, -shift:]
        out[:, shift:] = beat[:, -1:]
    return out


def zscore(beat):
    mu = beat.mean(1, keepdims=True)
    sd = beat.std(1, keepdims=True) + 1e-8
    return (beat - mu) / sd


# ---------------------------------------------------------------- HUCA
b, a = butter(2, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")
meta = pd.read_csv(os.path.join(HUCA_DIR, "metadata.csv"))
X_h, y_h, ids_h, nbeats_h = [], [], [], []
for pid, lab in zip(meta.patient_id, meta.brugada):
    if pid in HUCA_DUP_DROP or pid in HUCA_FLAT_DROP:
        continue
    raw = np.fromfile(os.path.join(HUCA_DIR, "files", str(pid), f"{pid}.dat"), dtype="<i2")
    sig = raw.reshape(-1, 12)[:, HUCA_IDX].astype(float).T / 1000.0     # (9, 1200) mV
    sig = filtfilt(b, a, sig, axis=1)
    env = np.abs(np.diff(sig, axis=1)).sum(0)
    peaks, _ = find_peaks(env, distance=int(0.33 * FS), height=0.35 * np.percentile(env, 99))
    peaks = [p for p in peaks if p - PRE >= 0 and p + POST <= sig.shape[1]]
    if len(peaks) < 3:
        print("bỏ qua (ít nhịp):", pid)
        continue
    beats = np.stack([sig[:, p - PRE:p + POST] for p in peaks])
    med = np.median(beats, axis=0)
    X_h.append(zscore(align(med)))
    y_h.append(int(lab > 0))                     # brugada 1 và 2 đều tính là Brugada
    ids_h.append(int(pid))
    nbeats_h.append(len(peaks))
X_h = np.stack(X_h).astype(np.float32)
np.savez_compressed(os.path.join(OUT, "huca.npz"), X=X_h, y=np.array(y_h, np.int8),
                    ids=np.array(ids_h), nbeats=np.array(nbeats_h), leads=np.array(LEADS), fs=FS)
print("HUCA:", X_h.shape, "Brugada =", int(np.sum(y_h)))

# ---------------------------------------------------------------- Melo (200 Hz -> 100 Hz)
for name in ["claris", "mortara"]:
    d = np.load(os.path.join(MELO_DIR, f"melo_{name}.npz"))
    X = d["X"][:, :9].astype(float)              # 9 chuyển đạo đầu: I, II, III, V1..V6
    X = resample_poly(X, 1, 2, axis=2)           # có lọc chống chồng phổ, 150 -> 75 mẫu
    X = np.stack([zscore(align(x)) for x in X]).astype(np.float32)
    np.savez_compressed(os.path.join(OUT, f"{name}.npz"), X=X, y=d["y"].astype(np.int8),
                        ids=np.arange(len(X)), leads=np.array(LEADS), fs=FS)
    print(name, X.shape, "Brugada =", int(d["y"].sum()))
