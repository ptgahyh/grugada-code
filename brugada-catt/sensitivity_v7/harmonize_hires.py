"""(Bước 7) Phân tích độ nhạy tiền xử lý: tính lại nhịp trung vị HUCA theo kiểu "máy đo".

Quy trình gốc (harmonized/harmonize.py): lọc 0.5–40 Hz ở 100 Hz, tìm đỉnh QRS, cắt nhịp theo đỉnh (độ phân giải 10 ms),
lấy trung vị. Nhịp trung vị của Melo thì do máy ECG tính ở tần số cao (200 Hz trở lên), căn chỉnh chính xác hơn.
Lệch căn chỉnh giữa các nhịp làm nhịp trung vị bị "nhòe" (QRS rộng hơn, đỉnh thấp hơn), có thể là một nguồn
chênh lệch HUCA ↔ Melo do chính quy trình của mình gây ra.

Quy trình thay thế (giống cách máy tính nhịp trung vị):
  1. lọc 0.5–40 Hz như cũ, rồi tăng lên 500 Hz (resample_poly ×5)
  2. tìm đỉnh năng lượng QRS ở 500 Hz
  3. căn từng nhịp bằng tương quan chéo với khuôn mẫu (trung vị sơ bộ), ±20 ms, bước 2 ms; lặp 2 lần
  4. trung vị ở 500 Hz, hạ về 100 Hz (resample_poly 1/5, có lọc chống chồng phổ)
  5. căn đỉnh năng lượng về ANCHOR và z-score như cũ
Chạy: python harmonize_hires.py <thư mục HUCA (metadata.csv, files/)> <harmonized gốc> <output>
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, resample_poly

HUCA_DIR, HD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FS, UP = 100, 5
FH = FS * UP
PRE, POST = 29, 46
ANCHOR = 28
HUCA_IDX = [0, 1, 2, 6, 7, 8, 9, 10, 11]
MAXLAG = 10            # ±10 mẫu ở 500 Hz = ±20 ms


def qrs_energy(beat):
    return int(np.argmax(np.abs(np.diff(beat, axis=1)).sum(0)))


def align(beat, anchor=ANCHOR):
    shift = anchor - qrs_energy(beat)
    if shift == 0:
        return beat
    out = np.empty_like(beat)
    if shift > 0:
        out[:, shift:] = beat[:, :-shift]; out[:, :shift] = beat[:, :1]
    else:
        out[:, :shift] = beat[:, -shift:]; out[:, shift:] = beat[:, -1:]
    return out


def zscore(beat):
    return (beat - beat.mean(1, keepdims=True)) / (beat.std(1, keepdims=True) + 1e-8)


ref = np.load(os.path.join(HD, "huca.npz"))
keep = set(ref["ids"].tolist())
b, a = butter(2, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")
meta = pd.read_csv(os.path.join(HUCA_DIR, "metadata.csv"))
X, y, ids, lag_sd = [], [], [], []
pre_h, post_h = PRE * UP, POST * UP
for pid, lab in zip(meta.patient_id, meta.brugada):
    if pid not in keep:
        continue
    raw = np.fromfile(os.path.join(HUCA_DIR, "files", str(pid), f"{pid}.dat"), dtype="<i2")
    sig = filtfilt(b, a, raw.reshape(-1, 12)[:, HUCA_IDX].astype(float).T / 1000.0, axis=1)
    hi = resample_poly(sig, UP, 1, axis=1)
    env = np.abs(np.diff(hi, axis=1)).sum(0)
    peaks, _ = find_peaks(env, distance=int(0.33 * FH), height=0.35 * np.percentile(env, 99))
    peaks = [p for p in peaks if p - pre_h - MAXLAG >= 0 and p + post_h + MAXLAG <= hi.shape[1]]
    beats = np.stack([hi[:, p - pre_h:p + post_h] for p in peaks])
    lags = np.zeros(len(peaks), int)
    for _ in range(2):
        tmpl = np.median(beats, axis=0)
        new = []
        for k, p in enumerate(peaks):
            best, bl = -np.inf, 0
            for L in range(-MAXLAG, MAXLAG + 1):
                seg = hi[:, p + L - pre_h:p + L + post_h]
                c = float((seg * tmpl).sum())
                if c > best:
                    best, bl = c, L
            lags[k] = bl
            new.append(hi[:, p + bl - pre_h:p + bl + post_h])
        beats = np.stack(new)
    med = np.median(beats, axis=0)
    lo = resample_poly(med, 1, UP, axis=1)[:, :PRE + POST]
    X.append(zscore(align(lo))); y.append(int(lab > 0)); ids.append(int(pid)); lag_sd.append(float(np.std(lags) * 1000 / FH))
X = np.stack(X).astype(np.float32)
order = np.argsort([list(ref["ids"]).index(i) for i in ids])
X, y, ids, lag_sd = X[order], np.array(y)[order], np.array(ids)[order], np.array(lag_sd)[order]
assert (ids == ref["ids"]).all() and (y == ref["y"]).all()
np.savez_compressed(os.path.join(OUT, "huca_hires.npz"), X=X, y=y.astype(np.int8), ids=ids, lag_sd_ms=lag_sd)
d = X - ref["X"]
print("HUCA hires:", X.shape, "| SD độ lệch căn chỉnh giữa các nhịp (ms): trung vị", round(float(np.median(lag_sd)), 2),
      "| tương quan TB với bản gốc", round(float(np.mean([np.corrcoef(X[i].ravel(), ref["X"][i].ravel())[0, 1] for i in range(len(X))])), 4),
      "| max|Δ|", round(float(np.abs(d).max()), 3))
