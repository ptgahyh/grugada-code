"""Bước 14: chuẩn hóa PTB-XL và ecg-arrhythmia (Chapman/Ningbo) về định dạng của harmonized/harmonize.py.

9 chuyển đạo (I, II, III, V1–V6), 100 Hz, lọc Butterworth bậc 2 0,5–40 Hz hai chiều, đỉnh năng lượng QRS
(khoảng cách 330 ms, ngưỡng 35% phân vị 99), nhịp trung vị 75 mẫu (29 trước + 46 sau), neo mẫu 28, z-score từng chuyển đạo.
Miền: ptb_cs (máy CS*), ptb_at (máy AT*), cn (ecg-arrhythmia, hạ 500 → 100 Hz bằng resample_poly(1, 5)).
Nhãn: dương = CRBBB (PTB-XL) / SNOMED 713427006 (CRBBB) hoặc 59118001 (RBBB) (cn).
Loại bản ghi chỉ có block nhánh phải không hoàn toàn: PTB-XL IRBBB, cn 713426002 (xem PLAN_vi.md, mục 6).

Chạy: python harmonize_rbbb.py <thư mục PTB-XL (ptbxl_database.csv, records100/)> <ecg-arrhythmia-1.0.0.zip> <out>
Đầu ra: <out>/{ptb_cs,ptb_at,cn}.npz với X (N, 9, 75) float32, y, ids; <out>/harmonize_log.json
"""
import ast
import io
import json
import os
import sys
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
import wfdb
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks, resample_poly

PTB, ZIP, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
FS, PRE, POST, ANCHOR = 100, 29, 46, 28
IDX9 = [0, 1, 2, 6, 7, 8, 9, 10, 11]          # I, II, III, V1..V6 trong thứ tự 12 chuyển đạo chuẩn
B, A = butter(2, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")
RBBB_POS = {"713427006", "59118001"}    # CRBBB, RBBB (không ghi rõ)
IRBBB = "713426002"


def align(beat):
    e = np.abs(np.diff(beat, axis=1)).sum(0)
    shift = ANCHOR - int(np.argmax(e))
    if shift == 0:
        return beat
    out = np.empty_like(beat)
    if shift > 0:
        out[:, shift:] = beat[:, :-shift]; out[:, :shift] = beat[:, :1]
    else:
        out[:, :shift] = beat[:, -shift:]; out[:, shift:] = beat[:, -1:]
    return out


def median_beat(sig):
    """sig: (9, T) mV ở 100 Hz -> nhịp trung vị đã căn và z-score, hoặc None."""
    if not np.all(np.isfinite(sig)) or np.any(sig.std(1) < 1e-6):
        return None
    sig = filtfilt(B, A, sig, axis=1)
    env = np.abs(np.diff(sig, axis=1)).sum(0)
    peaks, _ = find_peaks(env, distance=int(0.33 * FS), height=0.35 * np.percentile(env, 99))
    peaks = [p for p in peaks if p - PRE >= 0 and p + POST <= sig.shape[1]]
    if len(peaks) < 3:
        return None
    med = align(np.median(np.stack([sig[:, p - PRE:p + POST] for p in peaks]), 0))
    return ((med - med.mean(1, keepdims=True)) / (med.std(1, keepdims=True) + 1e-8)).astype(np.float32)


def ptb_one(args):
    rid, path = args
    rec = wfdb.rdrecord(os.path.join(PTB, path))
    assert rec.fs == 100 and rec.sig_name[6] == "V1", rec.sig_name
    return rid, median_beat(rec.p_signal[:, IDX9].T)


_Z = None


def cn_one(name):
    global _Z
    if _Z is None:                       # mỗi tiến trình con mở zip một lần
        _Z = zipfile.ZipFile(ZIP)
    hea = _Z.read(name + ".hea").decode()
    m = loadmat(io.BytesIO(_Z.read(name + ".mat")))["val"].astype(float)
    lines = hea.splitlines()
    nsig, fs = int(lines[0].split()[1]), int(float(lines[0].split()[2]))
    sig_lines = lines[1:1 + nsig]
    gains = np.array([float(l.split()[2].split("/")[0].split("(")[0]) for l in sig_lines])
    zeros = np.array([int(l.split()[4]) for l in sig_lines])
    leads = [l.split()[-1] for l in sig_lines]
    assert leads[6] == "V1" and leads[0] == "I", leads
    dx = next((l.split(":", 1)[1].strip() for l in lines if l.startswith("#Dx")), "")
    sig = (m - zeros[:, None]) / gains[:, None]
    sig = resample_poly(sig[IDX9], 1, fs // FS, axis=1) if fs != FS else sig[IDX9]
    codes = set(dx.split(","))
    y = int(bool(codes & RBBB_POS))
    if not y and IRBBB in codes:          # chỉ có block nhánh phải không hoàn toàn: loại
        return os.path.basename(name), -1, None
    return os.path.basename(name), y, median_beat(sig)


if __name__ == "__main__":
    log = {}
    # ---------------------------------------------------------------- PTB-XL
    d = pd.read_csv(os.path.join(PTB, "ptbxl_database.csv"))
    codes = d.scp_codes.apply(ast.literal_eval)
    d["y"] = codes.apply(lambda c: int("CRBBB" in c))
    d["irbbb_only"] = codes.apply(lambda c: "IRBBB" in c and "CRBBB" not in c)
    dev = d.device.str.strip()
    d["dom"] = np.where(dev.str.startswith("CS"), "ptb_cs", np.where(dev.str.startswith("AT"), "ptb_at", "other"))
    log["ptb_total"] = len(d)
    log["ptb_excluded_other_device"] = int((d.dom == "other").sum())
    log["ptb_excluded_irbbb_only"] = int(((d.dom != "other") & d.irbbb_only).sum())
    d = d[(d.dom != "other") & ~d.irbbb_only]
    with Pool(4) as p:
        beats = dict(p.map(ptb_one, list(zip(d.ecg_id, d.filename_lr)), chunksize=64))
    for dom in ("ptb_cs", "ptb_at"):
        s = d[d.dom == dom]
        ok = [i for i in s.ecg_id if beats[i] is not None]
        yy = s.set_index("ecg_id").loc[ok, "y"].to_numpy()
        np.savez_compressed(os.path.join(OUT, f"{dom}.npz"), X=np.stack([beats[i] for i in ok]), y=yy.astype(np.int8),
                            ids=np.array(ok))
        log[dom] = dict(n=len(ok), n_pos=int(yy.sum()), excluded_few_beats=int(len(s) - len(ok)))
        print(dom, log[dom], flush=True)
    # ---------------------------------------------------------------- ecg-arrhythmia
    with zipfile.ZipFile(ZIP) as z:
        names = sorted(n[:-4] for n in z.namelist() if n.endswith(".hea") and "WFDBRecords" in n)
    with Pool(4) as p:
        res = p.map(cn_one, names, chunksize=128)
    log["cn_excluded_irbbb_only"] = sum(r[1] == -1 for r in res)
    ok = [r for r in res if r[2] is not None]
    yy = np.array([r[1] for r in ok], np.int8)
    np.savez_compressed(os.path.join(OUT, "cn.npz"), X=np.stack([r[2] for r in ok]), y=yy, ids=np.array([r[0] for r in ok]))
    log["cn"] = dict(n_headers=len(names), n=len(ok), n_pos=int(yy.sum()), excluded_few_beats=len(names) - len(ok) - log["cn_excluded_irbbb_only"])
    print("cn", log["cn"], flush=True)
    json.dump(log, open(os.path.join(OUT, "harmonize_log.json"), "w"), indent=1)
