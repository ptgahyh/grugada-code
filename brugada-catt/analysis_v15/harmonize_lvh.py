"""Bước 15b: chuẩn hóa PTB-XL và ecg-arrhythmia cho bài toán LVH (theo PLAN_vi.md mục 2).

Tín hiệu xử lý giống hệt analysis_v14/harmonize_rbbb.py (9 chuyển đạo, 100 Hz, lọc 0,5–40 Hz, nhịp trung vị 75 mẫu,
neo 28). Khác biệt duy nhất: lưu thêm nhịp trung vị theo mV **trước** z-score (M), vì mỏ neo điện thế cần biên độ tuyệt đối,
và lưu giới tính (female = 1).

Nhãn:
  PTB-XL  dương = mã SCP LVH với likelihood ≥ 50; loại LVH likelihood < 50; loại VCLVH không kèm LVH.
  cn      dương = SNOMED 164873001 (LVH); loại 55827005 (LVHV, điện thế cao thất trái) không kèm LVH; loại giới tính Unknown.

Chạy: python harmonize_lvh.py <thư mục PTB-XL> <ecg-arrhythmia-1.0.0.zip> <out>
Đầu ra: <out>/{ptb_cs,ptb_at,cn}.npz với X (N, 9, 75) z-score, M (N, 9, 75) mV, y, female, ids; <out>/harmonize_log.json
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
IDX9 = [0, 1, 2, 6, 7, 8, 9, 10, 11]
B, A = butter(2, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")
LVH, LVHV = "164873001", "55827005"


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
    """sig: (9, T) mV ở 100 Hz -> (nhịp z-score, nhịp mV) hoặc None."""
    if not np.all(np.isfinite(sig)) or np.any(sig.std(1) < 1e-6):
        return None
    sig = filtfilt(B, A, sig, axis=1)
    env = np.abs(np.diff(sig, axis=1)).sum(0)
    peaks, _ = find_peaks(env, distance=int(0.33 * FS), height=0.35 * np.percentile(env, 99))
    peaks = [p for p in peaks if p - PRE >= 0 and p + POST <= sig.shape[1]]
    if len(peaks) < 3:
        return None
    med = align(np.median(np.stack([sig[:, p - PRE:p + POST] for p in peaks]), 0))
    z = (med - med.mean(1, keepdims=True)) / (med.std(1, keepdims=True) + 1e-8)
    return z.astype(np.float32), med.astype(np.float32)


def ptb_one(args):
    rid, path = args
    rec = wfdb.rdrecord(os.path.join(PTB, path))
    assert rec.fs == 100 and rec.sig_name[6] == "V1", rec.sig_name
    return rid, median_beat(rec.p_signal[:, IDX9].T)


_Z = None


def cn_one(name):
    try:
        return _cn_one(name)
    except (IndexError, ValueError, AssertionError):
        return os.path.basename(name), -2, None, None


def _cn_one(name):
    global _Z
    if _Z is None:
        _Z = zipfile.ZipFile(ZIP)
    hea = _Z.read(name + ".hea").decode()
    lines = hea.splitlines()
    dx = next((l.split(":", 1)[1].strip() for l in lines if l.startswith("#Dx")), "")
    sex = next((l.split(":", 1)[1].strip() for l in lines if l.startswith("#Sex")), "Unknown")
    codes = set(dx.split(","))
    y = int(LVH in codes)
    if not y and LVHV in codes:
        return os.path.basename(name), -1, None, None
    if sex not in ("Male", "Female"):
        return os.path.basename(name), -3, None, None
    m = loadmat(io.BytesIO(_Z.read(name + ".mat")))["val"].astype(float)
    nsig, fs = int(lines[0].split()[1]), int(float(lines[0].split()[2]))
    sig_lines = lines[1:1 + nsig]
    gains = np.array([float(l.split()[2].split("/")[0].split("(")[0]) for l in sig_lines])
    zeros = np.array([int(l.split()[4]) for l in sig_lines])
    leads = [l.split()[-1] for l in sig_lines]
    assert leads[6] == "V1" and leads[0] == "I", leads
    sig = (m - zeros[:, None]) / gains[:, None]
    sig = resample_poly(sig[IDX9], 1, fs // FS, axis=1) if fs != FS else sig[IDX9]
    return os.path.basename(name), y, int(sex == "Female"), median_beat(sig)


def save(fn, ids, beats, y, fem):
    np.savez_compressed(fn, X=np.stack([b[0] for b in beats]), M=np.stack([b[1] for b in beats]),
                        y=np.asarray(y, np.int8), female=np.asarray(fem, np.int8), ids=np.array(ids))


if __name__ == "__main__":
    log = {}
    d = pd.read_csv(os.path.join(PTB, "ptbxl_database.csv"))
    codes = d.scp_codes.apply(ast.literal_eval)
    lik = codes.apply(lambda c: c.get("LVH", -1))
    d["y"] = (lik >= 50).astype(int)
    d["excl_lowlik"] = (lik >= 0) & (lik < 50)
    d["excl_vclvh_only"] = codes.apply(lambda c: "VCLVH" in c and "LVH" not in c)
    dev = d.device.str.strip()
    d["dom"] = np.where(dev.str.startswith("CS"), "ptb_cs", np.where(dev.str.startswith("AT"), "ptb_at", "other"))
    log["ptb_total"] = len(d)
    log["ptb_excluded_other_device"] = int((d.dom == "other").sum())
    d = d[d.dom != "other"]
    log["ptb_excluded_lvh_likelihood_lt50"] = int(d.excl_lowlik.sum())
    log["ptb_excluded_vclvh_only"] = int(d.excl_vclvh_only.sum())
    d = d[~d.excl_lowlik & ~d.excl_vclvh_only]
    with Pool(4) as p:
        beats = dict(p.map(ptb_one, list(zip(d.ecg_id, d.filename_lr)), chunksize=64))
    for dom in ("ptb_cs", "ptb_at"):
        s = d[d.dom == dom].set_index("ecg_id")
        ok = [i for i in s.index if beats[i] is not None]
        save(os.path.join(OUT, f"{dom}.npz"), ok, [beats[i] for i in ok], s.loc[ok, "y"], s.loc[ok, "sex"])
        log[dom] = dict(n=len(ok), n_pos=int(s.loc[ok, "y"].sum()), excluded_few_beats=int(len(s) - len(ok)))
        print(dom, log[dom], flush=True)
    with zipfile.ZipFile(ZIP) as z:
        names = sorted(n[:-4] for n in z.namelist() if n.endswith(".hea") and "WFDBRecords" in n)
    with Pool(4) as p:
        res = p.map(cn_one, names, chunksize=128)
    log["cn_headers"] = len(names)
    log["cn_excluded_lvhv_only"] = sum(r[1] == -1 for r in res)
    log["cn_excluded_malformed_header"] = sum(r[1] == -2 for r in res)
    log["cn_excluded_sex_unknown"] = sum(r[1] == -3 for r in res)
    ok = [r for r in res if r[1] >= 0 and r[3] is not None]
    log["cn"] = dict(n=len(ok), n_pos=int(sum(r[1] for r in ok)),
                     excluded_few_beats=sum(r[1] >= 0 and r[3] is None for r in res))
    save(os.path.join(OUT, "cn.npz"), [r[0] for r in ok], [r[3] for r in ok], [r[1] for r in ok], [r[2] for r in ok])
    print("cn", log["cn"], flush=True)
    json.dump(log, open(os.path.join(OUT, "harmonize_log.json"), "w"), indent=1)
