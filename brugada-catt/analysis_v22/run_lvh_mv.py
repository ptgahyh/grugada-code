"""Bước 22: như analysis_v15/run_lvh.py nhưng mô hình đích f (và CNNalt) nhận nhịp trung vị tính bằng mV thay vì z-score. Xem PLAN_vi.md.


Mỗi (thiết lập, seed validation):
  f       CNN 9 chuyển đạo trên nhịp z-score (cbm_lib4.train_cnn), ensemble seed 0–2 (như bước 14)
  alt     cùng kiến trúc, seed 3–5 (mỏ neo phụ thuộc A4)
  a1      Sokolow-Lyon (mV): S_V1 + max(R_V5, R_V6), không học
  a2      Cornell hiệu chỉnh giới: R_aVL + S_V3 + 0,8·nữ (mV), aVL = (I − III)/2, không học
  a3      LVH-LR: hồi quy logistic L2 (C = 1) đã chuẩn hóa trên 6 đặc trưng (features() bên dưới)
Biên độ đo trên nhịp mV so với đường đẳng điện = trung vị mẫu ANCHOR−10..ANCHOR−7 (−100..−70 ms);
R = cực đại, S = −cực tiểu (≥ 0) trong cửa sổ QRS ANCHOR−8..ANCHOR+15.
Chạy: python run_lvh.py <thư mục npz> <thư mục điểm> <chunk> <số chunk>  ->  scores_<thiết lập>_<seed>.npz
"""
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "concepts_v4"))
import cbm_lib4 as L  # noqa: E402

torch.set_num_threads(1)
DD, OUT, CH, NCH = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(OUT, exist_ok=True)
DOMS = ["ptb_cs", "ptb_at", "cn"]
DAT = {d: np.load(os.path.join(DD, f"{d}.npz")) for d in DOMS}
X = {d: DAT[d]["M"].astype(np.float32) for d in DOMS}     # BƯỚC 22: f và alt nhận nhịp mV (không z-score)
Y = {d: DAT[d]["y"].astype(int) for d in DOMS}
SETTINGS = [(a, b) for a in DOMS for b in DOMS if a != b]
CAP, ANCHOR = 12000, 28
QRS = slice(ANCHOR - 8, ANCHOR + 16)
I_, III, V1, V3, V5, V6 = 0, 2, 3, 5, 7, 8


def qrs_rms(x, lo=ANCHOR - 8, hi=ANCHOR + 15, floor=0.1):
    """Như analysis_v14/run_rbbb.py."""
    d = np.diff(x, axis=2); s = x.std(2, keepdims=True) + 1e-6
    e = np.sqrt((d / s) ** 2 + 1e-6).sum(1)[:, lo:hi]
    e = e / (e.max(1, keepdims=True) + 1e-9)
    e = np.logaddexp(0, (e - floor) * 50) / 50
    t = np.arange(lo, hi); w = e / e.sum(1, keepdims=True)
    mu = (w * t).sum(1, keepdims=True)
    return np.sqrt((w * (t - mu) ** 2).sum(1)) * 10.0


def volt(m):
    """m: (N, 9, 75) mV -> dict các biên độ (mV)."""
    m = m - np.median(m[:, :, ANCHOR - 10:ANCHOR - 6], axis=2, keepdims=True)
    avl = (m[:, I_] - m[:, III]) / 2
    q = m[:, :, QRS]
    R, S = q.max(2), np.maximum(0, -q.min(2))
    return dict(sok=S[:, V1] + np.maximum(R[:, V5], R[:, V6]), corn=avl[:, QRS].max(1) + S[:, V3],
                maxR=np.maximum(R.max(1), avl[:, QRS].max(1)),
                strain=m[:, [V5, V6], ANCHOR + 12:ANCHOR + 31].mean(2).min(1))


def features(d, idx):
    v = volt(DAT[d]["M"][idx]); fem = DAT[d]["female"][idx].astype(float)
    a1 = v["sok"]; a2 = v["corn"] + 0.8 * fem
    lr = np.column_stack([v["sok"], v["corn"], fem, qrs_rms(X[d][idx]), v["strain"], v["maxR"]])
    return a1, a2, lr


def cap(y, rng):
    if len(y) <= CAP:
        return np.arange(len(y))
    idx = []
    for k in (0, 1):
        w = np.where(y == k)[0]
        idx.append(rng.choice(w, int(round(CAP * len(w) / len(y))), replace=False))
    return np.sort(np.concatenate(idx))


JOBS = [(i, vs) for i in range(len(SETTINGS)) for vs in (0, 1, 2)][CH::NCH]
t0 = time.time()
for i, vs in JOBS:
    fn = os.path.join(OUT, f"scores_{i}_{vs}.npz")
    if os.path.exists(fn):
        continue
    src, tgt = SETTINGS[i]
    keep = cap(Y[src], np.random.default_rng(0))
    xs, ys = X[src][keep], Y[src][keep]
    tr, va = next(StratifiedShuffleSplit(1, test_size=0.15, random_state=vs).split(np.zeros(len(ys)), ys))
    xt, yt = X[tgt], Y[tgt]; it = np.arange(len(yt))
    s1, s2, slr = features(src, keep); t1, t2, tlr = features(tgt, it)
    f_nets = [L.train_cnn(xs[tr], ys[tr], xs[va], ys[va], s) for s in (0, 1, 2)]
    a_nets = [L.train_cnn(xs[tr], ys[tr], xs[va], ys[va], s) for s in (3, 4, 5)]
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(slr[tr], ys[tr])
    out = dict(y_val=ys[va], y_tgt=yt,
               f_val=L.cnn_predict(f_nets, xs[va]), f_tgt=L.cnn_predict(f_nets, xt),
               alt_val=L.cnn_predict(a_nets, xs[va]), alt_tgt=L.cnn_predict(a_nets, xt),
               a1_val=s1[va], a1_tgt=t1, a2_val=s2[va], a2_tgt=t2,
               a3_val=lr.predict_proba(slr[va])[:, 1], a3_tgt=lr.predict_proba(tlr)[:, 1])
    np.savez_compressed(fn, **out)
    print(i, src, "->", tgt, vs, "AUROC f/alt/A1/A2/A3:",
          [round(roc_auc_score(yt, out[k]), 3) for k in ("f_tgt", "alt_tgt", "a1_tgt", "a2_tgt", "a3_tgt")],
          f"{time.time() - t0:.0f}s", flush=True)
print("xong", f"{time.time() - t0:.0f}s")
