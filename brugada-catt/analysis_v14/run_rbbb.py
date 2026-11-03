"""Bước 14: huấn luyện mô hình đích và mỏ neo cho RBBB, lưu điểm từng bản ghi (theo PLAN_vi.md mục 3).

Mỗi (thiết lập, seed validation):
  f       CNN 9 chuyển đạo (cbm_lib4.train_cnn), ensemble seed 0–2
  CNNalt  cùng kiến trúc, seed 3–5 (mỏ neo phụ thuộc, A3)
  A1      QRS_rms: thời lượng RMS (ms) của năng lượng đạo hàm 9 chuyển đạo, cửa sổ −80..+140 ms quanh đỉnh (không học)
  A2      RBBB-LR: hồi quy logistic L2 (C = 1) đã chuẩn hóa trên [A1, V1 tại +30..+70 ms, min I, min V6 cùng cửa sổ]
Nguồn: tối đa 12.000 bản ghi phân tầng theo nhãn (seed 0); validation 15% phân tầng theo nhãn (seed = seed validation).
Chạy: python run_rbbb.py <thư mục npz> <thư mục điểm> <chunk> <số chunk>  ->  scores_<thiết lập>_<seed>.npz
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
X = {d: np.load(os.path.join(DD, f"{d}.npz"))["X"].astype(np.float32) for d in DOMS}
Y = {d: np.load(os.path.join(DD, f"{d}.npz"))["y"].astype(int) for d in DOMS}
SETTINGS = [(a, b) for a in DOMS for b in DOMS if a != b]
CAP, ANCHOR = 12000, 28


def qrs_rms(x, lo=ANCHOR - 8, hi=ANCHOR + 15, floor=0.1):
    """Thời lượng RMS (ms) của năng lượng đạo hàm chuẩn hóa theo thang từng chuyển đạo (như concepts5.rms_duration)."""
    d = np.diff(x, axis=2); s = x.std(2, keepdims=True) + 1e-6
    e = np.sqrt((d / s) ** 2 + 1e-6).sum(1)[:, lo:hi]
    e = e / (e.max(1, keepdims=True) + 1e-9)
    e = np.logaddexp(0, (e - floor) * 50) / 50
    t = np.arange(lo, hi); w = e / e.sum(1, keepdims=True)
    mu = (w * t).sum(1, keepdims=True)
    return np.sqrt((w * (t - mu) ** 2).sum(1)) * 10.0


def rbbb_feats(x):
    win = slice(ANCHOR + 3, ANCHOR + 8)                 # +30..+70 ms sau đỉnh năng lượng
    return np.column_stack([qrs_rms(x), x[:, 3, win].mean(1), x[:, 0, win].min(1), x[:, 8, win].min(1)])


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
    xt, yt = X[tgt], Y[tgt]
    f_nets = [L.train_cnn(xs[tr], ys[tr], xs[va], ys[va], s) for s in (0, 1, 2)]
    a_nets = [L.train_cnn(xs[tr], ys[tr], xs[va], ys[va], s) for s in (3, 4, 5)]
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(rbbb_feats(xs[tr]), ys[tr])
    out = dict(y_val=ys[va], y_tgt=yt,
               f_val=L.cnn_predict(f_nets, xs[va]), f_tgt=L.cnn_predict(f_nets, xt),
               alt_val=L.cnn_predict(a_nets, xs[va]), alt_tgt=L.cnn_predict(a_nets, xt),
               a1_val=qrs_rms(xs[va]), a1_tgt=qrs_rms(xt),
               a2_val=lr.predict_proba(rbbb_feats(xs[va]))[:, 1], a2_tgt=lr.predict_proba(rbbb_feats(xt))[:, 1])
    np.savez_compressed(fn, **out)
    print(i, src, "->", tgt, vs, "AUROC f/alt/A1/A2:",
          [round(roc_auc_score(yt, out[k]), 3) for k in ("f_tgt", "alt_tgt", "a1_tgt", "a2_tgt")],
          f"{time.time() - t0:.0f}s", flush=True)
print("xong", f"{time.time() - t0:.0f}s")
