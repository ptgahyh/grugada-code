"""Bước 21c: đường đáp ứng của f theo tham số dạng sóng (phantom đơn giản trên nhịp âm thật). Xem PLAN_vi.md.

Chạy: python analysis_v21/response_v21.py <harmonized> <thư mục riêng có model_*.pt> analysis_v21
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "concepts_v4")); sys.path.insert(0, HERE)
import cbm_lib4 as L  # noqa: E402
from train_v21 import shift  # noqa: E402

HD, PRIV, OUT = sys.argv[1:4]
X, Y = L.load(HD); DOM = L.DOMAINS; T = np.arange(75, dtype=float)


def zs(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-8)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def perturb(x, kind, lv):
    """x: (N, 3, 75) V1–V3 đã z-score. Biến đổi V1, V2 rồi z-score lại từng chuyển đạo."""
    x = x.copy(); tj = (np.argmin(x[:, 0, 18:40], 1) + 18 + 4).astype(float)[:, None]      # t_J = đáy S của V1 + 40 ms
    if kind == "P1_coved_ST":
        add = lv * np.exp(-(T[None] - tj) / 8.0) * (T[None] >= tj)
    elif kind == "P2_T_inversion":
        u = (T[None] - (tj + 12)) / 16.0; add = -lv * np.where((u >= 0) & (u <= 1), 0.5 - 0.5 * np.cos(2 * np.pi * u), 0.0)
    elif kind == "P3_r_prime":
        add = lv * np.exp(-0.5 * ((T[None] - (tj - 1)) / 1.5) ** 2)
    elif kind == "P4_QRS_stretch":
        src = 28 + (T - 28) / lv
        x[:, :2] = np.stack([[np.interp(src, T, x[i, l]) for l in (0, 1)] for i in range(len(x))]); add = None
    elif kind == "P5_lowpass":
        if lv > 0:
            x[:, :2] = gaussian_filter1d(x[:, :2], lv / 10.0, axis=-1, mode="nearest")
        add = None
    elif kind == "P6_shift":
        x[:, :2] = shift(x[:, :2], int(lv)); add = None
    if add is not None:
        x[:, :2] += add[:, None, :]
    x[:, :2] = zs(x[:, :2])
    return x.astype(np.float32)


LEVELS = {"P1_coved_ST": np.arange(0, 2.01, 0.25), "P2_T_inversion": np.arange(0, 2.01, 0.25), "P3_r_prime": np.arange(0, 2.01, 0.25),
          "P4_QRS_stretch": np.arange(1.0, 1.51, 0.1), "P5_lowpass": np.array([0, 5, 10, 15, 20.0]), "P6_shift": np.arange(-3, 4.0)}
rows = []
for m in DOM:
    nets = []
    for sd in torch.load(os.path.join(PRIV, f"model_{m}.pt")):
        n = L.Net(3); n.load_state_dict(sd); n.eval(); nets.append(n)
    for b in DOM:
        x0 = X[b][Y[b] == 0][:, 3:6]; base = logit(L.cnn_predict(nets, perturb(x0, "P6_shift", 0)))
        for kind, lvs in LEVELS.items():
            for lv in lvs:
                d = logit(L.cnn_predict(nets, perturb(x0, kind, lv))) - base
                rows.append(dict(model=m, base=b, pert=kind, level=round(float(lv), 2), mean_dlogit=float(d.mean()),
                                 median_dlogit=float(np.median(d)), n=len(x0)))
    print("done", m, flush=True)
pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "response_v21.csv"), index=False)
