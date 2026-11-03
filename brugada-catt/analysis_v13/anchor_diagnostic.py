"""Bước 13f: vì sao ST_V2 bị dự đoán dư (1.17)? Kiểm giả thuyết "J̃ có hai đỉnh" (vi phạm điều kiện C1 của Định lý 5a).

Với mỗi (thiết lập, seed, mỏ neo):
  peak2_ratio   độ cao đỉnh phụ lớn nhất của J̃ (KDE Scott) so với đỉnh chính, đo từ đáy giữa hai đỉnh (0 = một đỉnh)
  peak_gap      khoảng cách (logit) giữa hai đỉnh
  sd_over_iqr   SD(t̂) / (IQR(t̂)/1.349) qua 300 mẫu con 35% cohort; Chernoff cho ~1.0, phân phối hai đỉnh cho > 1.2
  bimod_coef    hệ số hai đỉnh của Sarle (> 0.555 gợi ý hai đỉnh)
Kết hợp với brugada_unified_v2.csv để xem tỉ số dự đoán/quan sát có tăng theo các chỉ số này không.
Chạy: python analysis_v13/anchor_diagnostic.py analysis_v11 analysis_v13
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, kurtosis, skew, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis_v12"))
from common12 import youden  # noqa: E402

V11, OUT = sys.argv[1:3]
LABS = {"CBM5": "CBM-sd15", "CBM3": "CBM", "ST_V2": "ST_V2", "LogReg9": "LogReg9", "CNN9": "CNN9",
        "ResNet": "ResNet", "CNNalt": "CNNalt"}
rng = np.random.default_rng(1317)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def jtilde(f, pl, g):
    dg = g[1] - g[0]
    return np.cumsum(gaussian_kde(f[pl == 0])(g) - gaussian_kde(f[pl == 1])(g)) * dg


rows = []
for i in range(9):
    for vs in range(3):
        fn = os.path.join(V11, "scores", f"lab_{i}_{vs}.npz")
        if not os.path.exists(fn):
            continue
        Z = np.load(fn); yv, yt = Z["y_val"], Z["y_tgt"]; f = logit(Z["f_tgt"]); N = len(yt)
        g = np.linspace(np.percentile(f, 0.5), np.percentile(f, 99.5), 3001)
        for k, name in LABS.items():
            pl = (Z[f"{k}_tgt"] >= youden(yv, Z[f"{k}_val"])).astype(int)
            if pl.sum() < 10 or (1 - pl).sum() < 10:
                continue
            J = jtilde(f, pl, g); i0 = int(np.argmax(J))
            pk, _ = find_peaks(J)
            others = [p for p in pk if p != i0]
            ratio, gap = 0.0, 0.0
            if others:
                j = max(others, key=lambda p: J[p])
                lo, hi = sorted((i0, j)); valley = J[lo:hi + 1].min()
                ratio = float((J[j] - valley) / max(J[i0] - valley, 1e-12)); gap = float(abs(g[j] - g[i0]))
            n = int(round(0.35 * N)); th = []
            for _ in range(300):
                idx = rng.choice(N, n, replace=False)
                if pl[idx].min() != pl[idx].max():
                    th.append(youden(pl[idx], f[idx]))
            th = np.array(th); iqr = np.subtract(*np.percentile(th, [75, 25])) / 1.349
            bc = (skew(th) ** 2 + 1) / (kurtosis(th) + 3 * (len(th) - 1) ** 2 / ((len(th) - 2) * (len(th) - 3)))
            rows.append(dict(setting=i, vseed=vs, anchor=name, peak2_ratio=ratio, peak_gap=gap,
                             sd_over_iqr=float(th.std() / max(iqr, 1e-9)), bimod_coef=float(bc)))
        print("setting", i, "seed", vs, flush=True)
D = pd.DataFrame(rows)
D.round(4).to_csv(os.path.join(OUT, "anchor_diagnostic.csv"), index=False)
print(D.groupby("anchor")[["peak2_ratio", "sd_over_iqr", "bimod_coef"]].median().round(3))
V2 = os.path.join(OUT, "brugada_unified_v2.csv")
if os.path.exists(V2):
    B = pd.read_csv(V2)
    B = B[np.isclose(B.n / B.N, 0.35, atol=0.01)].assign(ratio=lambda d: d["sd_pred_1.0"] / d.sd_obs)
    M = B.merge(D, on=["setting", "vseed", "anchor"])
    for c in ("peak2_ratio", "sd_over_iqr", "bimod_coef"):
        print(f"Spearman(tỉ số dự đoán/quan sát, {c}) = {spearmanr(M.ratio, M[c])[0]:.2f}")
    M["two_peaks"] = M.peak2_ratio > 0.2
    print(M.groupby(["anchor", "two_peaks"]).ratio.median().round(2))
