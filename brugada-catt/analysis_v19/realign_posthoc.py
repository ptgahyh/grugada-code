"""Bước 19, QUAN SÁT SAU (không có trong kế hoạch): gom cụm lại sau khi căn từng bản ghi theo đáy sóng S của V1.

Lý do: hình cụm cho thấy đáy QRS ở V1 lệch khoảng 20 ms giữa HUCA và Melo, nên các cụm ban đầu một phần phản ánh
độ lệch căn thời gian chứ không phải hình dạng ST. Ở đây cửa sổ đặc trưng là [đáy V1 − 40 ms, đáy V1 + 250 ms].
Chạy: python analysis_v19/realign_posthoc.py <harmonized> <analysis_v8/scores> analysis_v19
"""
import json
import os
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit

HD, SC, OUT = sys.argv[1:4]
DOM = ["huca", "claris", "mortara"]; SEED = 1901; K = 5; BROKEN = [0, 1, 2, 4, 6]
SETTINGS = [([a], b) for a in DOM for b in DOM if a != b] + [([a for a in DOM if a != b], b) for b in DOM]
Z = {n: np.load(os.path.join(HD, f"{n}.npz")) for n in DOM}
X = {n: Z[n]["X"].astype(float) for n in DOM}; Y = {n: Z[n]["y"].astype(int) for n in DOM}
NAD = {n: np.argmin(X[n][:, 3, 18:40], 1) + 18 for n in DOM}
FEAT = {n: np.stack([X[n][i, 3:5, NAD[n][i] - 4:NAD[n][i] + 26].ravel() for i in range(len(X[n]))]) for n in DOM}
rng = np.random.default_rng(SEED)
fit = np.concatenate([FEAT[n][rng.choice(len(FEAT[n]), 350, replace=False)] for n in DOM])
pca = PCA(8, random_state=SEED).fit(fit); km = KMeans(K, n_init=20, random_state=SEED).fit(pca.transform(fit))
C = {n: km.predict(pca.transform(FEAT[n])) for n in DOM}
sh = lambda c: np.bincount(c, minlength=K) / len(c)
tv = lambda a, b: 0.5 * float(np.abs(a - b).sum())
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
S = {"nadir_V1_ms_median": {n: float(np.median((NAD[n] - 28) * 10)) for n in DOM},
     "tv_neg": {f"{a}_{b}": round(tv(sh(C[a][Y[a] == 0]), sh(C[b][Y[b] == 0])), 3)
                for a, b in (("huca", "claris"), ("huca", "mortara"), ("claris", "mortara"))}}
num = den = 0.0
for i in BROKEN:
    src, tgt = SETTINGS[i]; d = c = w = 0.0
    y = np.concatenate([Y[a] for a in src]); cc = np.concatenate([C[a] for a in src])
    di = np.concatenate([np.full(len(Y[a]), k) for k, a in enumerate(src)])
    for vs in range(3):
        _, va = next(StratifiedShuffleSplit(1, test_size=0.15, random_state=vs).split(np.zeros(len(y)), di * 2 + y))
        r = np.load(os.path.join(SC, f"scores_{i}_{vs}.npz")); s = y[va] == 0; t = Y[tgt] == 0
        fs, cs, ft, ct = logit(r["cnn_val"])[s], cc[va][s], logit(r["cnn_tgt"])[t], C[tgt][t]
        mS = np.array([fs[cs == k].mean() if (cs == k).any() else fs.mean() for k in range(K)])
        mT = np.array([ft[ct == k].mean() if (ct == k).any() else 0.0 for k in range(K)])
        d += (ft.mean() - fs.mean()) / 3; w += float((sh(ct) * (mT - mS)).sum()) / 3
    num += w * np.sign(d); den += abs(d)
S["within_share_broken_neg"] = round(float(num / den), 3)
json.dump(S, open(os.path.join(OUT, "realign_posthoc.json"), "w"), indent=1); print(json.dumps(S, indent=1))
