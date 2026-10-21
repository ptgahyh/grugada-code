"""Prior method under our protocol: the Melo et al. (PNAS Nexus 2023) DNN, re-implemented.

Architecture as described in the paper: three hidden blocks (fully connected -> Gaussian noise -> ReLU ->
BatchNorm), sigmoid output, input = the concatenated median beats of 9 leads, ensemble of 7 networks.
Unit counts, noise level and optimiser are not reported in the main text; we use 128 units per block,
Gaussian noise sigma = 0.1, Adam (lr 1e-3, weight decay 1e-4), class-weighted BCE and early stopping on the
source-validation AUROC. Input here is our harmonized 100 Hz beat (Melo used 200 Hz).

Evaluated on the same 9 cross-site settings and the same 3 validation seeds as the main study.
Threshold: SRC (source Youden), CATT (pseudo-labels from CBM-sd15 at its own source threshold, read from
concepts_v5/v5_raw_*.jsonl so the labeller is identical to the paper), ORACLE.
Inputs: 9 leads (as Melo) and V1-V3.

Run: python analysis_v8/melo_repro.py harmonized concepts_v5 analysis_v8
"""
import copy
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "concepts_v4"))
from cbm_lib4 import DOMAINS, load, pack, strat_val  # noqa: E402

HD, V5, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(1)
X, Y = load(HD)
SETTINGS = [([a], b) for a in DOMAINS for b in DOMAINS if a != b] + \
           [([a for a in DOMAINS if a != b], b) for b in DOMAINS]
HELD = [2, 3, 4, 5, 6, 7, 8]
INPUTS = {"9 leads": list(range(9)), "V1-V3": [3, 4, 5]}
V5R = {}
for f in sorted(glob.glob(os.path.join(V5, "v5_raw_*.jsonl"))):
    for line in open(f):
        r = json.loads(line); V5R[(r["setting"], r["vseed"])] = r


class GaussNoise(nn.Module):
    def __init__(self, s):
        super().__init__(); self.s = s

    def forward(self, x):
        return x + self.s * torch.randn_like(x) if self.training else x


class MeloDNN(nn.Module):
    def __init__(self, d, h=128, s=0.1):
        super().__init__()
        blk = lambda i, o: [nn.Linear(i, o), GaussNoise(s), nn.ReLU(), nn.BatchNorm1d(o)]  # noqa: E731
        self.net = nn.Sequential(*blk(d, h), *blk(h, h), *blk(h, h), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x.flatten(1)).squeeze(1)


def train(xtr, ytr, xva, yva, seed, epochs=150, patience=15):
    torch.manual_seed(seed); g = np.random.default_rng(seed)
    net = MeloDNN(xtr[0].size); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ytr == 0).sum() / max(1, (ytr == 1).sum()), dtype=torch.float32))
    xt, yt, xv = torch.tensor(xtr), torch.tensor(ytr, dtype=torch.float32), torch.tensor(xva)
    best, state, wait = -1, None, 0
    for _ in range(epochs):
        net.train(); perm = g.permutation(len(xt))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            if len(b) < 2:
                continue
            opt.zero_grad(); lossf(net(xt[b]), yt[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            auc = roc_auc_score(yva, torch.sigmoid(net(xv)).numpy())
        if auc > best:
            best, state, wait = auc, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(state); net.eval()
    return net


def predict(nets, x):
    with torch.no_grad():
        return np.mean([torch.sigmoid(n(torch.tensor(x))).numpy() for n in nets], 0)


def youden(y, s):
    f, t, th = roc_curve(y, s)
    th = np.minimum(th, s.max() + 1e-9)
    return th[np.argmax(t - f)]


def bacc(y, s, t):
    p = s >= t
    return 0.5 * (p[y == 1].mean() + (~p)[y == 0].mean())


rows = []
for k, (src, tgt) in enumerate(SETTINGS):
    xtr, ytr, dtr, di = pack(X, Y, src)
    for vs in (0, 1, 2):
        tr, va = strat_val(ytr, di, vs)
        r5 = V5R[(k, vs)]
        assert ytr[va].tolist() == r5["y_val"]
        cv, ct = np.array(r5["models"]["CBM5-Sign"]["val"]), np.array(r5["models"]["CBM5-Sign"]["tgt"])
        pseudo = (ct >= youden(ytr[va], cv)).astype(int)
        for name, idx in INPUTS.items():
            a, b = xtr[:, idx], X[tgt][:, idx]
            nets = [train(a[tr], ytr[tr], a[va], ytr[va], 100 * vs + s) for s in range(7)]
            sv, st = predict(nets, a[va]), predict(nets, b)
            rows.append(dict(k=k, train="+".join(src), test=tgt, vseed=vs, input=name,
                             auroc=roc_auc_score(Y[tgt], st), SRC=bacc(Y[tgt], st, youden(ytr[va], sv)),
                             CATT=bacc(Y[tgt], st, youden(pseudo, st)), ORACLE=bacc(Y[tgt], st, youden(Y[tgt], st))))
            print({q: (round(v, 3) if isinstance(v, float) else v) for q, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "melo_repro_raw.csv"), index=False)

df = pd.DataFrame(rows)
per = df.groupby(["input", "k", "train", "test"])[["auroc", "SRC", "CATT", "ORACLE"]].mean().reset_index()
per.round(4).to_csv(os.path.join(OUT, "melo_repro_per_setting.csv"), index=False)
for scope, ks in (("all 9", range(9)), ("held-out 7", HELD)):
    sub = per[per.k.isin(list(ks))]
    print(f"\n{scope} settings:")
    print(sub.groupby("input")[["auroc", "SRC", "CATT", "ORACLE"]].agg(["mean", "min"]).round(3).to_string())
print("\nClaris -> Mortara (Melo reported AUROC 0.934 at 200 Hz with original labels):")
print(per[(per.train == "claris") & (per.test == "mortara")].round(3).to_string())
