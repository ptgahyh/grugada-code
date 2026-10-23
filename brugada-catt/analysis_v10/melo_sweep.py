"""Is the re-implemented Melo 2023 DNN a fair opponent? Hyperparameter sweep.

Stage 1 (validation seed 0 only): 9-lead input, hidden units {64, 128, 256} x Gaussian noise {0.05, 0.1, 0.2},
7-network ensemble, all 9 settings. Two picks:
  - "by source validation": highest mean source-validation AUROC (what a practitioner could do);
  - "best on target": highest mean cross-site AUROC (an upper bound that favours Melo).
Stage 2: the picked configuration(s) with the full protocol (3 validation seeds), scores saved, with SRC,
CATT (paper's CBM-sd15 labeller) and ORACLE.

Run: python analysis_v10/melo_sweep.py harmonized concepts_v5 analysis_v10
"""
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HELD, SETTINGS, VSEEDS, ScoreStore, cbm_pseudo, load, metrics, pack, strat_val, v5_raw  # noqa: E402

HD, V5, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(1)
X, Y = load(HD)
R5 = v5_raw(V5)


class GaussNoise(nn.Module):
    def __init__(self, s):
        super().__init__(); self.s = s

    def forward(self, x):
        return x + self.s * torch.randn_like(x) if self.training else x


def make(d, h, s):
    blk = lambda i, o: [nn.Linear(i, o), GaussNoise(s), nn.ReLU(), nn.BatchNorm1d(o)]  # noqa: E731
    return nn.Sequential(nn.Flatten(), *blk(d, h), *blk(h, h), *blk(h, h), nn.Linear(h, 1))


def train(xtr, ytr, xva, yva, seed, h, s, epochs=150, patience=15):
    torch.manual_seed(seed); g = np.random.default_rng(seed)
    net = make(xtr[0].size, h, s); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ytr == 0).sum() / max(1, (ytr == 1).sum()), dtype=torch.float32))
    xt, yt, xv = torch.tensor(xtr), torch.tensor(ytr, dtype=torch.float32), torch.tensor(xva)
    best, state, wait = -1, None, 0
    for _ in range(epochs):
        net.train(); perm = g.permutation(len(xt))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            if len(b) < 2:
                continue
            opt.zero_grad(); lossf(net(xt[b]).squeeze(1), yt[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            auc = roc_auc_score(yva, torch.sigmoid(net(xv).squeeze(1)).numpy())
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
        return np.mean([torch.sigmoid(n(torch.tensor(x)).squeeze(1)).numpy() for n in nets], 0)


CONFIGS = [(h, s) for h in (64, 128, 256) for s in (0.05, 0.1, 0.2)]
t0 = time.time()
# ---------------- stage 1
rows = []
for h, s in CONFIGS:
    for k, (src, tgt) in enumerate(SETTINGS):
        xtr, ytr, dtr, di = pack(X, Y, src)
        tr, va = strat_val(ytr, di, 0)
        nets = [train(xtr[tr], ytr[tr], xtr[va], ytr[va], q, h, s) for q in range(7)]
        sv, st = predict(nets, xtr[va]), predict(nets, X[tgt])
        rows.append(dict(hidden=h, noise=s, k=k, val_auroc=roc_auc_score(ytr[va], sv), auroc=roc_auc_score(Y[tgt], st)))
    print("config", h, s, f"{time.time() - t0:.0f}s", flush=True)
sw = pd.DataFrame(rows)
sw.round(4).to_csv(os.path.join(OUT, "melo_sweep_stage1.csv"), index=False)
g = sw.groupby(["hidden", "noise"])[["val_auroc", "auroc"]].mean()
print(g.round(3).to_string())
pick_val = g.val_auroc.idxmax(); pick_tgt = g.auroc.idxmax()
pick_val = (int(pick_val[0]), float(pick_val[1])); pick_tgt = (int(pick_tgt[0]), float(pick_tgt[1]))
json.dump(dict(by_source_validation=list(pick_val), best_on_target=list(pick_tgt)),
          open(os.path.join(OUT, "melo_sweep_pick.json"), "w"))
print("picked by source validation:", pick_val, "| best on target:", pick_tgt, flush=True)

# ---------------- stage 2
store = ScoreStore(os.path.join(OUT, "melo_scores.npz"))
rows = []
for tag, (h, s) in {"Melo DNN (pick: source val)": pick_val, "Melo DNN (pick: best on target)": pick_tgt}.items():
    for k, (src, tgt) in enumerate(SETTINGS):
        xtr, ytr, dtr, di = pack(X, Y, src)
        for vs in VSEEDS:
            tr, va = strat_val(ytr, di, vs)
            key = f"h{h}_s{s}"
            if not store.has(key, k, vs):
                nets = [train(xtr[tr], ytr[tr], xtr[va], ytr[va], 100 * vs + q, h, s) for q in range(7)]
                store.put(key, k, vs, predict(nets, xtr[va]), predict(nets, X[tgt]))
                store.save()
            sv, st = store.get(key, k, vs)
            rows.append(dict(model=tag, hidden=h, noise=s, k=k, vseed=vs,
                             **metrics(ytr[va], sv, Y[tgt], st, cbm_pseudo(R5[(k, vs)]))))
    print(tag, "done", f"{time.time() - t0:.0f}s", flush=True)
df = pd.DataFrame(rows)
df.round(4).to_csv(os.path.join(OUT, "melo_sweep_stage2_raw.csv"), index=False)
per = df.groupby(["model", "k"])[["auroc", "SRC", "CATT", "ORACLE"]].mean().reset_index()
A = per.groupby("model")["auroc"].agg(["mean", "min"]).add_prefix("auroc9_")
Hh = per[per.k.isin(HELD)].groupby("model")[["SRC", "CATT", "ORACLE"]].agg(["mean", "min"])
Hh.columns = [f"held_{a}_{b}" for a, b in Hh.columns]
summ = A.join(Hh).round(3)
summ.to_csv(os.path.join(OUT, "melo_sweep_summary.csv"))
print(summ.to_string())
