"""Hiệu ứng "chỉ V1–V3" có phụ thuộc kiến trúc không?

Mô hình, mỗi mô hình chạy 2 phiên bản (9 chuyển đạo và chỉ V1–V3), 9 thiết lập ngoài miền, seed validation 0:
  LogReg    hồi quy logistic L2 trên tín hiệu thô (C chọn bằng CV trong tập train)
  ResNet1D  3 khối residual (32-64-64 kênh), ensemble 3 seed
  CNN       CNN 1D như các bước trước (lấy lại từ compare_v4 cho V1–V3 và concepts_v2 cho 9 chuyển đạo)

Chạy: python arch_v4.py <harmonized> <output>
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cbm_lib4 import DOMAINS, load, pack, strat_val  # noqa: E402

HD, OUT = sys.argv[1], sys.argv[2]
X, Y = load(HD)
torch.set_num_threads(2)
SETTINGS = [([a], b) for a in DOMAINS for b in DOMAINS if a != b] + [([a for a in DOMAINS if a != b], b) for b in DOMAINS]


class Res(nn.Module):
    def __init__(self, c_in, c_out, k):
        super().__init__()
        self.a = nn.Sequential(nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.BatchNorm1d(c_out), nn.ReLU(),
                               nn.Conv1d(c_out, c_out, k, padding=k // 2), nn.BatchNorm1d(c_out))
        self.s = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return torch.relu(self.a(x) + self.s(x))


class ResNet1D(nn.Module):
    def __init__(self, n_leads):
        super().__init__()
        self.f = nn.Sequential(Res(n_leads, 32, 7), nn.MaxPool1d(2), Res(32, 64, 5), nn.MaxPool1d(2), Res(64, 64, 3),
                               nn.AdaptiveAvgPool1d(1))
        self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.h(self.f(x)).squeeze(1)


def train_net(xtr, ytr, xva, yva, seed, epochs=60, patience=12):
    TG = torch.Generator(); torch.manual_seed(seed); TG.manual_seed(seed); g = np.random.default_rng(seed)
    net = ResNet1D(xtr.shape[1]); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ytr == 0).sum() / max(1, (ytr == 1).sum()), dtype=torch.float32))
    xt, yt, xv = torch.tensor(xtr), torch.tensor(ytr, dtype=torch.float32), torch.tensor(xva)
    best, state, wait = -1, None, 0
    for _ in range(epochs):
        net.train(); perm = g.permutation(len(xt))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            xb = torch.roll(xt[b], int(g.integers(-2, 3)), dims=2)
            xb = xb * (1 + 0.1 * (torch.rand(xb.shape[0], xb.shape[1], 1, generator=TG) - 0.5) * 2) + 0.05 * torch.randn(xb.shape, generator=TG)
            opt.zero_grad(); lossf(net(xb), yt[b]).backward(); opt.step()
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


rows = []
t0 = time.time()
for src, tgt in SETTINGS:
    xtr, ytr, dtr, di = pack(X, Y, src)
    tr, va = strat_val(ytr, di, 0)
    r = dict(train="+".join(src), test=tgt)
    for tag, sl in [("9", slice(0, 9)), ("V1V3", slice(3, 6))]:
        a, b, t = xtr[:, sl], X[tgt][:, sl], tr
        m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[1e-4, 1e-3, 1e-2, 1e-1], cv=5, scoring="roc_auc",
                                                                 class_weight="balanced", max_iter=3000))
        m.fit(a[t].reshape(len(t), -1), ytr[t])
        r[f"LogReg-{tag}"] = round(roc_auc_score(Y[tgt], m.predict_proba(b.reshape(len(b), -1))[:, 1]), 3)
        nets = [train_net(a[tr], ytr[tr], a[va], ytr[va], s) for s in (0, 1, 2)]
        with torch.no_grad():
            s = np.mean([torch.sigmoid(n(torch.tensor(b))).numpy() for n in nets], 0)
        r[f"ResNet-{tag}"] = round(roc_auc_score(Y[tgt], s), 3)
    rows.append(r); print(r, f"{time.time() - t0:.0f}s", flush=True)
json.dump(rows, open(os.path.join(OUT, "arch_v4.json"), "w"), indent=1)
print("xong", f"{time.time() - t0:.0f}s")
