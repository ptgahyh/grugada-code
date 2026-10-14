"""Baseline trong miền và xuyên miền cho 3 bộ Brugada đã chuẩn hóa (huca, claris, mortara).

Mô hình:
  ST_V2  : quy tắc 1 đặc trưng, không huấn luyện: ST ở V2 (60-90 ms sau QRS) trừ nền PR
  LogReg : hồi quy logistic L2 trên 9x75 mẫu (C chọn bằng CV bên trong tập train)
  CNN    : CNN 1D nhỏ (3 khối conv), ensemble 3 seed

Giao thức:
  within : 5-fold stratified CV trong từng miền (gom các bản trùng của Claris vào cùng nhóm)
  cross  : train trên 1 miền, test trên miền khác (6 cặp)
  lodo   : train trên 2 miền, test trên miền còn lại
Ngưỡng phân loại chọn bằng Youden trên 15% dữ liệu train để riêng (validation),
không bao giờ chọn trên tập test.

Chạy: python baselines.py <thư mục harmonized>
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = sys.argv[1]
torch.set_num_threads(2)
DOMAINS = ["huca", "claris", "mortara"]
data = {n: np.load(os.path.join(D, f"{n}.npz")) for n in DOMAINS}
X = {n: data[n]["X"].astype(np.float32) for n in DOMAINS}
Y = {n: data[n]["y"].astype(int) for n in DOMAINS}
# nhóm cho các bản trùng của Claris (dòng 565-626, 596-800, 729-739)
G = {n: np.arange(len(Y[n])) for n in DOMAINS}
for a, b in [(565, 626), (596, 800), (729, 739)]:
    G["claris"][b] = a


# ------------------------------------------------------------------ helpers
def st_v2(x):
    return x[:, 4, 34:38].mean(1) - x[:, 4, 14:20].mean(1)


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return thr[np.argmax(tpr - fpr)]


def metrics(y, s, thr, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    auc = roc_auc_score(y, s)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        boots.append(roc_auc_score(y[i], s[i]))
    p = s >= thr
    return dict(auroc=round(auc, 3), ci=[round(float(np.percentile(boots, 2.5)), 3),
                                          round(float(np.percentile(boots, 97.5)), 3)],
                auprc=round(average_precision_score(y, s), 3),
                sens=round(float(p[y == 1].mean()), 3), spec=round(float((~p)[y == 0].mean()), 3),
                n=int(len(y)), n_pos=int(y.sum()))


def split_val(y, seed=0):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    return next(sss.split(np.zeros(len(y)), y))


# ------------------------------------------------------------------ models
def fit_logreg(xtr, ytr):
    m = make_pipeline(StandardScaler(),
                      LogisticRegressionCV(Cs=[1e-4, 1e-3, 1e-2, 1e-1], cv=5, scoring="roc_auc",
                                           class_weight="balanced", max_iter=3000))
    m.fit(xtr.reshape(len(xtr), -1), ytr)
    return lambda x: m.predict_proba(x.reshape(len(x), -1))[:, 1]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU())
        self.f = nn.Sequential(blk(9, 32, 7), nn.MaxPool1d(2), blk(32, 64, 5), nn.MaxPool1d(2),
                               blk(64, 64, 5), nn.AdaptiveAvgPool1d(1))
        self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.h(self.f(x)).squeeze(1)


def augment(x, g):
    shift = int(g.integers(-2, 3))
    x = torch.roll(x, shift, dims=2)
    x = x * (1 + 0.1 * (torch.rand(x.shape[0], x.shape[1], 1, generator=TG) - 0.5) * 2)
    return x + 0.05 * torch.randn(x.shape, generator=TG)


TG = torch.Generator().manual_seed(0)


def train_cnn(xtr, ytr, xva, yva, seed, epochs=60, patience=12):
    torch.manual_seed(seed); TG.manual_seed(seed); g = np.random.default_rng(seed)
    net = Net(); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-3)
    pw = torch.tensor((ytr == 0).sum() / max(1, (ytr == 1).sum()), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    xt, yt = torch.tensor(xtr), torch.tensor(ytr, dtype=torch.float32)
    xv = torch.tensor(xva)
    best, best_state, wait = -1, None, 0
    for ep in range(epochs):
        net.train(); perm = g.permutation(len(xt))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            opt.zero_grad(); lossf(net(augment(xt[b], g)), yt[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            sv = torch.sigmoid(net(xv)).numpy()
        auc = roc_auc_score(yva, sv)
        if auc > best:
            best, best_state, wait = auc, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state); net.eval()
    return lambda x: torch.sigmoid(net(torch.tensor(x))).detach().numpy()


def fit_cnn(xtr, ytr, seeds=(0, 1, 2)):
    tr, va = split_val(ytr)
    fs = [train_cnn(xtr[tr], ytr[tr], xtr[va], ytr[va], s) for s in seeds]
    return (lambda x: np.mean([f(x) for f in fs], 0)), (xtr[va], ytr[va])


def run(model, xtr, ytr, xte):
    """trả về (điểm test, ngưỡng chọn trên validation)."""
    if model == "ST_V2":
        tr, va = split_val(ytr)
        return st_v2(xte), youden(ytr[va], st_v2(xtr[va]))
    if model == "LogReg":
        tr, va = split_val(ytr)
        f = fit_logreg(xtr[tr], ytr[tr])
        return f(xte), youden(ytr[va], f(xtr[va]))
    if model == "CNN":
        f, (xv, yv) = fit_cnn(xtr, ytr)
        return f(xte), youden(yv, f(xv))


MODELS = ["ST_V2", "LogReg", "CNN"]
results = []
t0 = time.time()

# ------------------------------------------------------------------ within-domain
for n in DOMAINS:
    for m in MODELS:
        s_all, y_all = np.zeros(len(Y[n])), Y[n]
        pred = np.zeros(len(Y[n]), bool)
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
        for tr, te in cv.split(X[n], Y[n], G[n]):
            s, thr = run(m, X[n][tr], Y[n][tr], X[n][te])
            s_all[te] = s; pred[te] = s >= thr
        r = metrics(y_all, s_all, 0.5)
        r.update(sens=round(float(pred[y_all == 1].mean()), 3), spec=round(float((~pred)[y_all == 0].mean()), 3))
        results.append(dict(protocol="within", train=n, test=n, model=m, **r))
        print(results[-1], f"{time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ cross-domain
for a in DOMAINS:
    for b in DOMAINS:
        if a == b:
            continue
        for m in MODELS:
            s, thr = run(m, X[a], Y[a], X[b])
            results.append(dict(protocol="cross", train=a, test=b, model=m, **metrics(Y[b], s, thr)))
            print(results[-1], f"{time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ leave-one-domain-out
for b in DOMAINS:
    src = [a for a in DOMAINS if a != b]
    xtr = np.concatenate([X[a] for a in src]); ytr = np.concatenate([Y[a] for a in src])
    for m in MODELS:
        s, thr = run(m, xtr, ytr, X[b])
        results.append(dict(protocol="lodo", train="+".join(src), test=b, model=m, **metrics(Y[b], s, thr)))
        print(results[-1], f"{time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ domain classifier (shortcut check)
xa = np.concatenate([X[n] for n in DOMAINS]).reshape(-1, 9 * 75)
da = np.concatenate([np.full(len(X[n]), i) for i, n in enumerate(DOMAINS)])
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
pr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)),
                       xa, da, cv=5, method="predict_proba")
dom = {"acc_3class": round(float((pr.argmax(1) == da).mean()), 3),
       "auc_huca_vs_melo": round(roc_auc_score(da == 0, pr[:, 0]), 3),
       "auc_claris_vs_mortara": round(roc_auc_score(da[da > 0] == 1, pr[da > 0, 1]), 3)}
print("domain classifier:", dom)

with open(os.path.join(D, "baseline_results.json"), "w") as f:
    json.dump(dict(results=results, domain_classifier=dom), f, indent=1)
print("xong", f"{time.time() - t0:.0f}s")
