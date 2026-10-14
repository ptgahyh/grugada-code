"""Thư viện dùng chung cho bước 3: dữ liệu, CNN, CBM2 (có chuẩn hóa theo miền dạng trung bình/độ lệch
chuẩn hoặc dạng bền vững trung vị/IQR), cổng dấu. Logic huấn luyện giống hệt concepts_v2/run_v2.py.
"""
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "concepts_v2"))
from concepts2 import CONCEPT_NAMES, BrugadaConcepts  # noqa: E402

torch.set_num_threads(2)
DOMAINS = ["huca", "claris", "mortara"]
K = len(CONCEPT_NAMES)


def load(hd):
    X = {n: np.load(os.path.join(hd, f"{n}.npz"))["X"].astype(np.float32) for n in DOMAINS}
    Y = {n: np.load(os.path.join(hd, f"{n}.npz"))["y"].astype(int) for n in DOMAINS}
    return X, Y


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return thr[np.argmax(tpr - fpr)]


def strat_val(y, d, seed):
    return next(StratifiedShuffleSplit(1, test_size=0.15, random_state=seed).split(np.zeros(len(y)), d * 2 + y))


def pack(X, Y, src):
    xtr = np.concatenate([X[a] for a in src]); ytr = np.concatenate([Y[a] for a in src])
    dtr = np.concatenate([np.array([a] * len(X[a])) for a in src])
    di = np.concatenate([np.full(len(X[a]), i) for i, a in enumerate(src)])
    return xtr, ytr, dtr, di


# ------------------------------------------------------------------ CNN
class Net(nn.Module):
    def __init__(self, n_leads=9):
        super().__init__()
        def blk(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU())
        self.f = nn.Sequential(blk(n_leads, 32, 7), nn.MaxPool1d(2), blk(32, 64, 5), nn.MaxPool1d(2),
                               blk(64, 64, 5), nn.AdaptiveAvgPool1d(1))
        self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.h(self.f(x)).squeeze(1)


def train_cnn(xtr, ytr, xva, yva, seed, epochs=60, patience=12):
    TG = torch.Generator()
    torch.manual_seed(seed); TG.manual_seed(seed); g = np.random.default_rng(seed)
    net = Net(xtr.shape[1]); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ytr == 0).sum() / max(1, (ytr == 1).sum()), dtype=torch.float32))
    xt, yt, xv = torch.tensor(xtr), torch.tensor(ytr, dtype=torch.float32), torch.tensor(xva)
    best, state, wait = -1, None, 0
    for _ in range(epochs):
        net.train(); perm = g.permutation(len(xt))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            xb = torch.roll(xt[b], int(g.integers(-2, 3)), dims=2)
            xb = xb * (1 + 0.1 * (torch.rand(xb.shape[0], xb.shape[1], 1, generator=TG) - 0.5) * 2)
            xb = xb + 0.05 * torch.randn(xb.shape, generator=TG)
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


def cnn_predict(nets, x):
    with torch.no_grad():
        return np.mean([torch.sigmoid(n(torch.tensor(x))).numpy() for n in nets], 0)


# ------------------------------------------------------------------ CBM2
FROZEN = BrugadaConcepts(learn=False)


def concepts_fixed(x):
    with torch.no_grad():
        return FROZEN(torch.tensor(x)).numpy()


def sign_gate(xtr, ytr, dtr, margin=0.03):
    C = concepts_fixed(xtr); s = np.zeros(K)
    for k in range(K):
        a = [roc_auc_score(ytr[dtr == d], C[dtr == d, k]) - 0.5 for d in np.unique(dtr)]
        if all(v > margin for v in a):
            s[k] = 1
        elif all(v < -margin for v in a):
            s[k] = -1
    return s


def loc_scale(C, robust):
    if robust:
        med = C.median(0).values
        q = torch.quantile(C, torch.tensor([0.25, 0.75]), dim=0)
        return med, (q[1] - q[0]) / 1.349 + 1e-4
    return C.mean(0), C.std(0) + 1e-4


class CBM2(nn.Module):
    def __init__(self, signs=None, robust=False):
        super().__init__()
        self.concepts = BrugadaConcepts(learn=True)
        self.v = nn.Parameter(torch.zeros(K)); self.b = nn.Parameter(torch.zeros(1))
        self.signs = None if signs is None else torch.tensor(signs, dtype=torch.float32)
        self.robust = robust; self.stats = {}

    def weights(self):
        return self.v if self.signs is None else self.signs * F.softplus(self.v)

    def norm(self, C, dom, ada, freeze=False, fit=False):
        out = torch.empty_like(C)
        for g in (np.unique(dom) if ada else [None]):
            m = torch.ones(len(C), dtype=torch.bool) if g is None else torch.tensor(dom == g)
            name = "pool" if g is None else g
            if freeze and name in self.stats:
                mu, sd = self.stats[name]
            else:
                mu, sd = loc_scale(C[m], self.robust)
                if fit:
                    self.stats[name] = (mu.detach(), sd.detach())
            out[m] = (C[m] - mu) / sd
        return out

    def forward(self, x, dom, ada, freeze=False, fit=False):
        return self.norm(self.concepts(x), dom, ada, freeze, fit) @ self.weights() + self.b

    def concept_values(self, x):
        with torch.no_grad():
            return self.concepts(torch.tensor(x)).numpy()


def train_cbm2(xtr, ytr, dtr, xva, yva, dva, seed, ada, signs, robust=False, vrex=0.0, epochs=300, patience=40, l1=1e-3):
    torch.manual_seed(seed); g = np.random.default_rng(seed)
    model = CBM2(signs, robust)
    opt = torch.optim.Adam(model.parameters(), lr=2e-2)
    pw = torch.tensor((ytr == 0).sum() / max(1, (ytr == 1).sum()), dtype=torch.float32)
    xt, yt, xv = torch.tensor(xtr), torch.tensor(ytr, dtype=torch.float32), torch.tensor(xva)
    best, state, wait = -1, None, 0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        logit = model(torch.roll(xt, int(g.integers(-1, 2)), dims=2), dtr, ada, fit=True)
        R = torch.stack([F.binary_cross_entropy_with_logits(logit[torch.tensor(dtr == d)], yt[torch.tensor(dtr == d)], pos_weight=pw)
                         for d in np.unique(dtr)])
        (R.mean() + vrex * R.var(unbiased=False) + l1 * model.weights().abs().sum()).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            model(xt, dtr, ada, fit=True)
            auc = roc_auc_score(yva, model(xv, dva, ada, freeze=True).numpy())
        if auc > best:
            best, state, wait = auc, (copy.deepcopy(model.state_dict()), copy.deepcopy(model.stats)), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state[0]); model.stats = state[1]; model.eval()
    return model


def cbm_predict(models, x, ada, dom=None):
    """ada=True: chuẩn hóa bằng thống kê của chính lô x (không dùng nhãn)."""
    with torch.no_grad():
        if ada:
            d = np.array(["__batch__"] * len(x)) if dom is None else dom
            return np.mean([torch.sigmoid(m(torch.tensor(x), d, True, freeze=dom is not None)).numpy() for m in models], 0)
        d = np.array([None] * len(x))
        return np.mean([torch.sigmoid(m(torch.tensor(x), d, False, freeze=True)).numpy() for m in models], 0)
