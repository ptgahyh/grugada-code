"""Thí nghiệm v2: khái niệm v2 + các cơ chế bất biến giữa miền, chỉ đánh giá NGOÀI MIỀN.

Mô hình
  ST_V2        quy tắc 1 đặc trưng (không huấn luyện)
  CNN          CNN 1D (như baselines.py), ensemble 3 seed khởi tạo
  CBM2         concept bottleneck v2 (28 khái niệm), chuẩn hóa khái niệm gộp theo tập train
  CBM2-Ada     + chuẩn hóa khái niệm THEO TỪNG MIỀN; ở miền test dùng thống kê của chính miền test
               (không cần nhãn, giống AdaBN) — nhắm vào lỗi ngưỡng khi đổi miền
  CBM2-Sign    + lọc khái niệm theo chiều tác động (AUROC đơn biến) nhất quán trên MỌI miền train
               và ràng buộc dấu trọng số theo chiều đó (w_k = s_k · softplus(v_k))
  CBM2-SignAda Sign + Ada
  CBM2-VREx    + phạt phương sai rủi ro giữa các miền train (chỉ LODO, cần >= 2 miền)

Giao thức: cross (6 cặp) + LODO (3). Mỗi mô hình chạy với 3 seed chia validation (0, 1, 2);
điểm cuối = trung bình 3 lần → so sánh cặp bằng bootstrap, hiệu chỉnh Holm trên 9 thiết lập.
Ngưỡng: Youden trên validation (tách từ dữ liệu train).

Chạy: python run_v2.py <thư mục harmonized> <thư mục concepts_v2>
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit

HD, OD = sys.argv[1], sys.argv[2]
sys.path.insert(0, OD)
from concepts2 import CONCEPT_NAMES, BrugadaConcepts  # noqa: E402

torch.set_num_threads(2)
DOMAINS = ["huca", "claris", "mortara"]
X = {n: np.load(os.path.join(HD, f"{n}.npz"))["X"].astype(np.float32) for n in DOMAINS}
Y = {n: np.load(os.path.join(HD, f"{n}.npz"))["y"].astype(int) for n in DOMAINS}
K = len(CONCEPT_NAMES)
VAL_SEEDS = (0, 1, 2)


def st_v2(x):
    return x[:, 4, 34:38].mean(1) - x[:, 4, 14:20].mean(1)


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return thr[np.argmax(tpr - fpr)]


def strat_val(y, d, seed):
    """tách 15% validation, phân tầng theo (miền, nhãn)."""
    key = d * 2 + y
    return next(StratifiedShuffleSplit(1, test_size=0.15, random_state=seed).split(np.zeros(len(y)), key))


# ------------------------------------------------------------------ CNN (như baselines.py)
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


TG = torch.Generator()


def train_cnn(xtr, ytr, xva, yva, seed, epochs=60, patience=12):
    torch.manual_seed(seed); TG.manual_seed(seed); g = np.random.default_rng(seed)
    net = Net(); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-3)
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
    return lambda x: torch.sigmoid(net(torch.tensor(x))).detach().numpy()


# ------------------------------------------------------------------ CBM v2 + biến thể
FROZEN = BrugadaConcepts(learn=False)


def concepts_fixed(x):
    with torch.no_grad():
        return FROZEN(torch.tensor(x)).numpy()


def sign_gate(xtr, ytr, dtr, margin=0.03):
    """chiều tác động của từng khái niệm, chỉ dùng dữ liệu train. 0 = loại."""
    C = concepts_fixed(xtr)
    s = np.zeros(K)
    for k in range(K):
        aucs = [roc_auc_score(ytr[dtr == d], C[dtr == d, k]) - 0.5 for d in np.unique(dtr)]
        if all(a > margin for a in aucs):
            s[k] = 1
        elif all(a < -margin for a in aucs):
            s[k] = -1
    return s


class CBM2(nn.Module):
    def __init__(self, signs=None):
        super().__init__()
        self.concepts = BrugadaConcepts(learn=True)
        self.v = nn.Parameter(torch.zeros(K))
        self.b = nn.Parameter(torch.zeros(1))
        self.signs = None if signs is None else torch.tensor(signs, dtype=torch.float32)
        self.stats = {}

    def weights(self):
        if self.signs is None:
            return self.v
        return self.signs * F.softplus(self.v)        # dấu cố định, khái niệm bị loại có trọng số 0

    def norm(self, C, dom, ada, freeze=False, key=None):
        """chuẩn hóa: gộp (ada=False) hoặc theo từng miền (ada=True)."""
        out = torch.empty_like(C)
        groups = np.unique(dom) if ada else [None]
        for g in groups:
            m = torch.ones(len(C), dtype=torch.bool) if g is None else torch.tensor(dom == g)
            name = "pool" if g is None else g
            if freeze and name in self.stats:
                mu, sd = self.stats[name]
            else:
                mu, sd = C[m].mean(0), C[m].std(0) + 1e-4
                if key == "fit":
                    self.stats[name] = (mu.detach(), sd.detach())
            out[m] = (C[m] - mu) / sd
        return out

    def forward(self, x, dom, ada, freeze=False, key=None):
        Cn = self.norm(self.concepts(x), dom, ada, freeze, key)
        return Cn @ self.weights() + self.b


def train_cbm2(xtr, ytr, dtr, xva, yva, dva, seed, ada, signs, vrex=0.0, epochs=300, patience=40, l1=1e-3):
    torch.manual_seed(seed); g = np.random.default_rng(seed)
    model = CBM2(signs)
    opt = torch.optim.Adam(model.parameters(), lr=2e-2)
    pw = torch.tensor((ytr == 0).sum() / max(1, (ytr == 1).sum()), dtype=torch.float32)
    xt, yt, xv = torch.tensor(xtr), torch.tensor(ytr, dtype=torch.float32), torch.tensor(xva)
    best, state, wait = -1, None, 0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        xa = torch.roll(xt, int(g.integers(-1, 2)), dims=2)
        logit = model(xa, dtr, ada, key="fit")
        losses = []
        for d in np.unique(dtr):
            m = torch.tensor(dtr == d)
            losses.append(F.binary_cross_entropy_with_logits(logit[m], yt[m], pos_weight=pw))
        R = torch.stack(losses)
        loss = R.mean() + vrex * R.var(unbiased=False) + l1 * model.weights().abs().sum()
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            model(xt, dtr, ada, key="fit")                       # cập nhật thống kê chuẩn hóa với mốc mới
            sv = model(xv, dva, ada, freeze=True).numpy()          # validation dùng thống kê của miền train
        auc = roc_auc_score(yva, sv)
        if auc > best:
            best, state, wait = auc, (copy.deepcopy(model.state_dict()), copy.deepcopy(model.stats)), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state[0]); model.stats = state[1]; model.eval()
    return model


def predict_cbm2(models, x, ada):
    """miền test: Ada -> chuẩn hóa bằng thống kê của chính miền test (không dùng nhãn)."""
    with torch.no_grad():
        if ada:
            dom = np.array(["__test__"] * len(x))
            return np.mean([torch.sigmoid(m(torch.tensor(x), dom, True)).numpy() for m in models], 0)
        dom = np.array([None] * len(x))
        return np.mean([torch.sigmoid(m(torch.tensor(x), dom, False, freeze=True)).numpy() for m in models], 0)


VARIANTS = {"CBM2": dict(ada=False, sign=False, vrex=0.0),
            "CBM2-Ada": dict(ada=True, sign=False, vrex=0.0),
            "CBM2-Sign": dict(ada=False, sign=True, vrex=0.0),
            "CBM2-SignAda": dict(ada=True, sign=True, vrex=0.0),
            "CBM2-VREx": dict(ada=False, sign=False, vrex=10.0)}


def run(model, src, tgt, vseed):
    xtr = np.concatenate([X[a] for a in src]); ytr = np.concatenate([Y[a] for a in src])
    dtr = np.concatenate([np.array([a] * len(X[a])) for a in src])
    tr, va = strat_val(ytr, np.concatenate([np.full(len(X[a]), i) for i, a in enumerate(src)]), vseed)
    xte = X[tgt]
    if model == "ST_V2":
        return st_v2(xte), youden(ytr[va], st_v2(xtr[va])), None
    if model == "CNN":
        fs = [train_cnn(xtr[tr], ytr[tr], xtr[va], ytr[va], s) for s in (0, 1, 2)]
        f = lambda x: np.mean([h(x) for h in fs], 0)
        return f(xte), youden(ytr[va], f(xtr[va])), None
    cfg = VARIANTS[model]
    signs = sign_gate(xtr[tr], ytr[tr], dtr[tr]) if cfg["sign"] else None
    ms = [train_cbm2(xtr[tr], ytr[tr], dtr[tr], xtr[va], ytr[va], dtr[va], s, cfg["ada"], signs, cfg["vrex"]) for s in (0, 1)]
    # ngưỡng trên validation: chuẩn hóa validation theo thống kê miền train của chính nó
    with torch.no_grad():
        sv = np.mean([torch.sigmoid(m(torch.tensor(xtr[va]), dtr[va] if cfg["ada"] else np.array([None] * len(va)),
                                      cfg["ada"], freeze=True)).numpy() for m in ms], 0)
    extra = dict(signs=None if signs is None else signs.tolist(),
                 n_kept=None if signs is None else int((signs != 0).sum()),
                 fiducials=[m.concepts.fiducial_values() for m in ms],
                 weights=[m.weights().detach().numpy().round(3).tolist() for m in ms])
    return predict_cbm2(ms, xte, cfg["ada"]), youden(ytr[va], sv), extra


SETTINGS = [([a], b) for a in DOMAINS for b in DOMAINS if a != b] + [([a for a in DOMAINS if a != b], b) for b in DOMAINS]
MODELS = ["ST_V2", "CNN", "CBM2", "CBM2-Ada", "CBM2-Sign", "CBM2-SignAda", "CBM2-VREx"]
t0 = time.time()
raw = []
for src, tgt in SETTINGS:
    for m in MODELS:
        if m == "CBM2-VREx" and len(src) < 2:
            continue
        for vs in (VAL_SEEDS if m != "ST_V2" else (0,)):
            s, thr, extra = run(m, src, tgt, vs)
            y = Y[tgt]; p = s >= thr
            r = dict(train="+".join(src), test=tgt, model=m, vseed=vs, auroc=round(roc_auc_score(y, s), 4),
                     sens=round(float(p[y == 1].mean()), 3), spec=round(float((~p)[y == 0].mean()), 3),
                     scores=s.round(5).tolist(), extra=extra)
            raw.append(r)
            print({k: r[k] for k in ["train", "test", "model", "vseed", "auroc", "sens", "spec"]},
                  "" if not extra or extra["n_kept"] is None else f"kept={extra['n_kept']}", f"{time.time() - t0:.0f}s", flush=True)
json.dump(raw, open(os.path.join(OD, "v2_raw.json"), "w"))
print("xong", f"{time.time() - t0:.0f}s")
