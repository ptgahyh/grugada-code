"""So sánh mô hình khái niệm với baseline trên 3 miền (huca, claris, mortara).

Mô hình
  ST_V2      quy tắc 1 đặc trưng, không huấn luyện (giống baselines.py)
  LogReg     hồi quy logistic trên tín hiệu thô 9x75
  CNN        CNN 1D, ensemble 3 seed (giống baselines.py)
  ConceptLR  25 khái niệm với mốc CỐ ĐỊNH theo giá trị lâm sàng + hồi quy logistic
  CBM        concept bottleneck end-to-end: mốc học được + đầu tuyến tính, ensemble 3 seed

Giao thức và lần chia giống hệt baselines.py (StratifiedGroupKFold(5, shuffle, seed 0); validation 15%, seed 0).
Ngưỡng: Youden trên validation. Lưu điểm từng bệnh nhân để so sánh cặp (paired bootstrap ΔAUROC).

Chạy: python run_concepts.py <thư mục harmonized> <thư mục concepts>
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HD, CD = sys.argv[1], sys.argv[2]
sys.path.insert(0, CD)
from concepts import CONCEPT_NAMES, BrugadaConcepts, ConceptBottleneck  # noqa: E402

torch.set_num_threads(2)
DOMAINS = ["huca", "claris", "mortara"]
X = {n: np.load(os.path.join(HD, f"{n}.npz"))["X"].astype(np.float32) for n in DOMAINS}
Y = {n: np.load(os.path.join(HD, f"{n}.npz"))["y"].astype(int) for n in DOMAINS}
G = {n: np.arange(len(Y[n])) for n in DOMAINS}
for a, b in [(565, 626), (596, 800), (729, 739)]:
    G["claris"][b] = a


# ------------------------------------------------------------------ chung
def st_v2(x):
    return x[:, 4, 34:38].mean(1) - x[:, 4, 14:20].mean(1)


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return thr[np.argmax(tpr - fpr)]


def split_val(y, seed=0):
    return next(StratifiedShuffleSplit(1, test_size=0.15, random_state=seed).split(np.zeros(len(y)), y))


def boot_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], s[i]))
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


def paired_delta(y, s1, s2, n=2000, seed=1):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    d = np.array(d)
    return dict(delta=round(roc_auc_score(y, s1) - roc_auc_score(y, s2), 3),
                ci=[round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)],
                p_le0=round(float((d <= 0).mean()), 4))


# ------------------------------------------------------------------ mô hình baseline
def fit_logreg_raw(xtr, ytr):
    m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[1e-4, 1e-3, 1e-2, 1e-1], cv=5, scoring="roc_auc",
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


TG = torch.Generator()


def augment(x, g):
    x = torch.roll(x, int(g.integers(-2, 3)), dims=2)
    x = x * (1 + 0.1 * (torch.rand(x.shape[0], x.shape[1], 1, generator=TG) - 0.5) * 2)
    return x + 0.05 * torch.randn(x.shape, generator=TG)


def train_cnn(xtr, ytr, xva, yva, seed, epochs=60, patience=12):
    torch.manual_seed(seed); TG.manual_seed(seed); g = np.random.default_rng(seed)
    net = Net(); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ytr == 0).sum() / max(1, (ytr == 1).sum()), dtype=torch.float32))
    xt, yt, xv = torch.tensor(xtr), torch.tensor(ytr, dtype=torch.float32), torch.tensor(xva)
    best, state, wait = -1, None, 0
    for _ in range(epochs):
        net.train(); perm = g.permutation(len(xt))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]; opt.zero_grad(); lossf(net(augment(xt[b], g)), yt[b]).backward(); opt.step()
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


# ------------------------------------------------------------------ mô hình khái niệm
FROZEN = BrugadaConcepts(learn=False)


def concepts_fixed(x):
    with torch.no_grad():
        return FROZEN(torch.tensor(x)).numpy()


def fit_concept_lr(xtr, ytr):
    m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1.0], cv=5, scoring="roc_auc",
                                                             class_weight="balanced", max_iter=5000))
    m.fit(concepts_fixed(xtr), ytr)
    return (lambda x: m.predict_proba(concepts_fixed(x))[:, 1]), m


def refresh_norm(model, x):
    """tính lại thống kê chuẩn hóa khái niệm trên toàn bộ tập train với mốc cuối cùng."""
    model.norm.reset_running_stats(); model.norm.momentum = 1.0
    model.train()
    with torch.no_grad():
        model(x)
    model.norm.momentum = 0.1; model.eval()


def train_cbm(xtr, ytr, xva, yva, seed, epochs=400, patience=60, l1=1e-3):
    torch.manual_seed(seed); g = np.random.default_rng(seed)
    model = ConceptBottleneck(learn_fiducials=True)
    opt = torch.optim.Adam([{"params": model.head.parameters(), "lr": 2e-2, "weight_decay": 1e-3},
                            {"params": model.concepts.parameters(), "lr": 2e-2}])
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ytr == 0).sum() / max(1, (ytr == 1).sum()), dtype=torch.float32))
    xt, yt, xv = torch.tensor(xtr), torch.tensor(ytr, dtype=torch.float32), torch.tensor(xva)
    best, state, wait = -1, None, 0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        xa = torch.roll(xt, int(g.integers(-1, 2)), dims=2)          # lệch căn chỉnh ±10 ms
        loss = lossf(model(xa), yt) + l1 * model.head.weight.abs().sum()
        loss.backward(); opt.step()
        refresh_norm(model, xt)
        with torch.no_grad():
            auc = roc_auc_score(yva, model(xv).numpy())
        if auc > best:
            best, state, wait = auc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state); model.eval()
    return model


def fit_cbm(xtr, ytr, seeds=(0, 1, 2)):
    tr, va = split_val(ytr)
    models = [train_cbm(xtr[tr], ytr[tr], xtr[va], ytr[va], s) for s in seeds]

    def f(x):
        with torch.no_grad():
            return np.mean([torch.sigmoid(m(torch.tensor(x))).numpy() for m in models], 0)
    return f, (xtr[va], ytr[va]), models


def run(model, xtr, ytr, xte):
    tr, va = split_val(ytr)
    if model == "ST_V2":
        return st_v2(xte), youden(ytr[va], st_v2(xtr[va]))
    if model == "LogReg":
        f = fit_logreg_raw(xtr[tr], ytr[tr]); return f(xte), youden(ytr[va], f(xtr[va]))
    if model == "ConceptLR":
        f, _ = fit_concept_lr(xtr[tr], ytr[tr]); return f(xte), youden(ytr[va], f(xtr[va]))
    if model == "CNN":
        fs = [train_cnn(xtr[tr], ytr[tr], xtr[va], ytr[va], s) for s in (0, 1, 2)]
        f = lambda x: np.mean([h(x) for h in fs], 0)
        return f(xte), youden(ytr[va], f(xtr[va]))
    if model == "CBM":
        f, (xv, yv), _ = fit_cbm(xtr, ytr)
        return f(xte), youden(yv, f(xv))


MODELS = ["ST_V2", "LogReg", "CNN", "ConceptLR", "CBM"]
results, scores = [], {}
t0 = time.time()


def record(protocol, a, b, m, y, s, pred):
    r = dict(protocol=protocol, train=a, test=b, model=m, auroc=round(roc_auc_score(y, s), 3), ci=boot_ci(y, s),
             auprc=round(average_precision_score(y, s), 3),
             sens=round(float(pred[y == 1].mean()), 3), spec=round(float((~pred)[y == 0].mean()), 3))
    r["bal_acc"] = round((r["sens"] + r["spec"]) / 2, 3)
    results.append(r); scores[(protocol, a, b, m)] = (y, s)
    print({k: r[k] for k in ["protocol", "train", "test", "model", "auroc", "ci", "sens", "spec"]}, f"{time.time() - t0:.0f}s", flush=True)


for n in DOMAINS:
    for m in MODELS:
        s_all = np.zeros(len(Y[n])); pred = np.zeros(len(Y[n]), bool)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X[n], Y[n], G[n]):
            s, thr = run(m, X[n][tr], Y[n][tr], X[n][te]); s_all[te] = s; pred[te] = s >= thr
        record("within", n, n, m, Y[n], s_all, pred)

for a in DOMAINS:
    for b in DOMAINS:
        if a != b:
            for m in MODELS:
                s, thr = run(m, X[a], Y[a], X[b]); record("cross", a, b, m, Y[b], s, s >= thr)

for b in DOMAINS:
    src = [a for a in DOMAINS if a != b]
    xtr = np.concatenate([X[a] for a in src]); ytr = np.concatenate([Y[a] for a in src])
    for m in MODELS:
        s, thr = run(m, xtr, ytr, X[b]); record("lodo", "+".join(src), b, m, Y[b], s, s >= thr)

# ------------------------------------------------------------------ so sánh cặp
pairs = [("CBM", "CNN"), ("CBM", "ST_V2"), ("CBM", "ConceptLR"), ("ConceptLR", "CNN")]
paired = []
for (p, a, b, m), (y, s) in scores.items():
    if m != "CBM":
        continue
    for m1, m2 in pairs:
        y1, s1 = scores[(p, a, b, m1)]; _, s2 = scores[(p, a, b, m2)]
        paired.append(dict(protocol=p, train=a, test=b, cmp=f"{m1}-{m2}", **paired_delta(y1, s1, s2)))

# ------------------------------------------------------------------ dấu vân tay miền: thô vs khái niệm
xa = np.concatenate([X[n] for n in DOMAINS]); da = np.concatenate([np.full(len(X[n]), i) for i, n in enumerate(DOMAINS)])
dom = {}
for name, F in [("raw_beats", xa.reshape(len(xa), -1)), ("concepts_fixed", concepts_fixed(xa))]:
    pr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000)), F, da, cv=5, method="predict_proba")
    dom[name] = dict(acc_3class=round(float((pr.argmax(1) == da).mean()), 3),
                     auc_huca_vs_melo=round(roc_auc_score(da == 0, pr[:, 0]), 3),
                     auc_claris_vs_mortara=round(roc_auc_score(da[da > 0] == 1, pr[da > 0, 1]), 3))
print("domain:", dom)

# ------------------------------------------------------------------ nhất quán sinh lý: AUROC đơn biến từng khái niệm theo miền
uni = {}
for n in DOMAINS:
    C = concepts_fixed(X[n])
    uni[n] = {c: round(roc_auc_score(Y[n], C[:, i]), 3) for i, c in enumerate(CONCEPT_NAMES)}

# ------------------------------------------------------------------ mô hình CBM trên toàn bộ 3 miền để diễn giải
xall = np.concatenate([X[n] for n in DOMAINS]); yall = np.concatenate([Y[n] for n in DOMAINS])
_, _, models = fit_cbm(xall, yall)
interp_out = []
for mdl in models:
    w = mdl.head.weight.detach().numpy()[0]
    interp_out.append(dict(fiducials=mdl.concepts.fiducial_values(),
                           weights={c: round(float(v), 3) for c, v in zip(CONCEPT_NAMES, w)}))

out = dict(results=results, paired=paired, domain_fingerprint=dom, univariate_auc=uni, cbm_all_domains=interp_out)
json.dump(out, open(os.path.join(CD, "concept_results.json"), "w"), indent=1)
np.save(os.path.join(CD, "concept_scores.npy"), {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in scores.items()}, allow_pickle=True)
print("xong", f"{time.time() - t0:.0f}s")
