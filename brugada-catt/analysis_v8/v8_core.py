"""(Bước 8) Huấn luyện lại CNN-V1V3 (tái lập bước 5/6) và LƯU ĐIỂM TỪNG BỆNH NHÂN cho các phương pháp đặt ngưỡng,
thêm hai phương pháp thích nghi miền lúc huấn luyện (dùng ECG đích KHÔNG nhãn khi train):

  base   CNN-V1V3, 3 seed (phải trùng điểm bước 5)          -> điểm val nguồn, điểm đích (SRC, CATT, MED...)
  AdaBN  thống kê BatchNorm từ lô đích đầy đủ                -> điểm đích
  TENT   AdaBN + entropy min 10 bước (như bước 6)             -> điểm đích
  DANN   Ganin & Lempitsky 2015: lớp đảo gradient + bộ phân biệt miền trên đặc trưng 64 chiều,
         λ theo lịch 2/(1+exp(−10p)) − 1, phân biệt nguồn với đích (ECG đích không nhãn)
  CORAL  Sun & Saenko 2016 (Deep CORAL): phạt khoảng cách hiệp phương sai đặc trưng nguồn–đích, λ = 1
  Cả DANN và CORAL: cùng kiến trúc, tăng cường, lịch học, dừng sớm theo AUROC validation NGUỒN như base (không nhìn
  nhãn đích); siêu tham số chọn TRƯỚC theo bài gốc, không tinh chỉnh.

Đầu ra: <out>/scores_<setting>_<vseed>.npz với y_val, y_tgt, và điểm val/tgt của từng mô hình.
Chạy: python v8_core.py <harmonized> <concepts_v5> <out> <chunk> <số chunk>
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
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "concepts_v4"))
import cbm_lib4 as L  # noqa: E402

torch.set_num_threads(1)
HD, V5, OUT, CH, NCH = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
X, Y = L.load(HD)
DOM = L.DOMAINS
SETTINGS = [([a], b) for a in DOM for b in DOM if a != b] + [([a for a in DOM if a != b], b) for b in DOM]
V5ROWS = {(r["setting"], r["vseed"]): r for f in ["v5_raw_0.jsonl", "v5_raw_1.jsonl"]
          for r in map(json.loads, open(os.path.join(V5, f)))}


def predict(nets, x):
    with torch.no_grad():
        return np.mean([torch.sigmoid(n(torch.tensor(x))).numpy() for n in nets], 0)


def bn_layers(net):
    return [m for m in net.modules() if isinstance(m, nn.BatchNorm1d)]


def adabn(net, x):
    n = copy.deepcopy(net); n.eval()
    for m in bn_layers(n):
        m.reset_running_stats(); m.momentum = None; m.train()
    with torch.no_grad():
        n(torch.tensor(x))
    n.eval()
    return n


def tent(net, x, steps=10, lr=1e-3):
    n = adabn(net, x)
    params = [p for m in bn_layers(n) for p in (m.weight, m.bias)]
    for p in n.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    xt = torch.tensor(x)
    for m in bn_layers(n):
        m.train()
    for _ in range(steps):
        p = torch.sigmoid(n(xt)).clamp(1e-6, 1 - 1e-6)
        loss = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return adabn(n, x)


class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def coral(fs, ft):
    d = fs.shape[1]
    cs = torch.cov(fs.T); ct = torch.cov(ft.T)
    return ((cs - ct) ** 2).sum() / (4 * d * d)


def train_uda(xtr, ytr, xva, yva, xtg, seed, mode, epochs=60, patience=12):
    """Như L.train_cnn, thêm nhánh thích nghi miền. mode ∈ {'dann', 'coral'}."""
    TG = torch.Generator()
    torch.manual_seed(seed); TG.manual_seed(seed); g = np.random.default_rng(seed)
    net = L.Net(xtr.shape[1])
    disc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    params = list(net.parameters()) + (list(disc.parameters()) if mode == "dann" else [])
    opt = torch.optim.Adam(params, 1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ytr == 0).sum() / max(1, (ytr == 1).sum()), dtype=torch.float32))
    xt, yt, xv, xg = torch.tensor(xtr), torch.tensor(ytr, dtype=torch.float32), torch.tensor(xva), torch.tensor(xtg)
    nb = int(np.ceil(len(xt) / 64))
    best, state, wait = -1, None, 0
    for ep in range(epochs):
        net.train(); disc.train(); perm = g.permutation(len(xt))
        for bi, i in enumerate(range(0, len(perm), 64)):
            p = (ep * nb + bi) / (epochs * nb)
            lam = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            b = perm[i:i + 64]
            tb = g.choice(len(xg), size=len(b), replace=len(xg) < len(b))
            sh = int(g.integers(-2, 3))
            xb = torch.roll(xt[b], sh, dims=2)
            xb = xb * (1 + 0.1 * (torch.rand(xb.shape[0], xb.shape[1], 1, generator=TG) - 0.5) * 2)
            xb = xb + 0.05 * torch.randn(xb.shape, generator=TG)
            xgb = xg[tb]
            fs = net.f(xb).flatten(1); ft = net.f(xgb).flatten(1)
            loss = lossf(net.h(fs).squeeze(1), yt[b])
            if mode == "dann":
                f_all = torch.cat([fs, ft]); d_all = torch.cat([torch.zeros(len(fs)), torch.ones(len(ft))])
                loss = loss + F.binary_cross_entropy_with_logits(disc(GradRev.apply(f_all, lam)).squeeze(1), d_all)
            else:
                loss = loss + 1.0 * coral(fs, ft)
            opt.zero_grad(); loss.backward(); opt.step()
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


JOBS = [(i, vs) for i in range(len(SETTINGS)) for vs in (0, 1, 2)][CH::NCH]
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
for i, vs in JOBS:
    fn = os.path.join(OUT, f"scores_{i}_{vs}.npz")
    if os.path.exists(fn):
        continue
    src, tgt = SETTINGS[i]
    xtr, ytr, dtr, di = L.pack(X, Y, src)
    tr, va = L.strat_val(ytr, di, vs)
    yv, xv, xt, yt = ytr[va], xtr[va], X[tgt], Y[tgt]
    v5 = V5ROWS[(i, vs)]
    assert v5["y_val"] == yv.tolist()
    a, b = xtr[tr][:, 3:6], ytr[tr]
    nets = [L.train_cnn(a, b, xv[:, 3:6], yv, s) for s in (0, 1, 2)]
    sv = predict(nets, xv[:, 3:6]); st = predict(nets, xt[:, 3:6])
    repro = float(np.abs(st - np.array(v5["models"]["CNN-V1V3"]["tgt"])).max())
    s_ad = predict([adabn(n, xt[:, 3:6]) for n in nets], xt[:, 3:6])
    s_te = predict([tent(n, xt[:, 3:6]) for n in nets], xt[:, 3:6])
    dn = [train_uda(a, b, xv[:, 3:6], yv, xt[:, 3:6], s, "dann") for s in (0, 1, 2)]
    co = [train_uda(a, b, xv[:, 3:6], yv, xt[:, 3:6], s, "coral") for s in (0, 1, 2)]
    out = dict(y_val=yv, y_tgt=yt, dom_val=di[va],
               cnn_val=sv, cnn_tgt=st, adabn_tgt=s_ad, tent_tgt=s_te,
               dann_val=predict(dn, xv[:, 3:6]), dann_tgt=predict(dn, xt[:, 3:6]),
               coral_val=predict(co, xv[:, 3:6]), coral_tgt=predict(co, xt[:, 3:6]),
               cbm5_val=np.array(v5["models"]["CBM5-Sign"]["val"]), cbm5_tgt=np.array(v5["models"]["CBM5-Sign"]["tgt"]),
               cbm3_val=np.array(v5["models"]["CBM3-Sign"]["val"]), cbm3_tgt=np.array(v5["models"]["CBM3-Sign"]["tgt"]),
               stv2_val=xv[:, 4, 34:38].mean(1) - xv[:, 4, 14:20].mean(1),
               stv2_tgt=xt[:, 4, 34:38].mean(1) - xt[:, 4, 14:20].mean(1), repro=repro)
    np.savez_compressed(fn, **out)
    print(i, "+".join(src), "->", tgt, vs, f"repro {repro:.1e}",
          "AUROC cnn/dann/coral", round(roc_auc_score(yt, st), 3), round(roc_auc_score(yt, out["dann_tgt"]), 3),
          round(roc_auc_score(yt, out["coral_tgt"]), 3), f"{time.time() - t0:.0f}s", flush=True)
print("xong", f"{time.time() - t0:.0f}s")
