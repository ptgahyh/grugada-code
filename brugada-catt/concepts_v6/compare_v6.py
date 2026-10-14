"""Bước 6: (1) so CATT với các phương pháp không nhãn trong y văn ML; (2) thay mô hình khái niệm bằng
bộ gán nhãn giả "hộp đen" để kiểm tra lợi ích đến từ KHÁI NIỆM hay chỉ từ "một mô hình thứ hai".

Mô hình đích (cần đặt ngưỡng): CNN-V1V3, ensemble 3 seed — huấn luyện lại, phải trùng điểm của bước 5.

Phương pháp đặt ngưỡng / thích nghi KHÔNG dùng nhãn đích:
  SRC         ngưỡng Youden trên validation nguồn
  EM-prior    Saerens và cs. 2002: hiệu chuẩn Platt trên validation nguồn, EM ước lượng tỉ lệ bệnh đích π̂,
              hậu nghiệm đã chỉnh; ngưỡng tối ưu balanced accuracy là q ≥ π̂ (đại diện họ "label shift")
  BBSE-Q      Lipton và cs. 2018: π̂ = (q̂ − FPR)/(TPR − FPR) từ ma trận nhầm lẫn nguồn; ngưỡng tại phân vị
              (1 − π̂) của điểm đích (khớp tỉ lệ bệnh dự đoán)
  MED         chuẩn hóa trung vị/IQR của điểm đích về nguồn (như bước 5)
  AdaBN       Li và cs. 2016 / Schneider và cs. 2020: thay thống kê BatchNorm bằng thống kê của lô đích,
              giữ ngưỡng nguồn (đại diện họ "test-time adaptation")
  TENT        Wang và cs. 2021: AdaBN + cực tiểu entropy dự đoán, cập nhật tham số afin của BatchNorm
              (Adam, lr 1e-3, 10 bước toàn lô; chọn TRƯỚC, không tinh chỉnh), giữ ngưỡng nguồn
  AdaBN+CATT  AdaBN rồi đặt ngưỡng bằng CATT (CBM5)
  CATT[L]     CATT với bộ gán nhãn giả L (Youden của CNN-V1V3 theo nhãn giả của L tại ngưỡng nguồn của L):
                khái niệm:  CBM5 (CBM5-Sign, mặc định của bước 5), CBM3 (CBM3-Sign), ST_V2 (1 đặc trưng, không học)
                hộp đen:    CNN9 (CNN 9 chuyển đạo), ResNet (ResNet1D V1–V3), LogReg9 (tuyến tính trên tín hiệu thô
                            9 chuyển đạo), CNNalt (CNN-V1V3 seed 3–5: cùng kiến trúc, cùng đầu vào, khác seed)
  ORACLE      Youden dùng nhãn đích

Đo cơ chế (chỉ để phân tích, dùng nhãn đích, tỉ lệ bệnh tự nhiên), cho từng bộ gán nhãn:
  α = tỉ lệ âm tính bị gán dương, β = tỉ lệ dương tính bị gán âm, hệ số 1 − α − β (định lý Menon 2015);
  phụ thuộc có điều kiện = Spearman giữa điểm bộ gán nhãn và điểm CNN-V1V3 TRONG từng lớp thật (TB 2 lớp).

Chạy: python compare_v6.py <harmonized> <concepts_v5> <output> <chunk> <số chunk>
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "concepts_v4"))
import cbm_lib4 as L  # noqa: E402
from arch_v4_lib import ResNet1D, train_resnet  # noqa: E402

torch.set_num_threads(1)
HD, V5, OUT, CH, NCH = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
X, Y = L.load(HD)
DOM = L.DOMAINS
SETTINGS = [([a], b) for a in DOM for b in DOM if a != b] + [([a for a in DOM if a != b], b) for b in DOM]
V5ROWS = {(r["setting"], r["vseed"]): r for f in ["v5_raw_0.jsonl", "v5_raw_1.jsonl"] for r in map(json.loads, open(os.path.join(V5, f)))}
PREVS = [0.05, 0.10, 0.20, None]
NDRAW = 10
LABELERS = ["CBM5", "CBM3", "ST_V2", "CNN9", "ResNet", "LogReg9", "CNNalt"]


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    thr = np.minimum(thr, s.max() + 1e-9)
    return thr[np.argmax(tpr - fpr)]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def st_v2(x):
    return x[:, 4, 34:38].mean(1) - x[:, 4, 14:20].mean(1)


def bn_layers(net):
    return [m for m in net.modules() if isinstance(m, nn.BatchNorm1d)]


def adabn(net, x):
    """thống kê BatchNorm lấy từ lô đích x."""
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
        m.train()                                       # TENT: chuẩn hóa bằng thống kê lô khi tối ưu
    for _ in range(steps):
        p = torch.sigmoid(n(xt)).clamp(1e-6, 1 - 1e-6)
        loss = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return adabn(n, x)


def predict(nets, x):
    with torch.no_grad():
        return np.mean([torch.sigmoid(n(torch.tensor(x))).numpy() for n in nets], 0)


def em_prior(pv_cal, pt_cal, pi_s, iters=100):
    """Saerens 2002. pv_cal / pt_cal: hậu nghiệm đã hiệu chuẩn (nguồn / đích). Trả về π̂ và hậu nghiệm đã chỉnh."""
    pi = pi_s
    for _ in range(iters):
        a = pi / pi_s * pt_cal; b = (1 - pi) / (1 - pi_s) * (1 - pt_cal)
        q = a / (a + b); pi = float(np.clip(q.mean(), 1e-3, 1 - 1e-3))
    return pi, q


def bal(y, p):
    se = float(p[y == 1].mean()); sp = float((~p)[y == 0].mean())
    return se, sp, (se + sp) / 2


JOBS = [(i, vs) for i in range(len(SETTINGS)) for vs in (0, 1, 2)][CH::NCH]
fout = os.path.join(OUT, f"v6_raw_{CH}.jsonl")
done = set()
if os.path.exists(fout):
    done = {(r["setting"], r["vseed"]) for r in map(json.loads, open(fout))}
t0 = time.time()
for i, vs in JOBS:
    if (i, vs) in done:
        continue
    src, tgt = SETTINGS[i]
    xtr, ytr, dtr, di = L.pack(X, Y, src)
    tr, va = L.strat_val(ytr, di, vs)
    yv, xv, xt_full, yt_full = ytr[va], xtr[va], X[tgt], Y[tgt]
    v5 = V5ROWS[(i, vs)]
    assert v5["y_val"] == yv.tolist()
    # ---------------- mô hình đích
    nets = [L.train_cnn(xtr[tr][:, 3:6], ytr[tr], xv[:, 3:6], yv, s) for s in (0, 1, 2)]
    sv = predict(nets, xv[:, 3:6]); st_full = predict(nets, xt_full[:, 3:6])
    repro = float(np.abs(st_full - np.array(v5["models"]["CNN-V1V3"]["tgt"])).max())
    # ---------------- bộ gán nhãn giả: điểm validation (nguồn) và điểm đích
    LS = {"CBM5": (np.array(v5["models"]["CBM5-Sign"]["val"]), np.array(v5["models"]["CBM5-Sign"]["tgt"])),
          "CBM3": (np.array(v5["models"]["CBM3-Sign"]["val"]), np.array(v5["models"]["CBM3-Sign"]["tgt"])),
          "ST_V2": (st_v2(xv), st_v2(xt_full))}
    n9 = [L.train_cnn(xtr[tr], ytr[tr], xv, yv, s) for s in (0, 1, 2)]
    LS["CNN9"] = (predict(n9, xv), predict(n9, xt_full))
    rn = [train_resnet(xtr[tr][:, 3:6], ytr[tr], xv[:, 3:6], yv, s) for s in (0, 1, 2)]
    LS["ResNet"] = (predict(rn, xv[:, 3:6]), predict(rn, xt_full[:, 3:6]))
    lr9 = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[1e-4, 1e-3, 1e-2, 1e-1], cv=5, scoring="roc_auc",
                                                               class_weight="balanced", max_iter=3000))
    lr9.fit(xtr[tr].reshape(len(tr), -1), ytr[tr])
    LS["LogReg9"] = (lr9.predict_proba(xv.reshape(len(xv), -1))[:, 1], lr9.predict_proba(xt_full.reshape(len(xt_full), -1))[:, 1])
    na = [L.train_cnn(xtr[tr][:, 3:6], ytr[tr], xv[:, 3:6], yv, s) for s in (3, 4, 5)]
    LS["CNNalt"] = (predict(na, xv[:, 3:6]), predict(na, xt_full[:, 3:6]))
    LTHR = {k: youden(yv, v[0]) for k, v in LS.items()}
    # ---------------- cơ chế (tỉ lệ bệnh tự nhiên, chỉ phân tích)
    mech = {}
    for k, (lv, lt) in LS.items():
        pl = lt >= LTHR[k]
        a = float(pl[yt_full == 0].mean()); b = float((~pl)[yt_full == 1].mean())
        rho = np.mean([spearmanr(lt[yt_full == c], st_full[yt_full == c])[0] for c in (0, 1)])
        mech[k] = dict(alpha=a, beta=b, one_minus=1 - a - b, cond_rho=float(rho), auroc=float(roc_auc_score(yt_full, lt)))
    # ---------------- hiệu chuẩn Platt cho EM-prior
    platt = LogisticRegression(C=1e6).fit(logit(sv)[:, None], yv)
    pv_cal = platt.predict_proba(logit(sv)[:, None])[:, 1]
    t_src = youden(yv, sv)
    q_src_v = (sv >= t_src); tpr_s = q_src_v[yv == 1].mean(); fpr_s = q_src_v[yv == 0].mean()
    rng = np.random.default_rng(1000 * i + vs)
    res = []
    for prev in PREVS:
        for k in ([None] if prev is None else range(NDRAW)):
            if prev is None:
                idx = np.arange(len(yt_full))
            else:
                neg = np.where(yt_full == 0)[0]; pos = np.where(yt_full == 1)[0]
                npos = max(1, int(round(prev / (1 - prev) * len(neg))))
                if npos > len(pos):
                    npos = len(pos); neg = rng.choice(neg, int(round(npos * (1 - prev) / prev)), replace=False)
                idx = np.concatenate([neg, rng.choice(pos, npos, replace=False)])
            yt, xt = yt_full[idx], xt_full[idx]
            st = st_full[idx]
            P = {}                                   # phương pháp -> (điểm, ngưỡng)
            P["SRC"] = (st, t_src)
            pt_cal = platt.predict_proba(logit(st)[:, None])[:, 1]
            pi_em, q = em_prior(pv_cal, pt_cal, float(yv.mean()))
            P["EM-prior"] = (q, pi_em)
            qhat = (st >= t_src).mean()
            pi_bb = float(np.clip((qhat - fpr_s) / max(tpr_s - fpr_s, 1e-3), 0.001, 0.999))
            P["BBSE-Q"] = (st, np.quantile(st, 1 - pi_bb))
            q1v, mv, q3v = np.percentile(sv, [25, 50, 75]); q1t, mt, q3t = np.percentile(st, [25, 50, 75])
            P["MED"] = (st, mt + (t_src - mv) * (q3t - q1t) / max(q3v - q1v, 1e-9))
            s_ad = predict([adabn(n, xt[:, 3:6]) for n in nets], xt[:, 3:6])
            P["AdaBN"] = (s_ad, t_src)
            s_te = predict([tent(n, xt[:, 3:6]) for n in nets], xt[:, 3:6])
            P["TENT"] = (s_te, t_src)
            for kk in LABELERS:
                pl = (LS[kk][1][idx] >= LTHR[kk]).astype(int)
                P[f"CATT[{kk}]"] = (st, youden(pl, st) if 0 < pl.mean() < 1 else t_src)
            pl = (LS["CBM5"][1][idx] >= LTHR["CBM5"]).astype(int)
            P["AdaBN+CATT"] = (s_ad, youden(pl, s_ad) if 0 < pl.mean() < 1 else t_src)
            P["ORACLE"] = (st, youden(yt, st))
            for meth, (s, t) in P.items():
                se, sp, ba = bal(yt, s >= t)
                res.append([prev or "tự nhiên", -1 if k is None else k, meth, round(se, 4), round(sp, 4), round(ba, 4),
                            round(pi_em, 4) if meth == "EM-prior" else (round(pi_bb, 4) if meth == "BBSE-Q" else None)])
            if prev is None:
                auroc_adapt = dict(SRC=float(roc_auc_score(yt, st)), AdaBN=float(roc_auc_score(yt, s_ad)), TENT=float(roc_auc_score(yt, s_te)))
    rec = dict(setting=i, train="+".join(src), test=tgt, vseed=vs, repro_maxdiff=repro, mech=mech, auroc_adapt=auroc_adapt,
               cols=["prev", "draw", "method", "sens", "spec", "bacc", "pi_hat"], res=res)
    with open(fout, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(i, "+".join(src), "->", tgt, "vseed", vs, "repro", f"{repro:.1e}",
          {k: round(v["auroc"], 3) for k, v in mech.items()}, f"{time.time() - t0:.0f}s", flush=True)
print("xong", f"{time.time() - t0:.0f}s")
