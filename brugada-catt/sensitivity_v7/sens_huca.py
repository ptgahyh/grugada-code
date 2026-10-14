"""(Bước 7) Kết luận có đổi không khi HUCA được tính nhịp trung vị theo kiểu máy đo (huca_hires.npz)?

Chạy lại 7 thiết lập có HUCA (huấn luyện hoặc kiểm tra), 3 seed validation, cùng seed như bước 5:
  CNN-V1V3 (3 seed), CNN9 (9 chuyển đạo, 3 seed), CBM5-Sign (2 seed, bộ gán nhãn của CATT)
Lưu: AUROC; balanced accuracy của CNN-V1V3 với SRC / CATT / ORACLE ở tỉ lệ bệnh tự nhiên.
Chạy: python sens_huca.py <harmonized> <huca_hires.npz> <output> <chunk> <số chunk>
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "concepts_v4"))
sys.path.insert(0, os.path.join(HERE, "..", "concepts_v5"))
import cbm_lib4 as L  # noqa: E402
import concepts5  # noqa: E402

torch.set_num_threads(1)
L.CLS["v5"] = lambda learn=True: concepts5.BrugadaConcepts(learn=learn, qrs="sd15")
L.FROZEN["v5"] = L.CLS["v5"](learn=False)
HD, HIRES, OUT, CH, NCH = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
X, Y = L.load(HD)
X["huca"] = np.load(HIRES)["X"].astype(np.float32)
DOM = L.DOMAINS
SETTINGS = [([a], b) for a in DOM for b in DOM if a != b] + [([a for a in DOM if a != b], b) for b in DOM]
USE = [i for i, (s, t) in enumerate(SETTINGS) if "huca" in s or t == "huca"]
JOBS = [(i, vs) for i in USE for vs in (0, 1, 2)][CH::NCH]


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return thr[np.argmax(tpr - fpr)]


def bacc(y, p):
    return float((p[y == 1].mean() + (~p)[y == 0].mean()) / 2)


fout = os.path.join(OUT, f"sens_huca_{CH}.jsonl")
t0 = time.time()
for i, vs in JOBS:
    src, tgt = SETTINGS[i]
    xtr, ytr, dtr, di = L.pack(X, Y, src)
    tr, va = L.strat_val(ytr, di, vs)
    yv, yt = ytr[va], Y[tgt]
    nets = [L.train_cnn(xtr[tr][:, 3:6], ytr[tr], xtr[va][:, 3:6], yv, s) for s in (0, 1, 2)]
    sv, st = L.cnn_predict(nets, xtr[va][:, 3:6]), L.cnn_predict(nets, X[tgt][:, 3:6])
    n9 = [L.train_cnn(xtr[tr], ytr[tr], xtr[va], yv, s) for s in (0, 1, 2)]
    s9 = L.cnn_predict(n9, X[tgt])
    signs = L.sign_gate(xtr[tr], ytr[tr], dtr[tr], cv="v5")
    ms = [L.train_cbm2(xtr[tr], ytr[tr], dtr[tr], xtr[va], yv, dtr[va], s, False, signs, cv="v5") for s in (0, 1)]
    cv_, ct = L.cbm_predict(ms, xtr[va], False), L.cbm_predict(ms, X[tgt], False)
    t_src = youden(yv, sv)
    pl = (ct >= youden(yv, cv_)).astype(int)
    t_catt = youden(pl, st) if 0 < pl.mean() < 1 else t_src
    rec = dict(setting=i, train="+".join(src), test=tgt, vseed=vs,
               auroc={"CNN-V1V3": float(roc_auc_score(yt, st)), "CNN9": float(roc_auc_score(yt, s9)), "CBM5-Sign": float(roc_auc_score(yt, ct))},
               bacc={"SRC": bacc(yt, st >= t_src), "CATT": bacc(yt, st >= t_catt), "ORACLE": bacc(yt, st >= youden(yt, st))},
               spec_src=float((st < t_src)[yt == 0].mean()), sens_src=float((st >= t_src)[yt == 1].mean()))
    with open(fout, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(i, "+".join(src), "->", tgt, vs, {k: round(v, 3) for k, v in rec["auroc"].items()}, {k: round(v, 3) for k, v in rec["bacc"].items()},
          f"{time.time() - t0:.0f}s", flush=True)
print("xong")
