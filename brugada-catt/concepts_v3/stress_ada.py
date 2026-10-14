"""Stress test cho chuẩn hóa theo miền (Ada): Ada dùng thống kê của chính lô bệnh nhân ở miền mới,
nên phụ thuộc vào (a) tỉ lệ Brugada trong lô và (b) số bệnh nhân trong lô.

Mô hình (huấn luyện 1 lần cho mỗi thiết lập, seed validation 0, 2 seed khởi tạo):
  Sign            không thích nghi (tham chiếu; sens/spec không phụ thuộc tỉ lệ bệnh)
  SignAda         chuẩn hóa theo miền bằng trung bình / độ lệch chuẩn
  SignAda-robust  chuẩn hóa theo miền bằng trung vị / IQR (huấn luyện lại với thống kê bền vững)

Thí nghiệm
  A. Tỉ lệ bệnh: lô N = 100 rút từ miền test với tỉ lệ π ∈ {5, 10, 20, 35, 50, 70}%, 30 lần lặp
  B. Cỡ lô: tỉ lệ tự nhiên, N ∈ {20, 50, 100, 200, toàn bộ}, 30 lần lặp
Mọi mô hình được đánh giá trên CÙNG các lô rút ngẫu nhiên. Ngưỡng cố định, chọn trên validation.

Chạy: python stress_ada.py <thư mục harmonized> <thư mục output> [v2|v3]
  v2 (mặc định): lớp khái niệm v2, J và QRS_w từ năng lượng 9 chuyển đạo (kết quả gốc, stress_summary.csv)
  v3: lớp khái niệm chính của bài báo (concepts_v4/concepts3.py, năng lượng V1–V3),
      ghi ra stress_raw_v3.json, stress_summary_v3.csv, stress_per_setting_v3.csv
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

HD, OUT = sys.argv[1], sys.argv[2]
CV = sys.argv[3] if len(sys.argv) > 3 else "v2"
HERE = os.path.dirname(os.path.abspath(__file__))
if CV == "v2":
    sys.path.insert(0, HERE)
    from cbm_lib import (DOMAINS, cbm_predict, load, pack, sign_gate, strat_val, train_cbm2,  # noqa: E402
                         youden)
    SUFFIX = ""
else:
    import functools
    sys.path.insert(0, os.path.join(HERE, "..", "concepts_v4"))
    import cbm_lib4 as L4  # noqa: E402
    DOMAINS, cbm_predict, load, pack, strat_val, youden = L4.DOMAINS, L4.cbm_predict, L4.load, L4.pack, L4.strat_val, L4.youden
    sign_gate = functools.partial(L4.sign_gate, cv=CV)
    train_cbm2 = functools.partial(L4.train_cbm2, cv=CV)
    SUFFIX = f"_{CV}"

X, Y = load(HD)
SETTINGS = [([a], b) for a in DOMAINS for b in DOMAINS if a != b] + [([a for a in DOMAINS if a != b], b) for b in DOMAINS]
PREVS = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70]
SIZES = [20, 50, 100, 200, None]
R = 30
METHODS = {"Sign": (False, False), "SignAda": (True, False), "SignAda-robust": (True, True)}

t0 = time.time()
rows = []
for src, tgt in SETTINGS:
    xtr, ytr, dtr, di = pack(X, Y, src)
    tr, va = strat_val(ytr, di, 0)
    signs = sign_gate(xtr[tr], ytr[tr], dtr[tr])
    models, thr = {}, {}
    for name, (ada, robust) in METHODS.items():
        ms = [train_cbm2(xtr[tr], ytr[tr], dtr[tr], xtr[va], ytr[va], dtr[va], s, ada, signs, robust=robust) for s in (0, 1)]
        sv = cbm_predict(ms, xtr[va], ada, dom=dtr[va] if ada else None)
        models[name], thr[name] = ms, youden(ytr[va], sv)
    xt, yt = X[tgt], Y[tgt]
    pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]
    rng = np.random.default_rng(123)
    batches = []
    for p in PREVS:
        n1 = int(round(100 * p)); n0 = 100 - n1
        for r in range(R):
            batches.append(("prev", p, np.concatenate([rng.choice(pos, n1, replace=False), rng.choice(neg, n0, replace=False)])))
    for n in SIZES:
        for r in range(R if n else 1):
            if n is None:
                idx = np.arange(len(yt))
            else:
                while True:
                    idx = rng.choice(len(yt), n, replace=False)
                    if 0 < yt[idx].sum() < n:
                        break
            batches.append(("size", n if n else len(yt), idx))
    # điểm không thích nghi tính một lần trên toàn miền rồi cắt theo lô
    s_static = cbm_predict(models["Sign"], xt, False)
    for kind, val, idx in batches:
        yb = yt[idx]
        for name, (ada, robust) in METHODS.items():
            s = s_static[idx] if not ada else cbm_predict(models[name], xt[idx], True)
            p = s >= thr[name]
            sens, spec = p[yb == 1].mean(), (~p)[yb == 0].mean()
            rows.append(dict(train="+".join(src), test=tgt, kind=kind, value=val, method=name,
                             auroc=roc_auc_score(yb, s), sens=sens, spec=spec, bal_acc=(sens + spec) / 2))
    print(src, "->", tgt, f"{time.time() - t0:.0f}s", flush=True)

json.dump(rows, open(os.path.join(OUT, f"stress_raw{SUFFIX}.json"), "w"))
import pandas as pd  # noqa: E402
df = pd.DataFrame(rows)
agg = df.groupby(["kind", "value", "method"])[["auroc", "sens", "spec", "bal_acc"]].mean().round(3)
per = df.groupby(["kind", "value", "method", "train", "test"])[["bal_acc"]].mean().round(3)
agg.to_csv(os.path.join(OUT, f"stress_summary{SUFFIX}.csv")); per.to_csv(os.path.join(OUT, f"stress_per_setting{SUFFIX}.csv"))
pd.set_option("display.width", 200)
print(agg.to_string())
print("xong", f"{time.time() - t0:.0f}s")
