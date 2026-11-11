"""Bước 23: tổng hợp 23b (chấm điểm lại sau đồng bộ dải thông), 23c (huấn luyện lại), 23d (σ = 20 ms cho cả ba bộ).

Chạy: python analysis_v23/analyze_v23.py <thư mục riêng> <analysis_v8/scores hoặc ''> analysis_v23
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

PRIV, V8, OUT = sys.argv[1:4]
BROKEN = [0, 1, 2, 4, 6]; FACT = [0.0, 0.5, 1.0, 1.5, 2.0]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return np.minimum(thr, s.max() + 1e-9)[np.argmax(tpr - fpr)]


def bacc(y, s, t):
    p = s >= t
    return 0.5 * (p[y == 1].mean() + (~p)[y == 0].mean())


def metrics(yv, fv, yt, ft):
    return dict(auroc=roc_auc_score(yt, ft), ba_src=bacc(yt, ft, youden(yv, fv)), ba_oracle=bacc(yt, ft, youden(yt, ft)),
                delta_neg=ft[yt == 0].mean() - fv[yv == 0].mean())


rows, tt, rep = [], [], []
for v in ("orig", "matched", "common20"):
    for i in range(9):
        for vs in range(3):
            Z = np.load(os.path.join(PRIV, f"scores_{v}_{i}_{vs}.npz")); yv, yt = Z["y_val"], Z["y_tgt"]
            rows.append(dict(variant=v, setting=i, vseed=vs, **metrics(yv, logit(Z["cnn_val"]), yt, logit(Z["cnn_tgt"]))))
            if v == "orig":
                for k in FACT:
                    kv, kt = ("cnn_val", "cnn_tgt") if k == 0 else (f"cnn_val_s{k}", f"cnn_tgt_s{k}")
                    tt.append(dict(setting=i, vseed=vs, factor=k, **metrics(yv, logit(Z[kv]), yt, logit(Z[kt]))))
                if V8:
                    r = np.load(os.path.join(V8, f"scores_{i}_{vs}.npz")); rep.append(float(np.abs(r["cnn_tgt"] - Z["cnn_tgt"]).max()))
R = pd.DataFrame(rows); T = pd.DataFrame(tt)
R.round(4).to_csv(os.path.join(OUT, "retrain_v23.csv"), index=False); T.round(4).to_csv(os.path.join(OUT, "testtime_v23.csv"), index=False)
S = {"spectra": json.load(open(os.path.join(OUT, "spectra_v23.json")))}
if rep:
    S["repro_vs_step8_max_abs_prob_diff"] = round(max(rep), 6)
m = R.groupby(["variant", "setting"]).mean(numeric_only=True); o = m.loc["orig"]
tm = T.groupby(["factor", "setting"]).mean(numeric_only=True)
gap = lambda k: {i: float((tm.loc[(k, i), "ba_src"] - o.loc[i, "ba_src"]) / (o.loc[i, "ba_oracle"] - o.loc[i, "ba_src"])) for i in BROKEN}
g1 = gap(1.0)
S["23b"] = dict(A1_gap_closed_mean=round(float(np.mean(list(g1.values()))), 3), A1_per_setting={i: round(g, 3) for i, g in g1.items()},
                A2_abs_delta=[round(float(o.loc[BROKEN, "delta_neg"].abs().mean()), 3), round(float(tm.loc[1.0].loc[BROKEN, "delta_neg"].abs().mean()), 3)],
                A3_dose={str(k): dict(gap_closed=round(float(np.mean(list(gap(k).values()))), 3),
                                      ba_src_broken5=round(float(tm.loc[k].loc[BROKEN, "ba_src"].mean()), 3),
                                      abs_delta_broken5=round(float(tm.loc[k].loc[BROKEN, "delta_neg"].abs().mean()), 3),
                                      auroc_all9=round(float(tm.loc[k].auroc.mean()), 3)) for k in FACT})
brk = lambda t: [int(i) for i in t.index if t.loc[i, "ba_src"] < t.loc[i, "ba_oracle"] - 0.05]
for v in ("matched", "common20"):
    a = m.loc[v]
    S["23c" if v == "matched" else "23d"] = dict(
        broken_after=brk(a), B1_broken_of_original5_after=int(sum(i in brk(a) for i in BROKEN)),
        B2_ba_src_broken5=[round(float(o.loc[BROKEN, "ba_src"].mean()), 3), round(float(a.loc[BROKEN, "ba_src"].mean()), 3)],
        B2_ba_oracle_broken5=[round(float(o.loc[BROKEN, "ba_oracle"].mean()), 3), round(float(a.loc[BROKEN, "ba_oracle"].mean()), 3)],
        B2_auroc_all9=[round(float(o.auroc.mean()), 3), round(float(a.auroc.mean()), 3)],
        B3_abs_delta_broken5=[round(float(o.loc[BROKEN, "delta_neg"].abs().mean()), 3), round(float(a.loc[BROKEN, "delta_neg"].abs().mean()), 3)],
        per_setting={int(i): dict(ba_src=[round(float(o.loc[i, "ba_src"]), 3), round(float(a.loc[i, "ba_src"]), 3)],
                                  ba_oracle=[round(float(o.loc[i, "ba_oracle"]), 3), round(float(a.loc[i, "ba_oracle"]), 3)],
                                  delta=[round(float(o.loc[i, "delta_neg"]), 2), round(float(a.loc[i, "delta_neg"]), 2)]) for i in range(9)})
json.dump(S, open(os.path.join(OUT, "summary_v23.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in S.items() if k != "spectra"}, indent=1, ensure_ascii=False))
