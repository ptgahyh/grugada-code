"""Bước 21: tổng hợp 21a (dịch nhịp đích), 21b (căn chung rồi huấn luyện lại), 21c (đường đáp ứng). Xem PLAN_vi.md.

Chạy: python analysis_v21/analyze_v21.py <thư mục riêng> <analysis_v8/scores hoặc ''> analysis_v21
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

PRIV, V8, OUT = sys.argv[1:4]
BROKEN = [0, 1, 2, 4, 6]; SSTAR = {0: -2, 1: -2, 2: 2, 4: 2, 6: 2}; SHIFTS = [-3, -2, -1, 0, 1, 2, 3]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return np.minimum(thr, s.max() + 1e-9)[np.argmax(tpr - fpr)]


def bacc(y, s, t):
    p = s >= t
    return 0.5 * (p[y == 1].mean() + (~p)[y == 0].mean())


rows, sh, rep = [], [], []
for v in ("orig", "realign"):
    for i in range(9):
        for vs in range(3):
            Z = np.load(os.path.join(PRIV, f"scores_{v}_{i}_{vs}.npz")); yv, yt = Z["y_val"], Z["y_tgt"]
            fv, ft = logit(Z["cnn_val"]), logit(Z["cnn_tgt"]); ts = youden(yv, fv)
            rows.append(dict(variant=v, setting=i, vseed=vs, auroc=roc_auc_score(yt, ft), ba_src=bacc(yt, ft, ts),
                             ba_oracle=bacc(yt, ft, youden(yt, ft)), delta_neg=ft[yt == 0].mean() - fv[yv == 0].mean()))
            if v == "orig":
                for s in SHIFTS:
                    f = logit(Z[f"cnn_tgt_shift{s}"])
                    sh.append(dict(setting=i, vseed=vs, shift=s, ba_src=bacc(yt, f, ts), auroc=roc_auc_score(yt, f),
                                   delta_neg=f[yt == 0].mean() - fv[yv == 0].mean()))
                if V8:
                    r = np.load(os.path.join(V8, f"scores_{i}_{vs}.npz"))
                    rep.append((float(np.abs(r["cnn_tgt"] - Z["cnn_tgt"]).max()), float(pearsonr(logit(r["cnn_tgt"]), ft)[0])))
R = pd.DataFrame(rows); H = pd.DataFrame(sh)
R.round(4).to_csv(os.path.join(OUT, "retrain_v21.csv"), index=False); H.round(4).to_csv(os.path.join(OUT, "shift_v21.csv"), index=False)
S = {}
if rep:
    S["repro_vs_step8"] = dict(max_abs_prob_diff=round(max(a for a, _ in rep), 6), min_pearson_logit=round(min(b for _, b in rep), 4))
m = R.groupby(["variant", "setting"]).mean(numeric_only=True); o, a = m.loc["orig"], m.loc["realign"]
hm = H.groupby(["setting", "shift"]).mean(numeric_only=True)
# 21a
gap = {i: float((hm.loc[(i, SSTAR[i]), "ba_src"] - o.loc[i, "ba_src"]) / (o.loc[i, "ba_oracle"] - o.loc[i, "ba_src"])) for i in BROKEN}
S["21a"] = dict(A1_gap_closed_mean=round(float(np.mean(list(gap.values()))), 3), A1_per_setting={i: round(g, 3) for i, g in gap.items()},
                A2_abs_delta_before=round(float(o.loc[BROKEN, "delta_neg"].abs().mean()), 3),
                A2_abs_delta_at_sstar=round(float(np.mean([abs(hm.loc[(i, SSTAR[i]), "delta_neg"]) for i in BROKEN])), 3),
                A3_ba_src_by_shift={i: {int(s): round(float(hm.loc[(i, s), "ba_src"]), 3) for s in SHIFTS} for i in BROKEN},
                A3_delta_by_shift={i: {int(s): round(float(hm.loc[(i, s), "delta_neg"]), 2) for s in SHIFTS} for i in BROKEN},
                auroc_by_shift={int(s): round(float(H[H["shift"] == s].auroc.mean()), 3) for s in SHIFTS})
# 21b
brk = lambda t: [int(i) for i in t.index if t.loc[i, "ba_src"] < t.loc[i, "ba_oracle"] - 0.05]
S["21b"] = dict(broken_orig=brk(o), broken_realign=brk(a), B1_broken_of_original5_after=int(sum(i in brk(a) for i in BROKEN)),
                B2_ba_src_broken5=[round(float(o.loc[BROKEN, "ba_src"].mean()), 3), round(float(a.loc[BROKEN, "ba_src"].mean()), 3)],
                B2_ba_oracle_broken5=[round(float(o.loc[BROKEN, "ba_oracle"].mean()), 3), round(float(a.loc[BROKEN, "ba_oracle"].mean()), 3)],
                B2_auroc_all9=[round(float(o.auroc.mean()), 3), round(float(a.auroc.mean()), 3)],
                B3_abs_delta_broken5=[round(float(o.loc[BROKEN, "delta_neg"].abs().mean()), 3), round(float(a.loc[BROKEN, "delta_neg"].abs().mean()), 3)],
                per_setting={int(i): dict(ba_src=[round(float(o.loc[i, "ba_src"]), 3), round(float(a.loc[i, "ba_src"]), 3)],
                                          delta=[round(float(o.loc[i, "delta_neg"]), 2), round(float(a.loc[i, "delta_neg"]), 2)]) for i in range(9)})
# 21c
fn = os.path.join(OUT, "response_v21.csv")
if os.path.exists(fn):
    C = pd.read_csv(fn); r1 = {}
    for (mo, b), g in C[C.pert == "P1_coved_ST"].groupby(["model", "base"]):
        r1[f"{mo}|{b}"] = round(float(spearmanr(g.level, g.mean_dlogit)[0]), 3)
    r2 = {}
    for b in C.base.unique():
        P = C[C.base == b].pivot_table(index=["pert", "level"], columns="model", values="mean_dlogit")
        for x, y in (("huca", "claris"), ("huca", "mortara"), ("claris", "mortara")):
            r2[f"{b}:{x}~{y}"] = round(float(pearsonr(P[x], P[y])[0]), 3)
    mx = C.assign(a=C.mean_dlogit.abs()).groupby("pert").a.max().round(2).to_dict()
    ext = C.loc[C.groupby(["pert", "model"]).level.transform(lambda s: s == s.abs().max() if s.min() < 0 else s == s.max())]
    S["21c"] = dict(R1_spearman=r1, R1_met=bool(min(r1.values()) >= 0.9), R2_model_agreement=r2, R3_max_abs_mean_dlogit=mx,
                    response_at_max_level=ext.groupby(["pert", "model"]).mean_dlogit.mean().round(2).unstack().to_dict())
json.dump(S, open(os.path.join(OUT, "summary_v21.json"), "w"), indent=1); print(json.dumps(S, indent=1))
