"""Bước 16: kiểm công thức CATT theo độ nhạy cố định trên Brugada (theo PLAN_vi.md, commit 34c9516).

C1  TPR(t~_s) - s = FDR_g * J(t~_s) nếu mỏ neo độc lập có điều kiện (đẳng thức chính xác).
C2  SD(t_hat) = sqrt(s(1-s)/(q n)) / p~1(t~_s) * sqrt(1 - n/N);  SD(TPR(t_hat)) = f1(t~_s) * SD(t_hat).
C2' SD(t_hat) không nhãn: bootstrap B = 200 trên một mẫu cỡ n, nhân sqrt(1 - n/N), trung vị 20 mẫu.
Chạy: python analysis_v16/sens_brugada.py analysis_v11 analysis_v16
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis_v12"))
from common12 import youden  # noqa: E402

V11, OUT = sys.argv[1:3]
LABS = {"CBM5": "CBM-sd15", "CBM3": "CBM", "ST_V2": "ST_V2", "LogReg9": "LogReg9", "CNN9": "CNN9",
        "ResNet": "ResNet", "CNNalt": "CNNalt"}
CONCEPT = {"CBM-sd15", "CBM", "ST_V2"}
TARGETS = (0.80, 0.90, 0.95)
rng = np.random.default_rng(1601)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def tpr(f, y, t):
    return float(np.mean(f[y == 1] >= t))


def fpr(f, y, t):
    return float(np.mean(f[y == 0] >= t))


c1, c2 = [], []
for i in range(9):
    for vs in range(3):
        Z = np.load(os.path.join(V11, "scores", f"lab_{i}_{vs}.npz"))
        yv, yt = Z["y_val"], Z["y_tgt"]; fv, ft = logit(Z["f_val"]), logit(Z["f_tgt"]); N = len(yt)
        kde_pos = gaussian_kde(ft[yt == 1])
        for k, name in LABS.items():
            pl = (Z[f"{k}_tgt"] >= youden(yv, Z[f"{k}_val"])).astype(int)
            if pl.sum() < 20:
                continue
            q = pl.mean(); fdr = float(np.mean(yt[pl == 1] == 0))
            a_t = float(np.mean(pl[yt == 0])); b_t = float(np.mean(pl[yt == 1] == 0))
            kde_pl = gaussian_kde(ft[pl == 1])
            for s in TARGETS:
                t = float(np.quantile(ft[pl == 1], 1 - s))
                s_emp = float(np.mean(ft[pl == 1] >= t))
                T, F = tpr(ft, yt, t), fpr(ft, yt, t)
                t_src = float(np.quantile(fv[yv == 1], 1 - s))
                c1.append(dict(setting=i, vseed=vs, anchor=name, concept=name in CONCEPT, s=s, N=N, q=q,
                               fdr_anchor=fdr, alpha=a_t, beta=b_t, t=t, s_emp=s_emp, tpr=T, fpr=F,
                               pred=s + fdr * (T - F), pred_semp=s_emp + fdr * (T - F),
                               spec=1 - F, tpr_src=tpr(ft, yt, t_src), spec_src=1 - fpr(ft, yt, t_src)))
                p1, f1 = float(kde_pl(t)[0]), float(kde_pos(t)[0])
                for fr in (0.2, 0.35, 0.5):
                    n = int(round(fr * N)); fpc = 1 - n / N
                    th, se = [], []
                    for _ in range(300):
                        idx = rng.choice(N, n, replace=False)
                        fp = ft[idx][pl[idx] == 1]
                        if len(fp) < 5:
                            continue
                        u = float(np.quantile(fp, 1 - s)); th.append(u); se.append(tpr(ft, yt, u))
                    boot = []
                    for _ in range(20):
                        idx = rng.choice(N, n, replace=False)
                        fs, ps = ft[idx], pl[idx]
                        if ps.sum() < 5:
                            continue
                        bs = []
                        for _ in range(200):
                            j = rng.integers(0, n, n)
                            fp = fs[j][ps[j] == 1]
                            if len(fp) >= 3:
                                bs.append(np.quantile(fp, 1 - s))
                        boot.append(np.std(bs) * np.sqrt(fpc))
                    sd_t = np.sqrt(s * (1 - s) / (q * n)) / p1 * np.sqrt(fpc)
                    c2.append(dict(setting=i, vseed=vs, anchor=name, concept=name in CONCEPT, s=s, N=N, n=n,
                                   sd_t_obs=float(np.std(th)), sd_t_pred=sd_t, sd_t_boot=float(np.median(boot)) if boot else np.nan,
                                   sd_tpr_obs=float(np.std(se)), sd_tpr_pred=f1 * sd_t))
        print("setting", i, "seed", vs, flush=True)

A = pd.DataFrame(c1); B = pd.DataFrame(c2)
A.round(5).to_csv(os.path.join(OUT, "sens_c1.csv"), index=False)
B.round(6).to_csv(os.path.join(OUT, "sens_c2.csv"), index=False)


def mr(p, o):
    ok = np.isfinite(p) & (o > 0) & (p > 0)
    r = np.log(p[ok] / o[ok])
    return dict(median_pred_over_obs=float(np.exp(np.median(r))),
                iqr=np.exp(np.percentile(r, [25, 75])).tolist(),
                spearman=float(spearmanr(p[ok], o[ok])[0]), n=int(ok.sum()))


def c1_block(d):
    return dict(n=len(d), median_abs_err=float(np.median(abs(d.tpr - d.pred))),
                median_abs_err_semp=float(np.median(abs(d.tpr - d.pred_semp))),
                spearman_overshoot=float(spearmanr(d.pred - d.s, d.tpr - d.s)[0]),
                frac_tpr_ge_s_minus_002=float(np.mean(d.tpr >= d.s - 0.02)),
                median_overshoot_obs=float(np.median(d.tpr - d.s)), median_overshoot_pred=float(np.median(d.pred - d.s)),
                src_frac_tpr_ge_s_minus_002=float(np.mean(d.tpr_src >= d.s - 0.02)),
                median_spec_catt=float(d.spec.median()), median_spec_src=float(d.spec_src.median()))


con, bb = A[A.concept], A[~A.concept]
S1a, S1b = c1_block(con), c1_block(bb)
summary = dict(
    S1a_S1b_concept=S1a, S1c_blackbox=S1b,
    S1a_met=bool(S1a["median_abs_err"] <= 0.03 and S1a["spearman_overshoot"] >= 0.7),
    S1b_met=bool(S1a["frac_tpr_ge_s_minus_002"] >= 0.8),
    by_anchor_c1=A.groupby("anchor").apply(lambda d: pd.Series(dict(
        abs_err=np.median(abs(d.tpr - d.pred)), frac_ok=np.mean(d.tpr >= d.s - 0.02),
        overshoot_obs=np.median(d.tpr - d.s), fdr_anchor=d.fdr_anchor.median()))).round(3).to_dict(orient="index"),
    by_target_c1={str(s): c1_block(d[d.concept]) for s, d in A.groupby("s")},
    S2_tpr_spread=mr(B.sd_tpr_pred, B.sd_tpr_obs),
    S2_threshold_spread_known=mr(B.sd_t_pred, B.sd_t_obs),
    S3_threshold_spread_bootstrap=mr(B.sd_t_boot, B.sd_t_obs),
)
summary["S2_met"] = bool(0.8 <= summary["S2_tpr_spread"]["median_pred_over_obs"] <= 1.25 and summary["S2_tpr_spread"]["spearman"] >= 0.7)
summary["S3_met"] = bool(0.8 <= summary["S3_threshold_spread_bootstrap"]["median_pred_over_obs"] <= 1.25 and summary["S3_threshold_spread_bootstrap"]["spearman"] >= 0.7)
summary["S2_by_anchor"] = B.groupby("anchor").apply(lambda d: mr(d.sd_tpr_pred, d.sd_tpr_obs)["median_pred_over_obs"]).round(3).to_dict()
summary["S2_by_target"] = B.groupby("s").apply(lambda d: mr(d.sd_tpr_pred, d.sd_tpr_obs)["median_pred_over_obs"]).round(3).to_dict()
json.dump(summary, open(os.path.join(OUT, "sens_summary.json"), "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
