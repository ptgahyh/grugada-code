"""Bước 24: tính tiêu chí T1–T6 từ các tệp kết quả. Chạy: python analysis_v24/analyze24.py analysis_v24"""
import json
import os
import sys

import numpy as np
import pandas as pd

OUT = sys.argv[1]; S = {}
C = pd.read_csv(os.path.join(OUT, "sim24_cells.csv")); C["r_exact"] = C.pred_exact / C.sd_obs; C["r_closed"] = C.pred_closed / C.sd_obs
byn = C.groupby("n").r_exact.median()
S["T1"] = dict(median_exact=round(float(C.r_exact.median()), 3), by_n={int(k): round(float(v), 3) for k, v in byn.items()},
               range_exact=[round(float(C.r_exact.min()), 3), round(float(C.r_exact.max()), 3)],
               median_closed=round(float(C.r_closed.median()), 3),
               closed_by_n={int(k): round(float(v), 3) for k, v in C.groupby("n").r_closed.median().items()},
               slope_log_sd_log_n=round(float(np.mean([np.polyfit(np.log(g.n), np.log(g.sd_obs), 1)[0] for _, g in C.groupby(["rho", "shift"])])), 3),
               bias_over_sd_range=[round(float(v), 3) for v in ((C.mean_t - C.t_tilde) / C.sd_obs).agg(["min", "max"])],
               met=bool(0.9 <= C.r_exact.median() <= 1.1 and byn.between(0.85, 1.15).all()))
D = pd.read_csv(os.path.join(OUT, "sim24_estimators.csv")); G = ["G1_closed", "G2_sandwich", "G3_hybrid", "G4_boot"]
r = np.log(D[G].div(D.sd_obs, axis=0)); t2 = {}
for g in G:
    t2[g] = dict(median=round(float(np.exp(r[g].median())), 3), abs_log=round(float(r[g].abs().median()), 3),
                 by_n={int(k): round(float(v), 3) for k, v in np.exp(r[g].groupby(D.n).median()).items()},
                 abs_log_by_n={int(k): round(float(v), 3) for k, v in r[g].abs().groupby(D.n).median().items()},
                 ok=bool(r[g].abs().median() <= 0.15 and np.exp(r[g].groupby(D.n).median()).between(0.8, 1.25).all()))
S["T2"] = dict(estimators=t2, met=bool(any(t2[g]["ok"] for g in G[:3])))
F = pd.read_csv(os.path.join(OUT, "fpc24.csv")); x = np.log(1 - F.frac.values); y = np.log(F.R.values)
gam = (x * y).sum() / (x * x).sum(); rng = np.random.default_rng(0)
bs = [(x[i] * y[i]).sum() / (x[i] * x[i]).sum() for i in (rng.integers(0, len(x), len(x)) for _ in range(2000))]
S["T3"] = dict(gamma=round(float(gam), 3), ci95=[round(float(v), 3) for v in np.percentile(bs, [2.5, 97.5])],
               R_by_frac={str(k): round(float(v), 3) for k, v in F.groupby("frac").R.median().items()}, met=bool(0.4 <= gam <= 0.6))
P = pd.read_csv(os.path.join(OUT, "plan24_plans.csv")); ok = P.groupby(["m", "eps"]).ok.mean()
S["T4"] = dict(share_ok={f"m{m}_eps{e}": round(float(v), 3) for (m, e), v in ok.items()},
               median_n_plan={f"m{m}_eps{e}": round(float(v)) for (m, e), v in P.groupby(["m", "eps"]).n_plan.median().items()},
               met=bool(ok.loc[300].min() >= 0.8))
Q = pd.read_csv(os.path.join(OUT, "plan24_pop_n.csv")); Q["ratio"] = Q.n_raw_pop / Q.n_smooth_pop
S["T4_pop_n"] = Q.round(1).to_dict("records")
B = pd.read_csv(os.path.join(OUT, "brugada24_single.csv")); rb = np.log(B[G[:3]].div(B.sd_obs, axis=0)); t5 = {}
for g in G[:3]:
    t5[g] = dict(median=round(float(np.exp(rb[g].median())), 3), iqr=[round(float(np.exp(rb[g].quantile(q))), 3) for q in (.25, .75)],
                 abs_log=round(float(rb[g].abs().median()), 3), ok=bool(0.8 <= np.exp(rb[g].median()) <= 1.25 and rb[g].abs().median() <= 0.20))
S["T5"] = dict(estimators=t5, met=bool(any(v["ok"] for v in t5.values())), points=len(B))
N = pd.read_csv(os.path.join(OUT, "brugada24_n_needed.csv"))
S["T6"] = {c: round(float(N[c].median())) for c in N.columns if c.startswith("n_")}
S["T6"]["ratio_raw_over_smooth_median"] = {e: round(float((N[f"n_raw_{e}"] / N[f"n_smooth_{e}"]).median()), 2) for e in ("0.15", "0.3")}
json.dump(S, open(os.path.join(OUT, "summary_v24.json"), "w"), indent=1); print(json.dumps(S, indent=1))
