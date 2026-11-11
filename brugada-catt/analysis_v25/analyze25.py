"""Bước 25: tính tiêu chí N1–N5, M1–M5. Chạy: python analysis_v25/analyze25.py analysis_v25"""
import json
import os
import sys

import numpy as np
import pandas as pd

OUT = sys.argv[1]; S = {}
C = pd.read_csv(os.path.join(OUT, "nongauss25_cells.csv"))
C["rmse_ratio"] = C.rmse_smooth / C.rmse_raw; C["loss_ratio"] = C.loss_smooth / C.loss_raw; C["pred_ratio"] = C.pred_exact / C.sd_smooth
E_ = pd.read_csv(os.path.join(OUT, "nongauss25_est.csv")); Nn = pd.read_csv(os.path.join(OUT, "nongauss25_n_needed.csv"))
per = {}
for sc, d in C.groupby("scenario"):
    e = E_[E_.scenario == sc]; nn = Nn[Nn.scenario == sc]
    per[sc] = dict(rmse_ratio=round(float(d.rmse_ratio.median()), 3), rmse_ratio_range=[round(float(d.rmse_ratio.min()), 3), round(float(d.rmse_ratio.max()), 3)],
                   loss_ratio=round(float(d.loss_ratio.median()), 3), loss_ratio_range=[round(float(d.loss_ratio.min()), 3), round(float(d.loss_ratio.max()), 3)],
                   pred_over_obs=round(float(d.pred_ratio.median()), 3), pred_range=[round(float(d.pred_ratio.min()), 3), round(float(d.pred_ratio.max()), 3)],
                   pred_by_n={int(k): round(float(v), 3) for k, v in d.groupby("n").pred_ratio.median().items()},
                   G1=dict(median=round(float(np.exp(np.log(e.G1 / e.sd_obs).median())), 3), abs_log=round(float(np.log(e.G1 / e.sd_obs).abs().median()), 3)),
                   G2=dict(median=round(float(np.exp(np.log(e.G2 / e.sd_obs).median())), 3), abs_log=round(float(np.log(e.G2 / e.sd_obs).abs().median()), 3)),
                   n_raw_over_smooth={str(k): round(float((g.n_raw / g.n_smooth).median()), 2) for k, g in nn.groupby("eps")},
                   N1=bool(d.rmse_ratio.median() <= 1.0), N2=bool(d.loss_ratio.median() <= 1.0), N3=bool(0.85 <= d.pred_ratio.median() <= 1.15))
S["25a"] = dict(per_scenario=per, rmse_ratio_all=round(float(C.rmse_ratio.median()), 3),
                N1_met=bool(all(v["N1"] for v in per.values()) and C.rmse_ratio.median() <= 0.85),
                N2_met=bool(all(v["N2"] for v in per.values())), N3_met=bool(all(v["N3"] for v in per.values())))
P = pd.read_csv(os.path.join(OUT, "brugada25_spread.csv")); A = pd.read_csv(os.path.join(OUT, "brugada25_acc.csv"))
G = pd.read_csv(os.path.join(OUT, "brugada25_single.csv")); Q = pd.read_csv(os.path.join(OUT, "brugada25_n_needed.csv"))
K = pd.read_csv(os.path.join(OUT, "brugada25_skipped.csv")) if os.path.getsize(os.path.join(OUT, "brugada25_skipped.csv")) > 2 else pd.DataFrame()
P["r"] = P.sd_smooth / P.sd_raw; H = A[A.setting >= 2]; lg = np.log(G.G2 / G.sd_obs); an = {}
for a in P.anchor.unique():
    h = H[H.anchor == a]; l = lg[G.anchor == a]; q = Q[Q.anchor == a]
    an[a] = dict(sd_ratio=round(float(P[P.anchor == a].r.median()), 3),
                 ba_n50=[round(float(h.ba_n50_raw.mean()), 4), round(float(h.ba_n50_smooth.mean()), 4)],
                 ba_n100=[round(float(h.ba_n100_raw.mean()), 4), round(float(h.ba_n100_smooth.mean()), 4)],
                 ba_full=[round(float(h.ba_full_raw.mean()), 4), round(float(h.ba_full_smooth.mean()), 4)],
                 G2_median=round(float(np.exp(l.median())), 3), G2_abs_log=round(float(l.abs().median()), 3),
                 n_needed_median={c: round(float(q[c].median())) for c in q.columns if c.startswith("n_")})
d50 = H.ba_n50_smooth.mean() - H.ba_n50_raw.mean(); d100 = H.ba_n100_smooth.mean() - H.ba_n100_raw.mean(); dfull = H.ba_full_smooth.mean() - H.ba_full_raw.mean()
S["25b"] = dict(per_anchor=an, skipped=len(K), sd_ratio_all=round(float(P.r.median()), 3), diff_n50=round(float(d50), 4), diff_n100=round(float(d100), 4),
                diff_full=round(float(dfull), 4), G2_all=dict(median=round(float(np.exp(lg.median())), 3), abs_log=round(float(lg.abs().median()), 3)),
                M1_met=bool(P.r.median() <= 0.85 and all(v["sd_ratio"] <= 1.0 for v in an.values())),
                M2_met=bool(d50 >= -0.005 and d100 >= -0.005), M3_met=bool(abs(dfull) <= 0.010),
                M4_met=bool(0.8 <= np.exp(lg.median()) <= 1.25 and lg.abs().median() <= 0.20))
json.dump(S, open(os.path.join(OUT, "summary_v25.json"), "w"), indent=1); print(json.dumps(S, indent=1))
