"""Step 12b: test the n_eff law (step 11, Proposition 3) directly on the Brugada cohorts.

For each setting (9), validation seed 0 and pseudo-labeller (7): the labelled sample size m* at which a Youden
threshold chosen with TRUE target labels has the same out-of-sample loss as CATT with n unlabelled target ECGs
(n = 20%, 35%, 50% of the target cohort). Subsamples are drawn without replacement and every threshold is
evaluated on the whole target cohort. Observed m*/n is compared with the predicted n_eff/n.

Run: python analysis_v12/brugada_neff.py analysis_v11 analysis_v12
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common12 import bacc, n_eff, youden  # noqa: E402

V11, OUT = sys.argv[1], sys.argv[2]
LABS = {"CBM5": "CBM-sd15", "CBM3": "CBM", "ST_V2": "ST_V2", "LogReg9": "LogReg9", "CNN9": "CNN9",
        "ResNet": "ResNet", "CNNalt": "CNNalt"}
CONCEPT = {"CBM-sd15", "CBM", "ST_V2"}
DRAWS = 300
rng = np.random.default_rng(1212)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


rows = []
for i in range(9):
    Z = np.load(os.path.join(V11, "scores", f"lab_{i}_0.npz"))
    yv, yt = Z["y_val"], Z["y_tgt"]; ft = logit(Z["f_tgt"]); N = len(yt)
    t_or = youden(yt, ft); best = bacc(yt, ft, t_or)

    def loss_curve(size, lab_all):
        out = []
        for _ in range(DRAWS):
            idx = rng.choice(N, size, replace=False)
            lab = lab_all[idx]
            if lab.min() == lab.max():
                continue
            out.append(best - bacc(yt, ft, youden(lab, ft[idx])))
        return float(np.mean(out))
    m_grid = np.unique(np.round(np.geomspace(8, N // 2, 12)).astype(int))
    lab_curve = np.array([loss_curve(m, yt) for m in m_grid])
    lx, ly = np.log(m_grid), np.log(np.maximum(lab_curve, 1e-6))
    order = np.argsort(ly)
    for k, name in LABS.items():
        gv, gt = Z[f"{k}_val"], Z[f"{k}_tgt"]
        pl = (gt >= youden(yv, gv)).astype(int)
        a = pl[yt == 0].mean(); b = 1 - pl[yt == 1].mean(); q = pl.mean()
        if 1 - a - b <= 0.05:
            continue
        r_pred = n_eff(1, yt.mean(), a, b, q)
        for fr in (0.2, 0.35, 0.5):
            n_u = int(round(fr * N))
            lc = loss_curve(n_u, pl)
            m_star = float(np.exp(np.interp(np.log(max(lc, 1e-6)), ly[order], lx[order])))
            rows.append(dict(setting=i, anchor=name, N=N, n_unlabelled=n_u, loss_catt=lc, m_star=m_star,
                             ratio_obs=m_star / n_u, ratio_pred=r_pred,
                             at_grid_edge=bool(lc >= lab_curve.max() or lc <= lab_curve.min())))
    print("setting", i, "N", N, flush=True)
M = pd.DataFrame(rows)
M.round(4).to_csv(os.path.join(OUT, "brugada_neff.csv"), index=False)
Mv = M[~M.at_grid_edge]
c = Mv.anchor.isin(CONCEPT)
summary = dict(
    concept=dict(median_obs_over_pred=float(np.median(Mv[c].ratio_obs / Mv[c].ratio_pred)),
                 iqr=np.percentile(Mv[c].ratio_obs / Mv[c].ratio_pred, [25, 75]).tolist(),
                 spearman_pred_obs=float(spearmanr(Mv[c].ratio_pred, Mv[c].ratio_obs)[0]), n=int(c.sum()),
                 median_ratio_obs=float(Mv[c].ratio_obs.median()), median_ratio_pred=float(Mv[c].ratio_pred.median())),
    blackbox=dict(median_obs_over_pred=float(np.median(Mv[~c].ratio_obs / Mv[~c].ratio_pred)),
                  iqr=np.percentile(Mv[~c].ratio_obs / Mv[~c].ratio_pred, [25, 75]).tolist(), n=int((~c).sum())),
    points_at_grid_edge=int(M.at_grid_edge.sum()),
    by_anchor=Mv.assign(r=Mv.ratio_obs / Mv.ratio_pred).groupby("anchor")[["ratio_pred", "ratio_obs", "r"]].median().round(3).to_dict(orient="index"),
)
json.dump(summary, open(os.path.join(OUT, "brugada_neff_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
