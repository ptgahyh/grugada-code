"""Step 13c: Brugada re-test of Proposition 5 after review (fixes and the direct test).

Changes from brugada_unified.py (v1):
  * ONE centre and ONE smoothed curve. t~, V~, A~ come from the KDE-smoothed pseudo-label Youden curve; the
    bias and the loss are computed on the KDE-smoothed balanced-accuracy curve BA_s (class-wise KDEs of f with the
    target labels, evaluation only). v1 mixed the KDE centre with the jagged empirical t_c and empirical BA,
    which made the predicted sampling term negative at 40/189 points.
  * Direct test of the core of Proposition 5, free of bias and of any BA curve: the spread of the CATT threshold
    t_hat over 300 subsamples (SD, and IQR/1.349) against the predicted a * SD(Z).
  * Label-free plug-in: a estimated from ONE unlabelled subsample of size n (f and pseudo-labels only), i.e.
    what a new site could compute before it has any labels. Median of 20 plug-ins vs the observed SD.
  * Bandwidth sensitivity: KDE bandwidth x 0.5, 1, 2 (Scott's rule = 1).
  * Chernoff samples from chernoff.py (40,000 draws, Var 0.261 vs 0.26355 theory).
  * All 3 validation seeds (27 score files), 7 anchors, n = 20/35/50% of the target cohort.
  * Finite population: sampling without replacement, n replaced by n / (1 - n/N) (heuristic correction).
Run: python analysis_v13/brugada_unified_v2.py analysis_v11 analysis_v13
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
ZS = np.load(os.path.join(OUT, "chernoff_samples.npy")).astype(float)
SDZ = float(np.sqrt(0.26355))
DRAWS, PLUG = 300, 20
rng = np.random.default_rng(1314)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def kde(x, m):
    return gaussian_kde(x, bw_method=lambda k: m * k.scotts_factor())


def pseudo_terms(f, pl, m, g):
    """t~, V~, A~ from the KDE-smoothed pseudo-label Youden curve J~ = F0 - F1 on grid g."""
    q = pl.mean(); dg = g[1] - g[0]
    d1, d0 = kde(f[pl == 1], m)(g), kde(f[pl == 0], m)(g)
    i = int(np.argmax(np.cumsum(d0 - d1)))
    i = min(max(i, 2), len(g) - 3)
    A = abs(np.gradient(d0 - d1, dg)[i])
    return g[i], d1[i] / q + d0[i] / (1 - q), A


def smooth_ba(f, y, m, g):
    dg = g[1] - g[0]
    c1 = np.cumsum(kde(f[y == 1], m)(g)) * dg; c0 = np.cumsum(kde(f[y == 0], m)(g)) * dg
    ba = 0.5 * ((c1[-1] - c1) + c0)
    return lambda t: np.interp(t, g, ba), float(ba.max())


rows = []
for i in range(9):
    for vs in range(3):
        Z = np.load(os.path.join(V11, "scores", f"lab_{i}_{vs}.npz"))
        yv, yt = Z["y_val"], Z["y_tgt"]; ft = logit(Z["f_tgt"]); N = len(yt)
        g = np.linspace(np.percentile(ft, 0.5), np.percentile(ft, 99.5), 3001)
        BA = {m: smooth_ba(ft, yt, m, g) for m in (0.5, 1.0, 2.0)}
        for k, name in LABS.items():
            gv, gt = Z[f"{k}_val"], Z[f"{k}_tgt"]
            pl = (gt >= youden(yv, gv)).astype(int)
            a_, b_ = pl[yt == 0].mean(), 1 - pl[yt == 1].mean()
            if 1 - a_ - b_ <= 0.05 or pl.sum() < 10 or (1 - pl).sum() < 10:
                continue
            terms = {m: pseudo_terms(ft, pl, m, g) for m in (0.5, 1.0, 2.0)}
            t_emp = youden(pl, ft)
            for fr in (0.2, 0.35, 0.5):
                n = int(round(fr * N)); fpc = 1 - n / N
                th = []
                for _ in range(DRAWS):
                    idx = rng.choice(N, n, replace=False)
                    if pl[idx].min() == pl[idx].max():
                        continue
                    th.append(youden(pl[idx], ft[idx]))
                th = np.array(th)
                # label-free plug-in from single subsamples (f + pseudo-labels only)
                gp = np.linspace(np.percentile(ft, 0.5), np.percentile(ft, 99.5), 1201)
                plug = []
                for _ in range(PLUG):
                    idx = rng.choice(N, n, replace=False)
                    if pl[idx].sum() < 5 or (1 - pl[idx]).sum() < 5:
                        continue
                    _, Vp, Ap = pseudo_terms(ft[idx], pl[idx], 1.0, gp)
                    plug.append((4 * Vp * fpc / (Ap ** 2 * n)) ** (1 / 3) * SDZ)
                row = dict(setting=i, vseed=vs, anchor=name, concept=name in CONCEPT, N=N, n=n,
                           sd_obs=th.std(), iqr_sd_obs=np.subtract(*np.percentile(th, [75, 25])) / 1.349,
                           mean_that=th.mean(), t_emp=t_emp, sd_plugin=float(np.median(plug)))
                for m in (0.5, 1.0, 2.0):
                    tt, V, A = terms[m]; ba, best = BA[m]
                    a = (4 * V * fpc / (A ** 2 * n)) ** (1 / 3)
                    row[f"t_kde_{m}"] = tt
                    row[f"sd_pred_{m}"] = a * SDZ
                    row[f"bias_{m}"] = best - float(ba(tt))
                    row[f"loss_pred_{m}"] = float(np.mean(best - ba(tt + a * ZS)))
                    row[f"loss_obs_{m}"] = float(np.mean(best - ba(th)))
                rows.append(row)
        print("setting", i, "seed", vs, flush=True)
D = pd.DataFrame(rows)
D.round(6).to_csv(os.path.join(OUT, "brugada_unified_v2.csv"), index=False)


def med_ratio(p, o):
    return float(np.exp(np.median(np.log(p / o))))


def block(d):
    out = dict(n_points=len(d))
    out["spread_sd"] = dict(median_pred_over_obs=med_ratio(d["sd_pred_1.0"], d.sd_obs),
                            iqr_of_ratio=np.exp(np.percentile(np.log(d["sd_pred_1.0"] / d.sd_obs), [25, 75])).tolist(),
                            spearman=float(spearmanr(d["sd_pred_1.0"], d.sd_obs)[0]))
    out["spread_robust_iqr"] = dict(median_pred_over_obs=med_ratio(d["sd_pred_1.0"], d.iqr_sd_obs),
                                    spearman=float(spearmanr(d["sd_pred_1.0"], d.iqr_sd_obs)[0]))
    out["spread_label_free_plugin"] = dict(median_pred_over_obs=med_ratio(d.sd_plugin, d.sd_obs),
                                           iqr_of_ratio=np.exp(np.percentile(np.log(d.sd_plugin / d.sd_obs), [25, 75])).tolist(),
                                           spearman=float(spearmanr(d.sd_plugin, d.sd_obs)[0]))
    for m in (0.5, 1.0, 2.0):
        lo, lp, b = d[f"loss_obs_{m}"], d[f"loss_pred_{m}"], d[f"bias_{m}"]
        ok = lo > 1e-4
        samp_o, samp_p = lo - b, lp - b
        out[f"bandwidth_{m}"] = dict(
            spread_median_pred_over_obs=med_ratio(d[f"sd_pred_{m}"], d.sd_obs),
            loss_median_pred_over_obs=med_ratio(lp[ok], lo[ok]),
            loss_spearman=float(spearmanr(lp, lo)[0]),
            bias_share_of_observed_loss_median=float(np.median((b / lo)[ok])),
            sampling_term_spearman=float(spearmanr(samp_p, samp_o)[0]),
            negative_predicted_sampling_terms=int((samp_p < 0).sum()))
    out["centre"] = dict(median_abs_mean_that_minus_t_kde=float(np.median(abs(d.mean_that - d["t_kde_1.0"]))),
                         median_abs_mean_that_minus_t_emp=float(np.median(abs(d.mean_that - d.t_emp))),
                         median_sd_obs=float(d.sd_obs.median()))
    return out


summary = {"all": block(D), "concept": block(D[D.concept]), "blackbox": block(D[~D.concept]),
           "by_anchor_spread_pred_over_obs": D.groupby("anchor").apply(
               lambda d: med_ratio(d["sd_pred_1.0"], d.sd_obs)).round(3).to_dict(),
           "by_anchor_plugin_over_obs": D.groupby("anchor").apply(
               lambda d: med_ratio(d.sd_plugin, d.sd_obs)).round(3).to_dict(),
           "by_fraction_spread_pred_over_obs": D.assign(fr=(D.n / D.N).round(2)).groupby("fr").apply(
               lambda d: med_ratio(d["sd_pred_1.0"], d.sd_obs)).round(3).to_dict()}
json.dump(summary, open(os.path.join(OUT, "brugada_unified_v2_summary.json"), "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
