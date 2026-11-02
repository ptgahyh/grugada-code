"""Step 11: does the CATT bias theory explain the real cohorts?

For every setting (9) x validation seed (3) x pseudo-labeller (7), on the target cohort at natural prevalence:
  observed   t_CATT - t_ORACLE (logit scale of the CNN-V1V3 ensemble) and the balanced-accuracy gap
  sampling   predicted loss of an independent anchor (Proposition 3), n_eff = n pi(1-pi)(1-a-b)^2 / (q(1-q));
             the CNN's class densities are summarised by a binormal fit of its target logits
  dependence local error rates of the anchor at the CNN's own decision boundary (alpha*, beta*, kernel-weighted
             around t_ORACLE) -> Gamma = [(beta*-beta)/pi0 - (alpha*-alpha)/pi1] / (1-a-b)   (Proposition 2,
             linearised), predicted threshold shift Gamma / D'(t*), predicted loss J''/4 * shift^2
  diagnostic label-free anchor-threshold invariance (Proposition 4): the anchor's threshold is moved by
             d * SD(anchor, source validation), d in [-0.5, 0.5]; slope of the fraction of target ECGs the CNN
             flags at the CATT threshold, per unit d (no target labels used)
Gamma'(t*) is estimated by a central difference of the kernel estimates (step h/2).
Target labels are used only for the "observed" and "dependence" columns, never for the diagnostic.

Run: python analysis_v11/theory_real.py analysis_v11
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_curve

OUT = sys.argv[1]
SDIR = os.path.join(OUT, "scores")
LABS = ["CBM5", "CBM3", "ST_V2", "LogReg9", "CNN9", "ResNet", "CNNalt"]
NAMES = {"CBM5": "CBM-sd15", "CBM3": "CBM", "ST_V2": "ST_V2", "LogReg9": "LogReg9", "CNN9": "CNN9",
         "ResNet": "ResNet", "CNNalt": "CNNalt"}
DOM = ["huca", "claris", "mortara"]
SETTINGS = [([a], b) for a in DOM for b in DOM if a != b] + [([a for a in DOM if a != b], b) for b in DOM]
VAR_Z = 0.26355
DEV = (0, 1)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    thr = np.minimum(thr, s.max() + 1e-9)
    return thr[np.argmax(tpr - fpr)]


def bacc(y, s, t):
    p = s >= t
    return 0.5 * (p[y == 1].mean() + (~p)[y == 0].mean())


def binormal(f, y):
    m = [f[y == k].mean() for k in (0, 1)]; s = [f[y == k].std(ddof=1) for k in (0, 1)]
    pdf = lambda t, k: norm.pdf(t, m[k], s[k])
    lo, hi = min(m), max(m)
    try:
        ts = brentq(lambda t: pdf(t, 0) - pdf(t, 1), lo, hi)
    except ValueError:
        ts = 0.5 * (lo + hi)
    phi = pdf(ts, 0)
    d0 = -(ts - m[0]) / s[0] ** 2 * pdf(ts, 0); d1 = -(ts - m[1]) / s[1] ** 2 * pdf(ts, 1)
    j2 = abs(d0 - d1)
    dlog = (-(ts - m[1]) / s[1] ** 2) - (-(ts - m[0]) / s[0] ** 2)
    return ts, phi, j2, dlog


rows = []
for i, (src, tgt) in enumerate(SETTINGS):
    for vs in (0, 1, 2):
        Z = np.load(os.path.join(SDIR, f"lab_{i}_{vs}.npz"))
        yv, yt = Z["y_val"], Z["y_tgt"]
        fv, ft = logit(Z["f_val"]), logit(Z["f_tgt"])
        n = len(yt); pi = yt.mean(); p0, p1 = 1 - pi, pi
        t_or = youden(yt, ft)
        ba_or = bacc(yt, ft, t_or)
        _, phi, j2, dlog = binormal(ft, yt)
        h = 0.3 * np.sqrt(0.5 * (ft[yt == 0].var() + ft[yt == 1].var()))
        w = np.exp(-0.5 * ((ft - t_or) / h) ** 2)
        for lab in LABS:
            gv, gt = Z[f"{lab}_val"], Z[f"{lab}_tgt"]
            c = youden(yv, gv)
            pl = (gt >= c).astype(int)
            if pl.min() == pl.max():
                continue
            t_c = youden(pl, ft)
            a = pl[yt == 0].mean(); b = 1 - pl[yt == 1].mean(); om = 1 - a - b; q = pl.mean()
            def local(t):
                wt = np.exp(-0.5 * ((ft - t) / h) ** 2)
                return (np.sum(wt * pl * (yt == 0)) / np.sum(wt * (yt == 0)),
                        1 - np.sum(wt * pl * (yt == 1)) / np.sum(wt * (yt == 1)))

            def gam(t):
                al, bl = local(t)
                return ((bl - b) / p0 - (al - a) / p1) / om
            a_loc, b_loc = local(t_or)
            n_eff = n * p0 * p1 * om ** 2 / (q * (1 - q)) if om > 0 else np.nan
            samp = 0.25 * j2 * VAR_Z * (4 * phi / (j2 ** 2 * p0 * p1 * n_eff)) ** (2 / 3) if om > 0 else np.nan
            gamma = gam(t_or) if om > 0 else np.nan
            dgam = (gam(t_or + 0.5 * h) - gam(t_or - 0.5 * h)) / h if om > 0 else np.nan
            shift_pred = gamma / (dlog - dgam)
            bias_pred = 0.25 * j2 * shift_pred ** 2
            # label-free diagnostic: move the anchor's threshold, record the fraction the CNN flags
            sdv = gv.std(ddof=1)
            ds = np.linspace(-0.5, 0.5, 11); frac = []
            for d in ds:
                p2 = (gt >= c + d * sdv).astype(int)
                frac.append(np.mean(ft >= youden(p2, ft)) if 0 < p2.mean() < 1 else np.nan)
            frac = np.array(frac); ok = ~np.isnan(frac)
            slope = np.polyfit(ds[ok], frac[ok], 1)[0] if ok.sum() >= 3 else np.nan
            rows.append(dict(setting=i, split="dev" if i in DEV else "held-out", source="+".join(src), target=tgt,
                             vseed=vs, labeller=NAMES[lab], n=n, pi=pi, alpha=a, beta=b, one_minus=om, q=q,
                             alpha_local=a_loc, beta_local=b_loc, n_eff=n_eff, n_eff_ratio=n_eff / n,
                             shift_obs=t_c - t_or, gap=ba_or - bacc(yt, ft, t_c),
                             shift_pred=shift_pred, gamma=gamma, loss_samp_pred=samp, loss_bias_pred=bias_pred,
                             loss_pred=samp + bias_pred, frac_flagged=np.mean(ft >= t_c),
                             frac_flagged_oracle=np.mean(ft >= t_or), invariance_slope=slope))
R = pd.DataFrame(rows)
R.round(5).to_csv(os.path.join(OUT, "theory_real_raw.csv"), index=False)

# seed-averaged 63 points (labeller x setting)
P = R.groupby(["labeller", "setting", "split"], as_index=False).mean(numeric_only=True)
P["abs_slope"] = P.invariance_slope.abs()
P.round(5).to_csv(os.path.join(OUT, "theory_real_points.csv"), index=False)

# per labeller summary
S = P.groupby("labeller").agg(gap=("gap", "mean"), loss_pred=("loss_pred", "mean"), loss_samp_pred=("loss_samp_pred", "mean"),
                              loss_bias_pred=("loss_bias_pred", "mean"), one_minus=("one_minus", "mean"),
                              n_eff_ratio=("n_eff_ratio", "mean"), abs_slope=("abs_slope", "mean"),
                              shift_obs=("shift_obs", "mean"), shift_pred=("shift_pred", "mean")).reindex(list(NAMES.values()))
S.round(4).to_csv(os.path.join(OUT, "theory_real_by_labeller.csv"))
print(S.round(4).to_string())


def r2(y, X):
    X = np.column_stack([np.ones(len(y))] + X)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    return float(1 - res.var() / y.var()), beta.tolist()


ok = P.dropna(subset=["loss_pred", "shift_pred", "gap"])
summary = dict(
    n_points=len(ok),
    shift_pred_vs_obs_spearman=float(spearmanr(ok.shift_pred, ok.shift_obs)[0]),
    shift_pred_vs_obs_pearson=float(np.corrcoef(ok.shift_pred, ok.shift_obs)[0, 1]),
    shift_obs_on_pred_slope_intercept=np.polyfit(ok.shift_pred, ok.shift_obs, 1).tolist(),
    gap_concept_vs_sampling_pred=dict(gap=float(ok[ok.labeller.isin(["CBM-sd15", "CBM", "ST_V2"])].gap.mean()),
                                      sampling=float(ok[ok.labeller.isin(["CBM-sd15", "CBM", "ST_V2"])].loss_samp_pred.mean())),
    gap_blackbox_vs_sampling_pred=dict(gap=float(ok[~ok.labeller.isin(["CBM-sd15", "CBM", "ST_V2"])].gap.mean()),
                                       sampling=float(ok[~ok.labeller.isin(["CBM-sd15", "CBM", "ST_V2"])].loss_samp_pred.mean())),
    gamma_abs_vs_gap_spearman=float(spearmanr(ok.gamma.abs(), ok.gap)[0]),
    gap_vs_loss_pred_spearman=float(spearmanr(ok.loss_pred, ok.gap)[0]),
    gap_vs_loss_pred_pearson=float(np.corrcoef(ok.loss_pred, ok.gap)[0, 1]),
    r2_gap_on_samp_and_bias=r2(ok.gap.values, [ok.loss_samp_pred.values, ok.loss_bias_pred.values]),
    r2_gap_on_loss_pred=r2(ok.gap.values, [ok.loss_pred.values]),
    gap_vs_abs_slope_spearman=float(spearmanr(ok.abs_slope, ok.gap)[0]),
    abs_slope_concept_vs_blackbox=dict(concept=float(ok[ok.labeller.isin(["CBM-sd15", "CBM", "ST_V2"])].abs_slope.mean()),
                                       blackbox=float(ok[~ok.labeller.isin(["CBM-sd15", "CBM", "ST_V2"])].abs_slope.mean())),
)
json.dump(summary, open(os.path.join(OUT, "theory_real_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
