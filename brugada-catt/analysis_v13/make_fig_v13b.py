"""Figure for the post-review checks of Proposition 5.
A: Brugada, direct spread test SD(t_hat) observed vs predicted (population a, and label-free plug-in).
B: non-Gaussian simulations, predicted/observed for spread and loss by scenario and n.
Run: python analysis_v13/make_fig_v13b.py analysis_v13"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = sys.argv[1]
B = pd.read_csv(os.path.join(OUT, "brugada_unified_v2.csv"))
G = pd.read_csv(os.path.join(OUT, "nongauss_sim.csv"))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
a = ax[0]
for d, col, lab in ((B[B.concept], "#1f5fa8", "concept anchors"), (B[~B.concept], "#e08a1e", "black-box anchors")):
    a.scatter(d.sd_obs, d["sd_pred_1.0"], s=10, c=col, alpha=0.7, label=f"{lab}: population a")
a.scatter(B.sd_obs, B.sd_plugin, s=8, facecolors="none", edgecolors="#555555", lw=0.6, alpha=0.6,
          label="label-free plug-in (one unlabelled subsample)")
lo, hi = B.sd_obs.min() * 0.6, B.sd_obs.max() * 1.8
a.plot([lo, hi], [lo, hi], "k--", lw=0.8)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
r1 = np.exp(np.median(np.log(B["sd_pred_1.0"] / B.sd_obs))); Bp = B.dropna(subset=["sd_plugin"]); r2 = np.exp(np.median(np.log(Bp.sd_plugin / Bp.sd_obs)))
a.set_title(f"A. Brugada: spread of the CATT threshold ({len(B)} points)\nmedian pred/obs {r1:.2f}; label-free {r2:.2f}",
            fontsize=9.5)
a.set_xlabel("Observed SD of t_hat over 300 subsamples (logit)"); a.set_ylabel("Predicted a * SD(Z)")
a.legend(fontsize=7.5, frameon=False, loc="upper left")
a = ax[1]
sc = ["skewed", "heavy-tail", "bimodal", "discrete"]; ns = sorted(G.n.unique())
w = 0.18
for j, (col, mk) in enumerate((("sd", "o"), ("loss", "s"))):
    for k, s in enumerate(sc):
        d = G[G.scenario == s]
        r = [np.median(d[d.n == n][f"{col}_pred"] / d[d.n == n][f"{col}_obs"]) for n in ns]
        x = np.arange(len(ns)) + (k - 1.5) * w
        a.plot(x, r, mk + ("-" if j == 0 else ":"), c=plt.cm.tab10(k), ms=5,
               label=f"{s} ({'spread' if j == 0 else 'loss'})")
a.axhline(1, c="k", lw=0.8, ls="--")
a.set_xticks(range(len(ns))); a.set_xticklabels([f"n={n}" for n in ns]); a.set_ylim(0.8, 1.3)
a.set_ylabel("Predicted / observed (median over rho, anchor shift)")
a.set_title("B. Outside the Gaussian model (400 runs per cell)", fontsize=9.5)
a.legend(fontsize=7, ncol=2, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_v13b.png"), dpi=170); fig.savefig(os.path.join(OUT, "fig_v13b.pdf"))
