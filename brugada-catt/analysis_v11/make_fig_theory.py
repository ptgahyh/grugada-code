"""Figure for step 11: (a) population bias of CATT, theory vs Monte Carlo; (b) sampling law; (c) real cohorts.
Run: python analysis_v11/make_fig_theory.py analysis_v11"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = sys.argv[1]
A = pd.read_csv(os.path.join(OUT, "theory_population.csv"))
C = pd.read_csv(os.path.join(OUT, "theory_sampling.csv"))
P = pd.read_csv(os.path.join(OUT, "theory_real_points.csv"))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
cols = {0.0: "#1f4e79", 0.3: "#4f8bc4", 0.6: "#c88a00", 0.9: "#b0413e"}
for rho, g in A.groupby("rho"):
    ax[0].plot(g["shift"], g.t_pop - g.t_star, "-", color=cols[rho], label=f"ρ = {rho:g} (theory)")
    ax[0].plot(g["shift"], g.t_mc - g.t_star, "o", color=cols[rho], ms=4)
ax[0].axhline(0, color="grey", lw=0.6)
ax[0].set_xlabel("anchor threshold shift (SD)"); ax[0].set_ylabel("CATT threshold − true Youden threshold")
ax[0].set_title("a  Population bias: Proposition 1 (lines) vs\nMonte Carlo n = 400,000 (dots)", loc="left")
ax[0].legend(frameon=False, fontsize=8)
sc = ax[1].scatter(C.loss_pred, C.loss_catt, c=C.one_minus, cmap="viridis", s=18)
lim = [C[["loss_pred", "loss_catt"]].min().min() * 0.8, C[["loss_pred", "loss_catt"]].max().max() * 1.2]
ax[1].plot(lim, lim, "--", color="grey", lw=0.8)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("predicted loss  ∝ n_eff^(−2/3)"); ax[1].set_ylabel("observed balanced-accuracy loss")
ax[1].set_title("b  Independent anchor: sampling law\n(n = 100–6,400, 600 replicates)", loc="left")
fig.colorbar(sc, ax=ax[1], label="1 − α − β")
concept = P.labeller.isin(["CBM-sd15", "CBM", "ST_V2"])
ax[2].scatter(P.shift_pred[concept], P.shift_obs[concept], s=20, color="#1f4e79", label="concept anchors")
ax[2].scatter(P.shift_pred[~concept], P.shift_obs[~concept], s=20, color="#c88a00", label="black-box anchors")
lo, hi = min(P.shift_pred.min(), P.shift_obs.min()), max(P.shift_pred.max(), P.shift_obs.max())
ax[2].plot([lo, hi], [lo, hi], "--", color="grey", lw=0.8)
r = np.corrcoef(P.shift_pred, P.shift_obs)[0, 1]
ax[2].set_xlabel("predicted shift  Γ / (D′ − Γ′)  (logit)"); ax[2].set_ylabel("observed CATT − oracle threshold (logit)")
ax[2].set_title(f"c  Real cohorts, 63 anchor × setting points\n(r = {r:.2f})", loc="left")
ax[2].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_theory.png"), dpi=200)
fig.savefig(os.path.join(OUT, "fig_theory.pdf"))
