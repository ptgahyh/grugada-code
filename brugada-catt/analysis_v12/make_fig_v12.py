"""Figure for step 12: n_eff law on three sources; threshold shift on the heart-disease data.
Run: python analysis_v12/make_fig_v12.py analysis_v12"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
OUT = sys.argv[1]
B = pd.read_csv(os.path.join(OUT, "brugada_neff.csv")); H = pd.read_csv(os.path.join(OUT, "heart_neff.csv"))
S = pd.DataFrame(json.load(open(os.path.join(OUT, "sim_matched.json")))).rename(columns={"obs_over_pred": "r"})
P = pd.read_csv(os.path.join(OUT, "heart_points.csv"))
for d in (B, H):
    d["n_eff"] = d.ratio_pred * d.n_unlabelled; d["r"] = d.ratio_obs / d.ratio_pred
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
bins = [5, 20, 40, 80, 300]
for d, col, lab, mk in ((S[~S.edge], "#555555", "simulation (independent anchor)", "s"),
                        (B[~B.at_grid_edge], "#1f4e79", "Brugada cohorts (ECG)", "o"),
                        (H[~H.at_grid_edge], "#c88a00", "UCI heart disease (3 hospitals)", "^")):
    ax[0].scatter(d.n_eff, d.r, s=10, alpha=0.25, color=col, marker=mk)
    g = d.groupby(pd.cut(d.n_eff, bins), observed=True).agg(x=("n_eff", "median"), y=("r", "median"))
    ax[0].plot(g.x, g.y, "-" + mk, color=col, label=lab, ms=6)
ax[0].axhline(1, color="grey", lw=0.8, ls="--")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_ylim(0.3, 10)
ax[0].set_xlabel("predicted n_eff (labelled-equivalent sample size)")
ax[0].set_ylabel("observed / predicted n_eff")
ax[0].set_title("a  n_eff law: matched labelled sample size\n(lines: medians by n_eff; 1 = formula exact)", loc="left")
ax[0].legend(frameon=False, fontsize=8)
dep = ~P.anchor.isin(["EX", "EX-LR"])
ax[1].scatter(P.shift_pred[~dep], P.shift_obs[~dep], color="#1f4e79", s=22, label="exercise-test anchors (other view)")
ax[1].scatter(P.shift_pred[dep], P.shift_obs[dep], color="#c88a00", s=22, label="same-variable anchors (dependent)")
ax[1].plot([-3, 3], [-3, 3], "--", color="grey", lw=0.8)
ax[1].set_xlim(-3, 3); ax[1].set_ylim(-1.5, 2.5)
nout = int((P.shift_pred < -3).sum())
ax[1].annotate(f"{nout} dependent points with predicted shift < −3 not shown", (0.02, 0.02), xycoords="axes fraction", fontsize=7.5, color="#555555")
ax[1].set_xlabel("predicted shift Γ/(D′ − Γ′)"); ax[1].set_ylabel("observed CATT − oracle threshold")
ax[1].set_title("b  UCI heart disease: threshold shift\n(9 settings × 4 anchors, 10 seeds)", loc="left")
ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_v12.png"), dpi=200); fig.savefig(os.path.join(OUT, "fig_v12.pdf"))
