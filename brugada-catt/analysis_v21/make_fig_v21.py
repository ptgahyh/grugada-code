"""Bước 21c: hình đường đáp ứng. Chạy: python analysis_v21/make_fig_v21.py analysis_v21"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = sys.argv[1]
C = pd.read_csv(os.path.join(OUT, "response_v21.csv"))
COL = {"huca": "#2a78d6", "claris": "#eb6834", "mortara": "#1baf7a"}
NAME = {"huca": "HUCA", "claris": "Claris", "mortara": "Mortara"}
TITLE = {"P1_coved_ST": ("Coved ST elevation", "added amplitude (z units)"), "P2_T_inversion": ("T-wave inversion", "subtracted amplitude (z units)"),
         "P3_r_prime": ("r′ / J-point bump", "added amplitude (z units)"), "P4_QRS_stretch": ("QRS widening", "time-stretch factor"),
         "P5_lowpass": ("Low-pass smoothing", "Gaussian σ (ms)"), "P6_shift": ("Time shift of V1–V2", "shift (samples of 10 ms)")}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a85",
                     "axes.labelcolor": "#3d3d3a", "xtick.color": "#3d3d3a", "ytick.color": "#3d3d3a"})
fig, ax = plt.subplots(2, 3, figsize=(10, 5.6), sharey=True)
for a, (p, (ti, xl)) in zip(ax.ravel(), TITLE.items()):
    a.axhline(0, color="#c9c8c2", lw=0.6)
    for m in COL:
        for b, ls in (("huca", "-"), ("claris", "--")):
            g = C[(C.pert == p) & (C.model == m) & (C.base == b)]
            a.plot(g.level, g.mean_dlogit, color=COL[m], ls=ls, lw=1.6, marker="o", ms=3,
                   label=f"model trained on {NAME[m]}, {'HUCA' if b == 'huca' else 'Claris'} beats")
    a.set_title(ti, fontsize=9, color="#1f1f1d", loc="left"); a.set_xlabel(xl); a.grid(axis="y", color="#ecebe6", lw=0.6)
for a in ax[:, 0]:
    a.set_ylabel("Mean change in score (logit)")
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7.5, ncol=3, loc="lower center")
fig.suptitle("Response of the CNN score to synthetic waveform changes applied to non-Brugada beats (V1–V2)", fontsize=9.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.08, 1, 0.96))
fig.savefig(os.path.join(OUT, "fig_v21_response.png"), dpi=200); fig.savefig(os.path.join(OUT, "fig_v21_response.pdf"))
