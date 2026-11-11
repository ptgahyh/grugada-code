"""Bước 19: hình dạng sóng trung bình theo cụm (K = 5), V1 và V2, ba bộ dữ liệu. Chạy: python analysis_v19/make_fig_v19.py analysis_v19"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = sys.argv[1]
M = np.load(os.path.join(OUT, "cluster_mean_beats_K5.npz")); T = pd.read_csv(os.path.join(OUT, "clusters_summary.csv"))
COL = {"huca": "#2a78d6", "claris": "#eb6834", "mortara": "#1baf7a"}
NAME = {"huca": "HUCA (Spain)", "claris": "Claris (Italy)", "mortara": "Mortara (Italy)"}
t = (np.arange(75) - 28) * 10
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a85",
                     "axes.labelcolor": "#3d3d3a", "xtick.color": "#3d3d3a", "ytick.color": "#3d3d3a"})
fig, ax = plt.subplots(2, 5, figsize=(11, 4.4), sharex=True, sharey=True)
for k in range(5):
    r = T.iloc[k]
    for j, lead in enumerate(("V1", "V2")):
        a = ax[j, k]; a.axvspan(-40, 250, color="#f1f0ec", lw=0); a.axhline(0, color="#c9c8c2", lw=0.6)
        for n in COL:
            if f"{n}_{k}" in M.files:
                a.plot(t, M[f"{n}_{k}"][j], color=COL[n], lw=1.6, label=NAME[n])
        a.set_xlim(-150, 400)
        if k == 0:
            a.set_ylabel(f"{lead} (z-score)")
        if j == 1:
            a.set_xlabel("Time from QRS peak (ms)")
        if j == 0:
            a.set_title(f"Cluster {k + 1}\nH {r.share_neg_huca:.0%} · C {r.share_neg_claris:.0%} · M {r.share_neg_mortara:.0%}",
                        fontsize=8, color="#1f1f1d")
h, l = ax[0, 1].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8, ncol=3, loc="lower center")
fig.suptitle("Mean median beat by label-free waveform cluster. Shaded: clustering window. Under each title: share of non-Brugada records (H HUCA, C Claris, M Mortara)", fontsize=9.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
fig.savefig(os.path.join(OUT, "fig_v19_clusters.png"), dpi=200); fig.savefig(os.path.join(OUT, "fig_v19_clusters.pdf"))
