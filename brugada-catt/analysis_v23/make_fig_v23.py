"""Bước 23: hình phổ trung vị theo bộ dữ liệu. Chạy: python analysis_v23/make_fig_v23.py analysis_v23"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = sys.argv[1]; P = pd.read_csv(os.path.join(OUT, "spectra_v23.csv"))
COL = {"huca": "#2a78d6", "claris": "#eb6834", "mortara": "#1baf7a"}
NAME = {"huca": "HUCA (median of filtered beats)", "claris": "Claris (device average beat)", "mortara": "Mortara (device average beat)"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a85",
                     "axes.labelcolor": "#3d3d3a", "xtick.color": "#3d3d3a", "ytick.color": "#3d3d3a"})
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for a, matched in zip(ax, (False, True)):
    for n, c in COL.items():
        key = f"{n}_matched" if (matched and n != "huca") else n
        g = P[P.dataset == key]
        a.semilogy(g.freq_hz, g.median_power, color=c, lw=1.8, label=NAME[n] + (" + Gaussian σ = 6 ms" if matched and n != "huca" else ""))
    a.axvspan(15, 50, color="#f1f0ec", lw=0, zorder=0); a.set_xlabel("Frequency (Hz)"); a.set_xlim(0, 50)
    a.set_title("As harmonised" if not matched else "Italian beats smoothed to match HUCA", loc="left", fontsize=9, color="#1f1f1d")
    a.legend(frameon=False, fontsize=7); a.grid(axis="y", color="#ecebe6", lw=0.6)
ax[0].set_ylabel("Median normalised power, V1–V3")
fig.suptitle("Spectrum of the representative beat by dataset (shaded: ≥ 15 Hz, used for the high-frequency share)", fontsize=9.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(os.path.join(OUT, "fig_v23_spectra.png"), dpi=200); fig.savefig(os.path.join(OUT, "fig_v23_spectra.pdf"))
