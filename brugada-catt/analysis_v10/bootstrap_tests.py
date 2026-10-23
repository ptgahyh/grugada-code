"""Patient-level bootstrap for the comparisons added in October 2026, with Holm correction.

For every replicate, the patients of each target cohort are resampled once (stratified by class) and the
same resample is used for every setting and validation seed with that target. Source-validation
thresholds stay fixed; thresholds that depend on the target batch (CATT) are re-estimated on each
replicate. Statistics are averaged over validation seeds, then over settings.

Comparisons (A - B):
  C1  cross-site AUROC, 9 settings:           CNN-V1V3            vs  Melo DNN (config picked on source validation)
  C2  CATT balanced accuracy, 7 held-out:     CNN-V1V3 + CATT     vs  Melo DNN + CATT
  C3  cross-site AUROC, 9 settings:           CNN on V1 only      vs  CNN on V1-V3
  C4  cross-site AUROC, 9 settings:           CNN on V2 only      vs  CNN on V1-V3
  C5  cross-site AUROC, 9 settings:           CNN on V1-V3        vs  CNN on 9 leads
  C6  source-threshold bal. acc., 7 held-out: dev-chosen ST window vs whole beat
  C7  CATT balanced accuracy, 7 held-out:     single-lead pipeline (V2 CNN + V2 ST labeller) vs CNN-V1V3 + CBM CATT
  C8  AUROC on HUCA, 3 Melo sources:          simulated watch at V2 (RA) vs real V2

Run: python analysis_v10/bootstrap_tests.py harmonized concepts_v5 analysis_v10 [n_boot]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HELD, SETTINGS, VSEEDS, cbm_pseudo, load, pack, strat_val, stv2, v5_raw, youden  # noqa: E402

HD, V5, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NB = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
X, Y = load(HD)
R5 = v5_raw(V5)
L = dict(np.load(os.path.join(OUT, "leads_3seed_scores.npz")))
W = dict(np.load(os.path.join(OUT, "window_scores.npz")))
M = dict(np.load(os.path.join(OUT, "melo_scores.npz")))
WA = dict(np.load(os.path.join(OUT, "watch_scores.npz")))
chosen = json.load(open(os.path.join(OUT, "window_choice.json")))["chosen"]
h, s = json.load(open(os.path.join(OUT, "melo_sweep_pick.json")))["by_source_validation"]
melo_key = f"h{h}_s{s}"


def auc(y, sc):
    r = rankdata(sc); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bacc(y, sc, t):
    p = sc >= t
    return 0.5 * (p[y == 1].mean() + (~p)[y == 0].mean())


# ---- build units: one unit = (target cohort, metric, A arrays, B arrays) per setting and validation seed
def src_val(k, vs):
    src, _ = SETTINGS[k]
    xtr, ytr, dtr, di = pack(X, Y, src)
    tr, va = strat_val(ytr, di, vs)
    return xtr, ytr, va


def unit(store, name, k, vs):
    return store[f"{name}|{k}|{vs}|val"].astype(float), store[f"{name}|{k}|{vs}|tgt"].astype(float)


def build(metric, ks, getA, getB):
    units = []
    for k in ks:
        tgt = SETTINGS[k][1]
        xtr, ytr, va = src_val(k, vs=0)
        for vs in VSEEDS:
            xtr, ytr, va = src_val(k, vs)
            units.append(dict(k=k, tgt=tgt, yv=ytr[va], A=getA(k, vs, xtr, ytr, va), B=getB(k, vs, xtr, ytr, va)))
    return dict(metric=metric, units=units)


def cnn(k, vs, *_):
    r = R5[(k, vs)]; return dict(sv=np.array(r["models"]["CNN-V1V3"]["val"]), st=np.array(r["models"]["CNN-V1V3"]["tgt"]),
                                 pseudo=cbm_pseudo(r))


def from_store(store, name, labeller="cbm"):
    def f(k, vs, xtr, ytr, va):
        sv, st = unit(store, name, k, vs)
        if labeller == "cbm":
            pl = cbm_pseudo(R5[(k, vs)])
        else:
            tgt = SETTINGS[k][1]; pl = (stv2(X[tgt]) >= youden(ytr[va], stv2(xtr[va]))).astype(int)
        return dict(sv=sv, st=st, pseudo=pl)
    return f


COMP = {
    "C1 AUROC: CNN-V1V3 vs Melo DNN": build("auroc", range(9), cnn, from_store(M, melo_key)),
    "C2 CATT: CNN-V1V3 vs Melo DNN": build("catt", HELD, cnn, from_store(M, melo_key)),
    "C3 AUROC: V1 vs V1-V3": build("auroc", range(9), from_store(L, "V1"), cnn),
    "C4 AUROC: V2 vs V1-V3": build("auroc", range(9), from_store(L, "V2"), cnn),
    "C5 AUROC: V1-V3 vs 9 leads": build("auroc", range(9), cnn, from_store(L, "9 leads")),
    f"C6 SRC: window {chosen} vs whole beat": build("src", HELD, from_store(W, chosen), from_store(W, "whole beat")),
    "C7 CATT: V2-only pipeline vs CNN-V1V3": build("catt", HELD, from_store(L, "V2", labeller="stv2"), cnn),
}
# C8: watch simulation (target HUCA, 3 sources, index 0..2)
SRCS = (["claris"], ["mortara"], ["claris", "mortara"])
u8 = []
for si, src in enumerate(SRCS):
    xtr, ytr, dtr, di = pack(X, Y, src)
    for vs in VSEEDS:
        tr, va = strat_val(ytr, di, vs)
        a = unit(WA, "Watch at V2 (RA)", si, vs); b = unit(WA, "Real V2", si, vs)
        u8.append(dict(k=si, tgt="huca", yv=ytr[va], A=dict(sv=a[0], st=a[1]), B=dict(sv=b[0], st=b[1])))
COMP["C8 AUROC on HUCA: watch at V2 vs real V2"] = dict(metric="auroc", units=u8)


def stat(c, idx):
    """mean over settings of (mean over validation seeds) of metric(A) - metric(B); idx: cohort -> indices."""
    per = {}
    for u in c["units"]:
        yt = Y[u["tgt"]]; ii = idx[u["tgt"]] if idx is not None else np.arange(len(yt)); y = yt[ii]
        vals = []
        for side in ("A", "B"):
            d = u[side]; st = d["st"][ii]
            if c["metric"] == "auroc":
                vals.append(auc(y, st))
            elif c["metric"] == "src":
                vals.append(bacc(y, st, youden(u["yv"], d["sv"])))
            else:
                pl = d["pseudo"][ii]
                f, t, th = roc_curve(pl, st); th = np.minimum(th, st.max() + 1e-9)
                vals.append(bacc(y, st, th[np.argmax(t - f)]))
        per.setdefault(u["k"], []).append(vals)
    arr = np.array([np.mean(v, 0) for v in per.values()])           # settings x 2
    return arr[:, 0].mean(), arr[:, 1].mean()


rng = np.random.default_rng(2026)
pos = {d: (np.where(Y[d] == 1)[0], np.where(Y[d] == 0)[0]) for d in Y}
rows = []
for name, c in COMP.items():
    a0, b0 = stat(c, None)
    deltas = np.empty(NB)
    for bi in range(NB):
        idx = {d: np.concatenate([rng.choice(p, len(p)), rng.choice(n, len(n))]) for d, (p, n) in pos.items()}
        a, b = stat(c, idx); deltas[bi] = a - b
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = max(2 * min((deltas <= 0).mean(), (deltas >= 0).mean()), 1 / NB)
    rows.append(dict(comparison=name, A=a0, B=b0, delta=a0 - b0, ci_lo=lo, ci_hi=hi, p_boot=p))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows)
order = np.argsort(df.p_boot.values); m = len(df); adj = np.empty(m); run = 0
for r, i in enumerate(order):                                        # Holm step-down
    run = max(run, min(1.0, (m - r) * df.p_boot.values[i])); adj[i] = run
df["p_holm"] = adj
df.round(4).to_csv(os.path.join(OUT, "bootstrap_tests.csv"), index=False)
pd.set_option("display.width", 250)
print(df.round(4).to_string(index=False))
