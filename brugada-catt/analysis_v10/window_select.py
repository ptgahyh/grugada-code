"""Choose the "hint" window of the beat without looking at the held-out settings.

Last week's early-ST window (QRS peak +50..+130 ms) was found after seeing all 9 settings. Here the window
is chosen on the 2 development settings only (HUCA -> Claris, HUCA -> Mortara), with a rule fixed in
advance, and then evaluated on the 7 held-out settings.

Windows are anchored at each record's soft J point (concept layer, concepts_v4/concepts3.py, validated on
LUDB), so a wide QRS does not push the window into the QRS:
    start = J + offset,  offset in {-20, -10, 0, +10, +20} ms
    length in {60, 90, 120} ms                                  -> 15 windows
plus the whole beat and last week's QRS-anchored window. Outside the window the beat is set to 0.
Model: CNN on V1-V3, 3-seed ensemble, 3 validation seeds (same protocol as the paper).

Selection rule (fixed before the run): highest mean source-threshold balanced accuracy on the 2
development settings x 3 validation seeds; ties (within 0.005) broken by higher development AUROC.
Held-out runs: the selected window, its neighbours (length +-30 ms, offset +-10 ms), the whole beat and
last week's window.

Run: python analysis_v10/window_select.py harmonized concepts_v5 analysis_v10
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "concepts_v4"))
from common import (DEV, HELD, SETTINGS, VSEEDS, ScoreStore, cbm_pseudo, cnn_predict, load, metrics, pack,  # noqa: E402
                    strat_val, train_cnn, v5_raw)
import concepts3  # noqa: E402

HD, V5, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(1)
X, Y = load(HD)
R5 = v5_raw(V5)
with torch.no_grad():
    JP = {d: np.rint(concepts3.BrugadaConcepts(learn=False).fiducials(torch.tensor(X[d]))[0].numpy()).astype(int)
          for d in X}
V13 = [3, 4, 5]
WIN = {f"J{o:+d}ms_L{L}ms": ("J", o // 10, L // 10) for L in (60, 90, 120) for o in (-20, -10, 0, 10, 20)}
WIN["whole beat"] = ("abs", 0, 75)
WIN["QRS+50ms_L90ms (last week)"] = ("abs", 33, 9)


def masked(x, j, spec):
    kind, a, L = spec
    m = np.zeros_like(x)
    for i in range(len(x)):
        s = (j[i] + a) if kind == "J" else a
        s = max(0, s); e = min(x.shape[-1], s + L)
        m[i, :, s:e] = x[i, :, s:e]
    return m


store = ScoreStore(os.path.join(OUT, "window_scores.npz"))


def run(names, ks):
    t0 = time.time()
    for k in ks:
        src, tgt = SETTINGS[k]
        xtr, ytr, dtr, di = pack(X, Y, src)
        jtr = np.concatenate([JP[d] for d in src])
        for vs in VSEEDS:
            tr, va = strat_val(ytr, di, vs)
            for name in names:
                if store.has(name, k, vs):
                    continue
                a, b = masked(xtr[:, V13], jtr, WIN[name]), masked(X[tgt][:, V13], JP[tgt], WIN[name])
                nets = [train_cnn(a[tr], ytr[tr], a[va], ytr[va], s) for s in (0, 1, 2)]
                store.put(name, k, vs, cnn_predict(nets, a[va]), cnn_predict(nets, b))
            store.save()
            print("setting", k, "vseed", vs, f"{time.time() - t0:.0f}s", flush=True)


def table(names, ks):
    rows = []
    for name in names:
        for k in ks:
            src, tgt = SETTINGS[k]
            xtr, ytr, dtr, di = pack(X, Y, src)
            for vs in VSEEDS:
                tr, va = strat_val(ytr, di, vs)
                sv, st = store.get(name, k, vs)
                rows.append(dict(window=name, k=k, vseed=vs, **metrics(ytr[va], sv, Y[tgt], st, cbm_pseudo(R5[(k, vs)]))))
    df = pd.DataFrame(rows)
    per = df.groupby(["window", "k"])[["auroc", "SRC", "CATT", "ORACLE"]].mean().reset_index()
    s = per.groupby("window")[["auroc", "SRC", "CATT", "ORACLE"]].agg(["mean", "min"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    return df, s


# ---------------- stage 1: development settings only
run(list(WIN), DEV)
dev_raw, dev = table(list(WIN), DEV)
dev = dev.sort_values(["SRC_mean", "auroc_mean"], ascending=False)
dev.round(4).to_csv(os.path.join(OUT, "window_dev_summary.csv"))
print("\nDevelopment settings (selection only):\n", dev.round(3).to_string())
cand = dev.drop(index=["whole beat", "QRS+50ms_L90ms (last week)"])
top = cand.SRC_mean.max()
chosen = cand[cand.SRC_mean >= top - 0.005].sort_values("auroc_mean", ascending=False).index[0]
_, o, L = WIN[chosen]
neigh = [n for n in (f"J{o * 10:+d}ms_L{(L - 3) * 10}ms", f"J{o * 10:+d}ms_L{(L + 3) * 10}ms",
                     f"J{(o - 1) * 10:+d}ms_L{L * 10}ms", f"J{(o + 1) * 10:+d}ms_L{L * 10}ms") if n in WIN]
held_names = [chosen] + neigh + ["whole beat", "QRS+50ms_L90ms (last week)"]
json.dump(dict(rule="max mean dev SRC balanced accuracy; ties within 0.005 -> higher dev AUROC",
               chosen=chosen, held_out_windows=held_names), open(os.path.join(OUT, "window_choice.json"), "w"), indent=1)
print("\nChosen on development settings:", chosen, "| held-out windows:", held_names, flush=True)

# ---------------- stage 2: held-out settings
for k in HELD:                       # whole beat on V1-V3 = the paper's CNN-V1V3: reuse its scores
    for vs in VSEEDS:
        r5m = R5[(k, vs)]["models"]["CNN-V1V3"]
        if not store.has("whole beat", k, vs):
            store.put("whole beat", k, vs, r5m["val"], r5m["tgt"])
run(held_names, HELD)
held_raw, held = table(held_names, HELD)
held = held.reindex(held_names)
held.round(4).to_csv(os.path.join(OUT, "window_heldout_summary.csv"))
pd.concat([dev_raw.assign(stage="dev"), held_raw.assign(stage="held")]).round(4).to_csv(
    os.path.join(OUT, "window_raw.csv"), index=False)
print("\nHeld-out settings:\n", held.round(3).to_string())
