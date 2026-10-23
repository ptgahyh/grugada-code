"""Shared helpers for step 10 (robustness of the week-of-17-Oct results)."""
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "concepts_v4"))
from cbm_lib4 import DOMAINS, cnn_predict, load, pack, strat_val, train_cnn  # noqa: E402,F401

SETTINGS = [([a], b) for a in DOMAINS for b in DOMAINS if a != b] + \
           [([a for a in DOMAINS if a != b], b) for b in DOMAINS]
DEV = [0, 1]                      # HUCA -> Claris, HUCA -> Mortara: used for design decisions only
HELD = [2, 3, 4, 5, 6, 7, 8]
VSEEDS = (0, 1, 2)


def youden(y, s):
    f, t, th = roc_curve(y, s)
    th = np.minimum(th, s.max() + 1e-9)
    return th[np.argmax(t - f)]


def bacc(y, s, t):
    p = s >= t
    return 0.5 * (p[y == 1].mean() + (~p)[y == 0].mean())


def stv2(x9):
    """ST level at V2 (60-90 ms after the QRS peak) minus PR baseline, on the 9-lead beat."""
    return x9[:, 4, 34:38].mean(1) - x9[:, 4, 14:20].mean(1)


def v5_raw(v5_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(v5_dir, "v5_raw_*.jsonl"))):
        for line in open(f):
            r = json.loads(line); out[(r["setting"], r["vseed"])] = r
    return out


def cbm_pseudo(r5):
    """CATT pseudo-labels of the paper: CBM-sd15 (CBM5-Sign) at its own source-validation Youden threshold."""
    yv = np.array(r5["y_val"])
    cv, ct = np.array(r5["models"]["CBM5-Sign"]["val"]), np.array(r5["models"]["CBM5-Sign"]["tgt"])
    return (ct >= youden(yv, cv)).astype(int)


def metrics(yv, sv, yt, st, pseudo=None):
    out = dict(auroc=roc_auc_score(yt, st), SRC=bacc(yt, st, youden(yv, sv)), ORACLE=bacc(yt, st, youden(yt, st)))
    if pseudo is not None:
        out["CATT"] = bacc(yt, st, youden(pseudo, st))
    return out


class ScoreStore:
    """Per-patient scores, keyed name|setting|vseed|{val,tgt}; saved as one compressed npz."""

    def __init__(self, path):
        self.path = path
        self.d = dict(np.load(path)) if os.path.exists(path) else {}

    def has(self, name, k, vs):
        return f"{name}|{k}|{vs}|tgt" in self.d

    def put(self, name, k, vs, sv, st):
        self.d[f"{name}|{k}|{vs}|val"] = np.asarray(sv, np.float32)
        self.d[f"{name}|{k}|{vs}|tgt"] = np.asarray(st, np.float32)

    def get(self, name, k, vs):
        return self.d[f"{name}|{k}|{vs}|val"], self.d[f"{name}|{k}|{vs}|tgt"]

    def save(self):
        np.savez_compressed(self.path, **self.d)
