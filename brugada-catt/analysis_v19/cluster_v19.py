"""Bước 19: gom cụm dạng sóng V1–V2 (không nhãn) và tách độ trôi ngưỡng thành "thành phần" và "trong cụm". Xem PLAN_vi.md.

Chạy: python analysis_v19/cluster_v19.py <harmonized> <analysis_v8/scores> <metadata.csv của HUCA> analysis_v19 <thư mục riêng>
Nhãn cụm từng bệnh nhân ghi vào <thư mục riêng>, không vào kho.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedShuffleSplit

HD, SC, META, OUT, PRIV = sys.argv[1:6]
DOM = ["huca", "claris", "mortara"]
SETTINGS = [([a], b) for a in DOM for b in DOM if a != b] + [([a for a in DOM if a != b], b) for b in DOM]
BROKEN = [0, 1, 2, 4, 6]
SEED = 1901
Z = {n: np.load(os.path.join(HD, f"{n}.npz")) for n in DOM}
X = {n: Z[n]["X"].astype(float) for n in DOM}; Y = {n: Z[n]["y"].astype(int) for n in DOM}
FEAT = {n: X[n][:, 3:5, 24:54].reshape(len(X[n]), -1) for n in DOM}          # V1, V2; −40..+250 ms


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def youden(y, s, w=None):
    if y.min() == y.max():
        return None
    fpr, tpr, thr = roc_curve(y, s, sample_weight=w)
    return np.minimum(thr, s.max() + 1e-9)[np.argmax(tpr - fpr)]


def bacc(y, s, t):
    p = s >= t
    return 0.5 * (p[y == 1].mean() + (~p)[y == 0].mean())


def fit_clusters(K):
    rng = np.random.default_rng(SEED)
    fit = np.concatenate([FEAT[n][rng.choice(len(FEAT[n]), 350, replace=False)] for n in DOM])
    pca = PCA(8, random_state=SEED).fit(fit)
    km = KMeans(K, n_init=20, random_state=SEED).fit(pca.transform(fit))
    lab = {n: km.predict(pca.transform(FEAT[n])) for n in DOM}
    # đánh số lại cụm theo tỉ lệ Brugada gộp tăng dần, để bảng dễ đọc
    allc = np.concatenate([lab[n] for n in DOM]); ally = np.concatenate([Y[n] for n in DOM])
    order = np.argsort([ally[allc == k].mean() for k in range(K)]); remap = np.empty(K, int); remap[order] = np.arange(K)
    return {n: remap[lab[n]] for n in DOM}, float(pca.explained_variance_ratio_.sum())


def shares(c, K, w=None):
    return np.bincount(c, weights=w, minlength=K) / (len(c) if w is None else w.sum())


def load_setting(i, vs, C):
    """Trả về (f, y, cụm) cho validation nguồn và cho đích, cùng điểm mỏ neo."""
    src, tgt = SETTINGS[i]
    y = np.concatenate([Y[a] for a in src]); c = np.concatenate([C[a] for a in src])
    di = np.concatenate([np.full(len(Y[a]), k) for k, a in enumerate(src)])
    _, va = next(StratifiedShuffleSplit(1, test_size=0.15, random_state=vs).split(np.zeros(len(y)), di * 2 + y))
    r = np.load(os.path.join(SC, f"scores_{i}_{vs}.npz"))
    assert (r["y_val"] == y[va]).all() and (r["y_tgt"] == Y[tgt]).all()
    return dict(fs=logit(r["cnn_val"]), ys=y[va], cs=c[va], ft=logit(r["cnn_tgt"]), yt=Y[tgt], ct=C[tgt],
                gs=r["cbm5_val"], gt=r["cbm5_tgt"])


def decomp(fs, cs, ft, ct, K):
    """Δ = thành phần + trong cụm, cho một lớp. Trả về (Δ, thành phần, trong cụm, tỉ trọng đích ở cụm vắng nguồn)."""
    wS, wT = shares(cs, K), shares(ct, K); mS_all = fs.mean()
    mS = np.array([fs[cs == k].mean() if (cs == k).any() else mS_all for k in range(K)])
    mT = np.array([ft[ct == k].mean() if (ct == k).any() else 0.0 for k in range(K)])
    comp = float(((wT - wS) * mS).sum()); within = float((wT * (mT - mS)).sum())
    miss = float(wT[[not (cs == k).any() for k in range(K)]].sum())
    return ft.mean() - fs.mean(), comp, within, miss


def d1(C, K, cls, rng=None):
    """Một lần tính D1 cho mọi thiết lập × seed; rng != None thì lấy lại bản ghi (bootstrap)."""
    rows = []
    for i in range(9):
        for vs in range(3):
            d = load_setting(i, vs, C); s = np.where(d["ys"] == cls)[0]; t = np.where(d["yt"] == cls)[0]
            if rng is not None:
                s = rng.choice(s, len(s)); t = rng.choice(t, len(t))
            rows.append((i, vs, *decomp(d["fs"][s], d["cs"][s], d["ft"][t], d["ct"][t], K)))
    return pd.DataFrame(rows, columns=["setting", "vseed", "delta", "comp", "within", "miss"])


def pooled_share(D, sets):
    m = D[D.setting.isin(sets)].groupby("setting")[["delta", "comp", "within"]].mean()
    return float((m.within * np.sign(m.delta)).sum() / m.delta.abs().sum())


def tv(a, b):
    return 0.5 * float(np.abs(a - b).sum())


S = {}
for K in (5, 3, 8):
    C, ev = fit_clusters(K); key = f"K{K}"; S[key] = dict(pca_explained=round(ev, 3))
    # ---------- D1
    for cls, nm in ((0, "neg"), (1, "pos")):
        D = d1(C, K, cls); rng = np.random.default_rng(SEED + K + cls)
        bs_b, bs_i = [], []
        for _ in range(1000 if K == 5 else 300):
            Db = d1(C, K, cls, rng); bs_b.append(pooled_share(Db, BROKEN)); bs_i.append(pooled_share(Db, [3, 5, 7, 8]))
        S[key][f"D1_{nm}"] = dict(
            within_share_broken=round(pooled_share(D, BROKEN), 3), ci95=[round(float(v), 3) for v in np.percentile(bs_b, [2.5, 97.5])],
            within_share_intact=round(pooled_share(D, [3, 5, 7, 8]), 3),
            mean_abs_delta_broken=round(float(D[D.setting.isin(BROKEN)].groupby("setting").delta.mean().abs().mean()), 3),
            mean_abs_delta_intact=round(float(D[~D.setting.isin(BROKEN)].groupby("setting").delta.mean().abs().mean()), 3),
            max_missing_target_weight=round(float(D.miss.max()), 3))
        if K == 5:
            D.assign(cls=nm).round(4).to_csv(os.path.join(OUT, f"decomp_v19_{nm}.csv"), index=False)
    # ---------- D2
    rows = []
    for i in range(9):
        for vs in range(3):
            d = load_setting(i, vs, C); w = np.ones(len(d["ys"]))
            for cls in (0, 1):
                s = d["ys"] == cls; wS = shares(d["cs"][s], K); wT = shares(d["ct"][d["yt"] == cls], K)
                ratio = np.where(wS > 0, wT / np.where(wS > 0, wS, 1), 0.0); w[s] = ratio[d["cs"][s]]
            t_src = youden(d["ys"], d["fs"]); t_comp = youden(d["ys"], d["fs"], w); t_or = youden(d["yt"], d["ft"])
            pl = (d["gt"] >= youden(d["ys"], d["gs"])).astype(int); t_catt = youden(pl, d["ft"])
            rows.append(dict(setting=i, vseed=vs, ba_src=bacc(d["yt"], d["ft"], t_src), ba_comp=bacc(d["yt"], d["ft"], t_comp),
                             ba_catt=bacc(d["yt"], d["ft"], t_catt), ba_oracle=bacc(d["yt"], d["ft"], t_or),
                             t_src=t_src, t_comp=t_comp, t_catt=t_catt, t_oracle=t_or))
    R = pd.DataFrame(rows); m = R.groupby("setting").mean()
    b = m.loc[BROKEN]
    S[key]["D2"] = dict(gap_closed_comp=round(float(((b.ba_comp - b.ba_src) / (b.ba_oracle - b.ba_src)).mean()), 3),
                        gap_closed_catt=round(float(((b.ba_catt - b.ba_src) / (b.ba_oracle - b.ba_src)).mean()), 3),
                        ba_broken={c: round(float(b[c].mean()), 3) for c in ("ba_src", "ba_comp", "ba_catt", "ba_oracle")},
                        per_setting_gap_closed_comp={int(k): round(float(v), 3) for k, v in
                                                     ((b.ba_comp - b.ba_src) / (b.ba_oracle - b.ba_src)).items()})
    if K == 5:
        R.round(4).to_csv(os.path.join(OUT, "d2_threshold_v19.csv"), index=False)
    # ---------- D3
    rng = np.random.default_rng(SEED + 100 + K); d3 = {}
    for cls, nm in ((0, "neg"), (1, "pos")):
        idx = {n: np.where(Y[n] == cls)[0] for n in DOM}
        for a, b_ in (("huca", "claris"), ("huca", "mortara"), ("claris", "mortara")):
            pt = tv(shares(C[a][idx[a]], K), shares(C[b_][idx[b_]], K))
            bs = [tv(shares(C[a][rng.choice(idx[a], len(idx[a]))], K), shares(C[b_][rng.choice(idx[b_], len(idx[b_]))], K))
                  for _ in range(1000)]
            d3[f"{nm}_{a}_{b_}"] = dict(tv=round(pt, 3), ci95=[round(float(v), 3) for v in np.percentile(bs, [2.5, 97.5])])
    d3["criterion_met_neg"] = bool(min(d3["neg_huca_claris"]["tv"], d3["neg_huca_mortara"]["tv"]) > d3["neg_claris_mortara"]["tv"])
    S[key]["D3"] = d3
    if K != 5:
        continue
    # ---------- mô tả cụm (K = 5)
    meta = pd.read_csv(META).set_index("patient_id").loc[Z["huca"]["ids"]]
    rows = []
    for k in range(K):
        row = dict(cluster=k)
        for n in DOM:
            row[f"n_{n}"] = int((C[n] == k).sum())
            row[f"share_neg_{n}"] = round(float((C[n][Y[n] == 0] == k).mean()), 3)
            row[f"share_pos_{n}"] = round(float((C[n][Y[n] == 1] == k).mean()), 3)
            row[f"brugada_rate_{n}"] = round(float(Y[n][C[n] == k].mean()), 3) if (C[n] == k).any() else np.nan
        h = C["huca"] == k
        row["huca_basal_rate"] = round(float(meta.basal_pattern.values[h].mean()), 3) if h.any() else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "clusters_summary.csv"), index=False)
    np.savez_compressed(os.path.join(PRIV, "cluster_labels_K5.npz"), **{n: C[n] for n in DOM})
    np.savez_compressed(os.path.join(OUT, "cluster_mean_beats_K5.npz"),
                        **{f"{n}_{k}": X[n][C[n] == k][:, 3:6].mean(0) for n in DOM for k in range(K) if (C[n] == k).sum() >= 5})
    # ---------- D4
    rows = []
    for i in BROKEN:
        for vs in range(3):
            d = load_setting(i, vs, C); t_src = youden(d["ys"], d["fs"]); p = d["ft"] >= t_src
            fp = p & (d["yt"] == 0); fn = (~p) & (d["yt"] == 1)
            kind, err, base = ("FP", fp, d["yt"] == 0) if fp.mean() / max((d["yt"] == 0).mean(), 1e-9) > fn.mean() / max((d["yt"] == 1).mean(), 1e-9) \
                else ("FN", fn, d["yt"] == 1)
            for k in range(K):
                inb = base & (d["ct"] == k)
                rows.append(dict(setting=i, vseed=vs, kind=kind, cluster=k, n_class_in_cluster=int(inb.sum()),
                                 error_rate=float(err[inb].mean()) if inb.any() else np.nan,
                                 share_of_errors=float((err & (d["ct"] == k)).sum() / max(err.sum(), 1)),
                                 share_of_class=float(inb.sum() / base.sum())))
    D4 = pd.DataFrame(rows).groupby(["setting", "kind", "cluster"]).mean(numeric_only=True).drop(columns="vseed").reset_index()
    D4["enrichment"] = D4.share_of_errors / D4.share_of_class
    D4.round(3).to_csv(os.path.join(OUT, "d4_errors_by_cluster.csv"), index=False)
json.dump(S, open(os.path.join(OUT, "summary_v19.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(S, indent=1, ensure_ascii=False))
