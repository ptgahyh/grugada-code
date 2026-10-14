"""(Bước 8) Phân tích trên điểm từng bệnh nhân do v8_core.py lưu.

1. Bootstrap theo bệnh nhân (ghép cặp) cho CATT so với từng phương pháp, từng thiết lập và gộp 7 thiết lập held-out.
   Ngưỡng nào phụ thuộc lô đích (CATT, MED, GMM, Otsu, ORACLE) được TÍNH LẠI trong mỗi lần lấy mẫu lại.
   AdaBN/TENT/DANN/CORAL: điểm lấy từ lô đích đầy đủ (không huấn luyện lại trong mỗi lần bootstrap; ghi rõ trong bài).
2. Tỉ lệ bệnh 5%: phân bố hiệu số gộp qua các lần rút lô (mỗi lần rút một lô cho mỗi thiết lập).
3. Cần bao nhiêu ECG đích không nhãn? Chọn ngưỡng trên n ECG đích ngẫu nhiên, đánh giá trên TOÀN BỘ cohort đích.
4. Ngưỡng không giám sát trên chính điểm của f: GMM hai thành phần (giao điểm mật độ), Otsu.
5. Biến thể thiết kế CATT: nhãn mềm, chỉ dùng nhãn giả chắc chắn, bộ gán nhãn ghép (CBM-sd15 + ST_V2).
6. Thích nghi lúc huấn luyện: DANN, Deep CORAL (ngưỡng nguồn, và CATT áp lên chúng).
Chạy: python v8_analyze.py <thư mục scores> <out>
"""
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")
SD, OUT = sys.argv[1], sys.argv[2]
B = int(os.environ.get("NBOOT", 1000))
NAMES = ["HUCA→Claris", "HUCA→Mortara", "Claris→HUCA", "Claris→Mortara", "Mortara→HUCA", "Mortara→Claris",
         "Claris+Mortara→HUCA", "HUCA+Mortara→Claris", "HUCA+Claris→Mortara"]
HELD = [2, 3, 4, 5, 6, 7, 8]
R = {}
for f in sorted(glob.glob(os.path.join(SD, "scores_*.npz"))):
    i, vs = map(int, os.path.basename(f)[7:-4].split("_"))
    R[(i, vs)] = dict(np.load(f))
print("loaded", len(R), "jobs; max repro diff", max(float(r["repro"]) for r in R.values()))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def youden(y, s, w1=None, w0=None):
    """Youden; nếu có trọng số mềm w1 (dương) / w0 (âm) thì TPR/FPR có trọng số."""
    if w1 is None:
        if y.min() == y.max():
            return None
        fpr, tpr, thr = roc_curve(y, s)
        thr = np.minimum(thr, s.max() + 1e-9)
        return thr[np.argmax(tpr - fpr)]
    o = np.argsort(-s); ss = s[o]
    tp = np.cumsum(w1[o]) / w1.sum(); fp = np.cumsum(w0[o]) / w0.sum()
    k = np.argmax(tp - fp)
    return ss[k]


def bacc(y, s, t):
    p = s >= t
    return 0.5 * (p[y == 1].mean() + (~p)[y == 0].mean())


def gmm_thr(s):
    z = logit(s)[:, None]
    g = GaussianMixture(2, random_state=0, n_init=1, max_iter=200).fit(z)
    m = g.means_.ravel(); sd = np.sqrt(g.covariances_.ravel())
    lo, hi = np.argsort(m)
    grid = np.linspace(m[lo], m[hi], 400)
    d = norm.pdf(grid, m[hi], sd[hi]) - norm.pdf(grid, m[lo], sd[lo])
    k = np.where(np.diff(np.sign(d)) != 0)[0]
    zt = grid[k[0]] if len(k) else (m[lo] + m[hi]) / 2
    return 1 / (1 + np.exp(-zt))


def otsu_thr(s):
    z = logit(s); h, e = np.histogram(z, 64); c = (e[:-1] + e[1:]) / 2
    w = np.cumsum(h); w2 = w[-1] - w
    m1 = np.cumsum(h * c) / np.maximum(w, 1); m2 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w2, 1)
    k = np.argmax(w * w2 * (m1 - m2) ** 2)
    return 1 / (1 + np.exp(-c[k]))


def thresholds(r, idx):
    """Mọi ngưỡng cho lô đích idx (chỉ số vào cohort đích). Trả về {phương pháp: (điểm đích đầy đủ, ngưỡng)}."""
    yv, sv = r["y_val"], r["cnn_val"]
    st, st_b = r["cnn_tgt"], r["cnn_tgt"][idx]
    t_src = youden(yv, sv)
    P = {"SRC": (st, t_src)}
    q1v, mv, q3v = np.percentile(sv, [25, 50, 75]); q1t, mt, q3t = np.percentile(st_b, [25, 50, 75])
    P["MED"] = (st, mt + (t_src - mv) * (q3t - q1t) / max(q3v - q1v, 1e-9))
    # UPA (Roschewitz và cs., Nat Commun 2023): khớp toàn bộ hàm phân phối điểm đích về validation nguồn rồi dùng ngưỡng
    # nguồn; tương đương ngưỡng = phân vị của điểm đích tại F_nguồn(t_src). Giả định tỉ lệ bệnh không đổi.
    P["UPA"] = (st, np.quantile(st_b, (sv < t_src).mean()))
    P["GMM-2"] = (st, gmm_thr(st_b))
    P["Otsu"] = (st, otsu_thr(st_b))
    P["AdaBN"] = (r["adabn_tgt"], t_src)
    P["TENT"] = (r["tent_tgt"], t_src)
    # CATT với CBM-sd15
    tg = youden(yv, r["cbm5_val"]); gb = r["cbm5_tgt"][idx]
    pl = (gb >= tg).astype(int)
    t_catt = youden(pl, st_b)
    P["CATT"] = (st, t_catt if t_catt is not None else t_src)
    # nhãn mềm: xác suất của g đã dời về ngưỡng nguồn của g (logit(g) − logit(t_g)), trọng số sigmoid
    w1 = 1 / (1 + np.exp(-(logit(gb) - logit(tg)))); P["CATT-soft"] = (st, youden(None, st_b, w1, 1 - w1))
    # chỉ dùng nhãn giả chắc chắn: bỏ dải ±0.5 SD logit(g) quanh ngưỡng
    zg = logit(gb) - logit(tg); keep = np.abs(zg) > 0.5 * zg.std()
    tc = youden(pl[keep], st_b[keep]) if keep.sum() > 10 else None
    P["CATT-confident"] = (st, tc if tc is not None else P["CATT"][1])
    # bộ gán nhãn ghép: trung bình điểm z (chuẩn hóa theo validation nguồn) của logit CBM-sd15 và ST_V2
    def zsrc(v, t):
        return (v - v.mean()) / (v.std() + 1e-9), (t - v.mean()) / (v.std() + 1e-9)
    a_v, a_t = zsrc(logit(r["cbm5_val"]), logit(r["cbm5_tgt"])); b_v, b_t = zsrc(r["stv2_val"], r["stv2_tgt"])
    ev, et = (a_v + b_v) / 2, (a_t + b_t) / 2
    ple = (et[idx] >= youden(yv, ev)).astype(int); te = youden(ple, st_b)
    P["CATT-ensemble"] = (st, te if te is not None else t_src)
    # thích nghi lúc huấn luyện
    for m in ("dann", "coral"):
        P[m.upper() if m == "dann" else "CORAL"] = (r[f"{m}_tgt"], youden(yv, r[f"{m}_val"]))
        tm = youden(pl, r[f"{m}_tgt"][idx])
        P[("DANN" if m == "dann" else "CORAL") + "+CATT"] = (r[f"{m}_tgt"], tm if tm is not None else youden(yv, r[f"{m}_val"]))
    to = youden(r["y_tgt"][idx], st_b)
    P["ORACLE"] = (st, to if to is not None else t_src)   # lô nhỏ không có ca dương: oracle không xác định
    return P


METHODS = ["SRC", "MED", "UPA", "GMM-2", "Otsu", "AdaBN", "TENT", "DANN", "CORAL", "CATT", "CATT-soft", "CATT-confident",
           "CATT-ensemble", "DANN+CATT", "CORAL+CATT", "ORACLE"]


def eval_batch(i, idx_eval_same=True, rng=None, idx=None):
    """bal-acc trung bình 3 seed validation cho thiết lập i, ngưỡng chọn trên lô idx, đánh giá trên lô idx
    (bootstrap) hoặc trên toàn cohort (đường cong cỡ lô)."""
    out = {m: [] for m in METHODS}
    for vs in (0, 1, 2):
        r = R[(i, vs)]; y = r["y_tgt"]
        P = thresholds(r, idx)
        for m in METHODS:
            s, t = P[m]
            out[m].append(bacc(y[idx], s[idx], t) if idx_eval_same else bacc(y, s, t))
    return {m: float(np.mean(v)) for m, v in out.items()}


# ---------------------------------------------------------------- 0. ước lượng điểm (natural) + AUROC
rows = []
for i in range(9):
    n = len(R[(i, 0)]["y_tgt"])
    e = eval_batch(i, idx=np.arange(n))
    au = {f"AUROC_{m}": float(np.mean([roc_auc_score(R[(i, vs)]["y_tgt"], R[(i, vs)][k]) for vs in (0, 1, 2)]))
          for m, k in [("CNN", "cnn_tgt"), ("AdaBN", "adabn_tgt"), ("TENT", "tent_tgt"), ("DANN", "dann_tgt"), ("CORAL", "coral_tgt")]}
    rows.append(dict(setting=i, name=NAMES[i], split="dev" if i < 2 else "held-out", **e, **au))
point = pd.DataFrame(rows)
point.round(4).to_csv(os.path.join(OUT, "v8_point_natural.csv"), index=False)
summ = point.groupby("split")[METHODS + [c for c in point if c.startswith("AUROC")]].agg(["mean", "min"]).round(3)
print(summ.T.to_string())
summ.to_csv(os.path.join(OUT, "v8_point_summary.csv"))

ONLY_SIZE = os.environ.get("ONLY_SIZE") == "1"
# ---------------------------------------------------------------- 1. bootstrap (natural prevalence)
rng = np.random.default_rng(2026)
B = 0 if ONLY_SIZE else B
COMP = [m for m in METHODS if m not in ("CATT", "ORACLE")]
boot = {i: [] for i in range(9)}
for i in range(9):
    y = R[(i, 0)]["y_tgt"]; pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])   # bootstrap phân tầng theo lớp
        boot[i].append(eval_batch(i, idx=idx))
    print("bootstrap", NAMES[i], flush=True)
brow = []
for i in range(9):
    d = pd.DataFrame(boot[i])
    for m in COMP + ["ORACLE"]:
        diff = d["CATT"] - d[m]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
        brow.append(dict(setting=i, name=NAMES[i], vs=m, delta=float(point.loc[i, "CATT"] - point.loc[i, m]),
                         lo=lo, hi=hi, p=min(1.0, p)))
bt = pd.DataFrame(brow)
pooled = []
for m in COMP + ["ORACLE"]:
    for split, S in [("held-out", HELD), ("all", list(range(9)))]:
        diff = np.mean([pd.DataFrame(boot[i])["CATT"] - pd.DataFrame(boot[i])[m] for i in S], 0)
        lo, hi = np.percentile(diff, [2.5, 97.5])
        pooled.append(dict(split=split, vs=m, delta=float(np.mean([point.loc[i, "CATT"] - point.loc[i, m] for i in S])),
                           lo=lo, hi=hi, p=min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))))
pooled = pd.DataFrame(pooled)
# Holm trong từng phương pháp so sánh, qua 9 thiết lập
bt["p_holm"] = np.nan
for m, g in bt.groupby("vs"):
    o = g.sort_values("p"); k = len(o); adj = np.maximum.accumulate([(k - j) * p for j, p in enumerate(o.p)])
    bt.loc[o.index, "p_holm"] = np.minimum(1, adj)
if not ONLY_SIZE:
    bt.round(4).to_csv(os.path.join(OUT, "v8_bootstrap_per_setting.csv"), index=False)
    pooled.round(4).to_csv(os.path.join(OUT, "v8_bootstrap_pooled.csv"), index=False)
print(pooled.round(3).to_string())

# ---------------------------------------------------------------- 2. tỉ lệ bệnh 5%: phân bố qua các lô
ND = 0 if ONLY_SIZE else int(os.environ.get("NDRAW", 300))
draws = {i: [] for i in range(9)}
for i in range(9):
    y = R[(i, 0)]["y_tgt"]; pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    npos = max(2, int(round(0.05 / 0.95 * len(neg))))
    for k in range(ND):
        idx = np.concatenate([neg, rng.choice(pos, npos, replace=False)])
        draws[i].append(eval_batch(i, idx=idx))
p5 = []
for m in COMP + ["ORACLE", "CATT"]:
    per = {i: pd.DataFrame(draws[i])[m].mean() for i in range(9)}
    p5.append(dict(method=m, heldout_mean=np.mean([per[i] for i in HELD]), heldout_worst=min(per[i] for i in HELD),
                   dev_mean=np.mean([per[0], per[1]])))
p5 = pd.DataFrame(p5)
p5d = []
for m in COMP + ["ORACLE"]:
    diff = np.mean([pd.DataFrame(draws[i])["CATT"] - pd.DataFrame(draws[i])[m] for i in HELD], 0)
    p5d.append(dict(vs=m, delta=float(diff.mean()), lo=np.percentile(diff, 2.5), hi=np.percentile(diff, 97.5),
                    frac_catt_better=float((diff > 0).mean())))
if not ONLY_SIZE:
    p5.round(4).to_csv(os.path.join(OUT, "v8_prev5_summary.csv"), index=False)
    pd.DataFrame(p5d).round(4).to_csv(os.path.join(OUT, "v8_prev5_pooled_diff.csv"), index=False)
print(p5.round(3).to_string()); print(pd.DataFrame(p5d).round(3).to_string())

# ---------------------------------------------------------------- 3. cỡ lô đích cần thiết
NS = [20, 30, 50, 75, 100, 150, 200, 300]
NSD = int(os.environ.get("NSIZE", 200))
cur = []
for i in range(9):
    y = R[(i, 0)]["y_tgt"]; N = len(y)
    for n in NS + [N]:
        if n > N:
            continue
        vals = []
        for k in range(NSD if n < N else 1):
            idx = rng.choice(N, n, replace=False) if n < N else np.arange(N)
            vals.append(eval_batch(i, idx_eval_same=False, idx=idx))
        d = pd.DataFrame(vals)
        for m in ["SRC", "MED", "UPA", "GMM-2", "CATT", "ORACLE"]:
            cur.append(dict(setting=i, name=NAMES[i], n=n if n < N else "all", method=m, mean=d[m].mean(),
                            p10=np.percentile(d[m], 10), p90=np.percentile(d[m], 90)))
    print("size curve", NAMES[i], flush=True)
cur = pd.DataFrame(cur)
cur.round(4).to_csv(os.path.join(OUT, "v8_sample_size_curve.csv"), index=False)
cs = cur[cur.setting.isin(HELD)].groupby(["n", "method"])[["mean", "p10"]].mean().unstack().round(3)
print(cs.to_string()); cs.to_csv(os.path.join(OUT, "v8_sample_size_heldout.csv"))
print("xong")
