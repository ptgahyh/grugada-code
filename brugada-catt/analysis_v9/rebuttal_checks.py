"""(Bước 9) Ba kiểm tra cho phản biện, dùng điểm từng bệnh nhân trong v5_raw (không huấn luyện lại).

A. Chia đôi: ngưỡng của mỗi phương pháp chọn trên một nửa ECG đích (ngẫu nhiên, không dùng nhãn), balanced accuracy
   đo trên nửa còn lại. 100 lần chia × 3 seed validation; so với cách chọn và đo trên toàn cohort.
B. Khoảng tin cậy của giá trị TỆ NHẤT (min qua 7 thiết lập held-out):
   - tỉ lệ tự nhiên: bootstrap bệnh nhân phân tầng theo lớp trong mỗi cohort đích (1000 lần), mọi ngưỡng phụ thuộc lô
     được ước lượng lại trên mỗi mẫu bootstrap (như v8);
   - tỉ lệ 5%: 200 lô 5% mỗi thiết lập (như v8), bootstrap các lô (1000 lần).
C. ST_V2 so với CBM-sd15 làm bộ gán nhãn, theo tam phân vị độ rộng QRS (QRS_w của lớp khái niệm V1–V3 ở mốc khởi tạo,
   tam phân vị tính riêng trong mỗi cohort đích, không dùng nhãn): balanced accuracy và AUROC của bộ gán nhãn.

Phương pháp đặt ngưỡng (giống v8_analyze.py): SRC, MED, UPA, GMM-2, Otsu, CATT (CBM-sd15), ORACLE.
Chạy: python rebuttal_checks.py <concepts_v5> <harmonized> <output>
"""
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "concepts_v4"))
import cbm_lib4 as L  # noqa: E402
import concepts3  # noqa: E402

warnings.filterwarnings("ignore")
V5, HD, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
X, Y = L.load(HD)
DOM = L.DOMAINS
SETTINGS = [([a], b) for a in DOM for b in DOM if a != b] + [([a for a in DOM if a != b], b) for b in DOM]
NAME = {0: "HUCA→Claris", 1: "HUCA→Mortara", 2: "Claris→HUCA", 3: "Claris→Mortara", 4: "Mortara→HUCA",
        5: "Mortara→Claris", 6: "Claris+Mortara→HUCA", 7: "HUCA+Mortara→Claris", 8: "HUCA+Claris→Mortara"}
HELD = [2, 3, 4, 5, 6, 7, 8]
METHODS = ["SRC", "MED", "UPA", "GMM-2", "Otsu", "CATT", "ORACLE"]


# ------------------------------------------------------------------ hàm đặt ngưỡng (chép từ v8_analyze.py)
def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def youden(y, s):
    if y.min() == y.max():
        return None
    fpr, tpr, thr = roc_curve(y, s)
    thr = np.minimum(thr, s.max() + 1e-9)
    return thr[np.argmax(tpr - fpr)]


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
    """Ngưỡng của mọi phương pháp, chọn trên lô đích idx (chỉ số vào cohort đích)."""
    yv, sv, st_b = r["y_val"], r["cnn_val"], r["cnn_tgt"][idx]
    t_src = youden(yv, sv)
    q1v, mv, q3v = np.percentile(sv, [25, 50, 75]); q1t, mt, q3t = np.percentile(st_b, [25, 50, 75])
    T = {"SRC": t_src, "MED": mt + (t_src - mv) * (q3t - q1t) / max(q3v - q1v, 1e-9),
         "UPA": np.quantile(st_b, (sv < t_src).mean()), "GMM-2": gmm_thr(st_b), "Otsu": otsu_thr(st_b)}
    pl = (r["cbm5_tgt"][idx] >= youden(yv, r["cbm5_val"])).astype(int)
    tc = youden(pl, st_b)
    T["CATT"] = tc if tc is not None else t_src
    to = youden(r["y_tgt"][idx], st_b)
    T["ORACLE"] = to if to is not None else t_src
    return T


def eval_setting(i, idx_sel, idx_eval):
    """balanced accuracy trung bình 3 seed validation: ngưỡng chọn trên idx_sel, đo trên idx_eval."""
    out = {m: [] for m in METHODS}
    for vs in (0, 1, 2):
        r = R[(i, vs)]
        T = thresholds(r, idx_sel)
        for m in METHODS:
            out[m].append(bacc(r["y_tgt"][idx_eval], r["cnn_tgt"][idx_eval], T[m]))
    return {m: float(np.mean(v)) for m, v in out.items()}


# ------------------------------------------------------------------ dữ liệu: v5_raw + ST_V2 + QRS_w
R = {}
rows5 = [json.loads(line) for f in sorted(glob.glob(os.path.join(V5, "v5_raw_*.jsonl"))) for line in open(f)]
for r5 in rows5:
    i, vs = r5["setting"], r5["vseed"]
    src, tgt = SETTINGS[i]
    xtr, ytr, dtr, di = L.pack(X, Y, src)
    tr, va = L.strat_val(ytr, di, vs)
    assert ytr[va].tolist() == r5["y_val"], (i, vs)
    stv2 = lambda x: x[:, 4, 34:38].mean(1) - x[:, 4, 14:20].mean(1)  # noqa: E731
    R[(i, vs)] = dict(y_val=np.array(r5["y_val"]), y_tgt=Y[tgt],
                      cnn_val=np.array(r5["models"]["CNN-V1V3"]["val"]), cnn_tgt=np.array(r5["models"]["CNN-V1V3"]["tgt"]),
                      cbm5_val=np.array(r5["models"]["CBM5-Sign"]["val"]), cbm5_tgt=np.array(r5["models"]["CBM5-Sign"]["tgt"]),
                      stv2_val=stv2(xtr[va]), stv2_tgt=stv2(X[tgt]))
assert len(R) == 27
with torch.no_grad():
    QRSW = {d: concepts3.BrugadaConcepts(learn=False)(torch.tensor(X[d]))[:, -1].numpy() * 100 for d in DOM}  # ms

# kiểm tra: điểm toàn cohort phải tái lập Bảng 11 (CATT held-out 0.799, SRC 0.701)
full = {i: eval_setting(i, np.arange(len(Y[SETTINGS[i][1]])), np.arange(len(Y[SETTINGS[i][1]]))) for i in range(9)}
print("Kiểm tra toàn cohort (held-out):", {m: round(np.mean([full[i][m] for i in HELD]), 3) for m in METHODS})

# ------------------------------------------------------------------ A. chia đôi
rng = np.random.default_rng(1)
rowsA = []
for i in range(9):
    n = len(Y[SETTINGS[i][1]])
    for k in range(100):
        p = rng.permutation(n); a, b = p[: n // 2], p[n // 2:]
        rowsA.append(dict(setting=i, split=k, how="chọn A, đo B", **eval_setting(i, a, b)))
        rowsA.append(dict(setting=i, split=k, how="chọn và đo trên A", **eval_setting(i, a, a)))
A = pd.DataFrame(rowsA)
A.to_csv(os.path.join(OUT, "split_half_raw.csv"), index=False)
perA = A.groupby(["how", "setting"])[METHODS].mean()
sumA = []
for how in ["chọn A, đo B", "chọn và đo trên A"]:
    p = perA.loc[how]
    sumA.append(dict(how=how, **{f"{m}_mean": p.loc[HELD, m].mean() for m in METHODS},
                     **{f"{m}_worst": p.loc[HELD, m].min() for m in METHODS}))
sumA.append(dict(how="toàn cohort (Bảng 11/12)", **{f"{m}_mean": np.mean([full[i][m] for i in HELD]) for m in METHODS},
                 **{f"{m}_worst": min(full[i][m] for i in HELD) for m in METHODS}))
sumA = pd.DataFrame(sumA).set_index("how").round(3)
sumA.to_csv(os.path.join(OUT, "split_half_summary.csv"))
perA.round(3).to_csv(os.path.join(OUT, "split_half_per_setting.csv"))
print("\nA. Chia đôi (held-out, 7 thiết lập):\n", sumA[[f"{m}_mean" for m in METHODS] + [f"{m}_worst" for m in METHODS]].T.to_string())

# ------------------------------------------------------------------ B. CI của giá trị tệ nhất
B = 1000
rng = np.random.default_rng(2)
boot = {i: [] for i in HELD}
for i in HELD:
    y = Y[SETTINGS[i][1]]; pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for b in range(B):
        idx = np.concatenate([rng.choice(neg, len(neg)), rng.choice(pos, len(pos))])
        boot[i].append(eval_setting(i, idx, idx))
nat = {m: np.array([[boot[i][b][m] for i in HELD] for b in range(B)]) for m in METHODS}  # (B, 7)

ND = 200
rng = np.random.default_rng(3)
draws = {}
for i in HELD:
    y = Y[SETTINGS[i][1]]; pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    npos = max(2, int(round(0.05 / 0.95 * len(neg))))
    draws[i] = pd.DataFrame([eval_setting(i, idx, idx) for idx in
                             (np.concatenate([neg, rng.choice(pos, npos, replace=False)]) for _ in range(ND))])
p5 = {m: np.array([[draws[i][m].to_numpy()[rng.integers(0, ND, ND)].mean() for i in HELD] for _ in range(B)])
      for m in METHODS}

rowsB = []
for label, D, point in [("tự nhiên", nat, {m: [full[i][m] for i in HELD] for m in METHODS}),
                        ("5%", p5, {m: [draws[i][m].mean() for i in HELD] for m in METHODS})]:
    for m in METHODS:
        w = D[m].min(1); dw = D["CATT"].min(1) - w
        rowsB.append(dict(prevalence=label, method=m, worst=min(point[m]),
                          worst_lo=np.percentile(w, 2.5), worst_hi=np.percentile(w, 97.5),
                          mean=np.mean(point[m]),
                          catt_minus_worst=min(point["CATT"]) - min(point[m]),
                          diff_lo=np.percentile(dw, 2.5), diff_hi=np.percentile(dw, 97.5),
                          p_catt_worst_higher=float((dw > 0).mean())))
Bdf = pd.DataFrame(rowsB).round(3)
Bdf.to_csv(os.path.join(OUT, "worst_case_ci.csv"), index=False)
print("\nB. Giá trị tệ nhất qua 7 thiết lập held-out, CI 95% bootstrap:\n", Bdf.to_string(index=False))

# ------------------------------------------------------------------ C. ST_V2 so với CBM-sd15 theo độ rộng QRS
rowsC = []
for i in range(9):
    tgt = SETTINGS[i][1]
    q = QRSW[tgt]; cuts = np.quantile(q, [1 / 3, 2 / 3])
    terc = np.digitize(q, cuts)  # 0 hẹp, 1 giữa, 2 rộng
    for vs in (0, 1, 2):
        r = R[(i, vs)]; y = r["y_tgt"]
        for lab, sv, stt in [("CBM-sd15", logit(r["cbm5_val"]), logit(r["cbm5_tgt"])), ("ST_V2", r["stv2_val"], r["stv2_tgt"])]:
            pl = stt >= youden(r["y_val"], sv)
            for t in (0, 1, 2):
                m = terc == t
                if y[m].min() == y[m].max():
                    continue
                rowsC.append(dict(setting=i, vseed=vs, labeller=lab, tertile=["hẹp", "giữa", "rộng"][t],
                                  qrs_ms_median=float(np.median(q[m])), n=int(m.sum()), n_pos=int(y[m].sum()),
                                  bacc=0.5 * (pl[m & (y == 1)].mean() + (~pl[m & (y == 0)]).mean()),
                                  auroc=roc_auc_score(y[m], stt[m])))
C = pd.DataFrame(rowsC)
C.to_csv(os.path.join(OUT, "stv2_vs_cbm_by_qrs_raw.csv"), index=False)
perC = C.groupby(["tertile", "labeller", "setting"])[["bacc", "auroc"]].mean().reset_index()
sumC = perC.groupby(["tertile", "labeller"])[["bacc", "auroc"]].mean().unstack("labeller")
# chênh CBM − ST_V2 trong từng tam phân vị, khoảng qua 9 thiết lập và số thiết lập CBM tốt hơn
d = perC.pivot_table(index=["tertile", "setting"], columns="labeller", values=["bacc", "auroc"])
diff = pd.DataFrame({"bacc_diff": d["bacc"]["CBM-sd15"] - d["bacc"]["ST_V2"],
                     "auroc_diff": d["auroc"]["CBM-sd15"] - d["auroc"]["ST_V2"]})
sumDiff = diff.groupby("tertile").agg(bacc_diff_mean=("bacc_diff", "mean"), bacc_cbm_better=("bacc_diff", lambda v: int((v > 0).sum())),
                                      auroc_diff_mean=("auroc_diff", "mean"), auroc_cbm_better=("auroc_diff", lambda v: int((v > 0).sum())))
sumC.round(3).to_csv(os.path.join(OUT, "stv2_vs_cbm_by_qrs_summary.csv"))
sumDiff.round(3).to_csv(os.path.join(OUT, "stv2_vs_cbm_by_qrs_diff.csv"))
print("\nC. Bộ gán nhãn theo tam phân vị QRS_w (trung bình 9 thiết lập):\n", sumC.round(3).to_string())
print("\nChênh CBM-sd15 − ST_V2 (số thiết lập CBM tốt hơn / 9):\n", sumDiff.round(3).to_string())
print("QRS_w trung vị theo tam phân vị:", C.groupby("tertile").qrs_ms_median.median().round(0).to_dict())
