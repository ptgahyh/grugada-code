"""(Bước 8) Kiểm tra nhãn HUCA và các nhận xét về code trích đặc trưng V1 của phiên khác.
- thành phần nhãn: brugada (0/1/2), basal_pattern, sudden_death; trùng lặp/flatline nằm ở đâu
- khái niệm của mình phân biệt basal_pattern (dạng ECG) hay brugada (chẩn đoán) tốt hơn
- RR_CV (nhịp không đều) với bộ dò đỉnh của harmonize.py
- đột tử: so thô và so có điều chỉnh theo nhãn Brugada
Chạy: python huca_label_check.py <thư mục HUCA> <harmonized> <out>
"""
import json, os, sys
import numpy as np, pandas as pd, torch
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import mannwhitneyu, fisher_exact
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "concepts_v4"))
import concepts3
HU, HD, OUT = sys.argv[1:4]
m = pd.read_csv(os.path.join(HU, "metadata.csv"))
H = np.load(os.path.join(HD, "huca.npz")); ids = H["ids"]
res = {}
res["all_363"] = {"brugada": m.brugada.value_counts().to_dict(), "basal_x_brugada": pd.crosstab(m.basal_pattern, m.brugada).to_dict(),
                  "sd_x_brugada": pd.crosstab(m.brugada, m.sudden_death).to_dict()}
removed = m[~m.patient_id.isin(ids)]
res["removed_records"] = removed.to_dict("records")
# RR_CV
b, a = butter(2, [0.5 / 50, 40 / 50], btype="band")
rr = {}
for pid in m.patient_id:
    fp = os.path.join(HU, "files", str(pid), f"{pid}.dat")
    if not os.path.exists(fp):
        continue
    raw = np.fromfile(os.path.join(HU, "files", str(pid), f"{pid}.dat"), dtype="<i2").reshape(-1, 12).astype(float).T / 1000
    sig = filtfilt(b, a, raw[[0, 1, 2, 6, 7, 8, 9, 10, 11]], axis=1)
    env = np.abs(np.diff(sig, axis=1)).sum(0)
    pk, _ = find_peaks(env, distance=33, height=0.35 * np.percentile(env, 99))
    d = np.diff(pk) * 10.0
    rr[pid] = dict(n_beats=len(pk), rr_mean=float(d.mean()) if len(d) else np.nan, rr_cv=float(d.std() / d.mean()) if len(d) > 1 else np.nan)
R = pd.DataFrame(rr).T; R.index.name = "patient_id"
res["n_files_available"] = len(R); res["rr_cv_gt_0.15_available"] = int((R.rr_cv > 0.15).sum()); res["rr_cv_gt_0.15_in356"] = int((R.loc[ids].rr_cv > 0.15).sum())
res["hr_median_bpm"] = float(60000 / R.rr_mean.median())
# khái niệm cố định (v3, tham số khởi tạo) trên 356 bản ghi
C = concepts3.BrugadaConcepts(learn=False)
with torch.no_grad():
    V = C(torch.tensor(H["X"].astype(np.float32))).numpy()
k = m.set_index("patient_id").loc[ids]
yb = (k.brugada.values > 0).astype(int); yp = k.basal_pattern.values; y1 = k.brugada.values
au = []
for j, nm in enumerate(concepts3.CONCEPT_NAMES):
    au.append(dict(concept=nm, auc_brugada=roc_auc_score(yb, V[:, j]), auc_basal=roc_auc_score(yp, V[:, j]),
                   auc_brugada_without_basal=roc_auc_score(yb[yp == 0], V[yp == 0, j]),
                   auc_label2_vs_0=roc_auc_score((y1[y1 != 1] == 2).astype(int), V[y1 != 1, j])))
A = pd.DataFrame(au); A["sep_brugada"] = (A.auc_brugada - 0.5).abs(); A["sep_basal"] = (A.auc_basal - 0.5).abs()
A.round(3).to_csv(os.path.join(OUT, "huca_concepts_by_label.csv"), index=False)
res["mean_sep_brugada"] = float(A.sep_brugada.mean()); res["mean_sep_basal"] = float(A.sep_basal.mean())
res["top_concepts"] = A.sort_values("sep_basal", ascending=False).head(6)[["concept", "auc_brugada", "auc_basal", "auc_brugada_without_basal"]].round(3).to_dict("records")
# đột tử: thô và trong từng tầng chẩn đoán, dùng J_elev_V1/V2, QRS_w
sd = k.sudden_death.values
for nm in ["J_elev_V1", "J_elev_V2", "ST80_V2", "QRS_w"]:
    j = concepts3.CONCEPT_NAMES.index(nm)
    raw_p = mannwhitneyu(V[sd == 1, j], V[sd == 0, j]).pvalue
    strat = {}
    for g, msk in [("brugada", yb == 1), ("non_brugada", yb == 0)]:
        if (sd[msk] == 1).sum() >= 3:
            strat[g] = dict(n_sd=int((sd[msk] == 1).sum()), p=float(mannwhitneyu(V[msk & (sd == 1), j], V[msk & (sd == 0), j]).pvalue))
    res[f"sd_{nm}"] = dict(p_crude=float(raw_p), stratified=strat)
t = pd.crosstab(yb, sd); res["sd_rate_brugada_vs_not"] = dict(table=t.values.tolist(), fisher_p=float(fisher_exact(t.values)[1]))
json.dump(res, open(os.path.join(OUT, "huca_label_check.json"), "w"), indent=1, default=str)
print(json.dumps({k_: v for k_, v in res.items() if k_ not in ("removed_records",)}, indent=1, default=str))
print(pd.DataFrame(res["removed_records"]))
