"""(Bước 8) Nhóm con trong HUCA khi HUCA là miền đích (Claris→HUCA, Mortara→HUCA, Claris+Mortara→HUCA).

Theo PhysioNet, cột `brugada` là CHẨN ĐOÁN hội chứng Brugada (0/1; giá trị 2 có ở 7 bản ghi nhưng không được định
nghĩa), còn `basal_pattern` cho biết ECG nền có dạng Brugada hay không. Chỉ 21/73 ca dương tính có dạng Brugada trên ECG
nền, và 23 ca âm tính lại có dạng này. Phân tích:
  1. AUROC theo nhóm: dương tính CÓ dạng nền vs dương tính KHÔNG có dạng nền (mỗi nhóm so với toàn bộ âm tính).
  2. Âm tính có dạng nền (23 ca): tỉ lệ bị báo dương ở ngưỡng SRC / CATT / ORACLE so với âm tính không có dạng nền.
  3. Loại 7 ca nhãn 2 khỏi đánh giá: AUROC và balanced accuracy (SRC, CATT, ORACLE) có đổi không.
Chạy: python huca_subgroups.py <thư mục HUCA có metadata.csv> <harmonized> <scores> <out>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

HU, HD, SD, OUT = sys.argv[1:5]
m = pd.read_csv(os.path.join(HU, "metadata.csv")).set_index("patient_id")
ids = np.load(os.path.join(HD, "huca.npz"))["ids"]
k = m.loc[ids]
lab, basal = k.brugada.values, k.basal_pattern.values
y = (lab > 0).astype(int)
NAMES = {2: "Claris→HUCA", 4: "Mortara→HUCA", 6: "Claris+Mortara→HUCA"}


def youden(yy, s):
    if yy.min() == yy.max():
        return None
    fpr, tpr, thr = roc_curve(yy, s)
    return np.minimum(thr, s.max() + 1e-9)[np.argmax(tpr - fpr)]


def bacc(yy, s, t):
    p = s >= t
    return 0.5 * (p[yy == 1].mean() + (~p)[yy == 0].mean())


rows = []
for i, nm in NAMES.items():
    for vs in (0, 1, 2):
        r = dict(np.load(os.path.join(SD, f"scores_{i}_{vs}.npz")))
        assert (r["y_tgt"] == y).all()
        for model, key in [("CNN-V1V3", "cnn"), ("CBM-sd15", "cbm5"), ("CBM", "cbm3")]:
            s = r[f"{key}_tgt"]
            neg = y == 0
            row = dict(setting=nm, vseed=vs, model=model,
                       auroc_all=roc_auc_score(y, s),
                       auroc_pos_with_basal=roc_auc_score(np.r_[np.ones((y & basal).sum()), np.zeros(neg.sum())],
                                                          np.r_[s[(y == 1) & (basal == 1)], s[neg]]),
                       auroc_pos_without_basal=roc_auc_score(np.r_[np.ones(((y == 1) & (basal == 0)).sum()), np.zeros(neg.sum())],
                                                             np.r_[s[(y == 1) & (basal == 0)], s[neg]]),
                       auroc_excl_label2=roc_auc_score(y[lab != 2], s[lab != 2]))
            if model == "CNN-V1V3":
                t_src = youden(r["y_val"], r["cnn_val"])
                pl = (r["cbm5_tgt"] >= youden(r["y_val"], r["cbm5_val"])).astype(int)
                t_catt = youden(pl, s); t_or = youden(y, s)
                for tn, t in [("SRC", t_src), ("CATT", t_catt), ("ORACLE", t_or)]:
                    p = s >= t
                    row[f"fpr_neg_basal_{tn}"] = p[(y == 0) & (basal == 1)].mean()
                    row[f"fpr_neg_nobasal_{tn}"] = p[(y == 0) & (basal == 0)].mean()
                    row[f"sens_pos_basal_{tn}"] = p[(y == 1) & (basal == 1)].mean()
                    row[f"sens_pos_nobasal_{tn}"] = p[(y == 1) & (basal == 0)].mean()
                    row[f"bacc_{tn}"] = bacc(y, s, t)
                    keep = lab != 2
                    t2 = youden(y[keep], s[keep]) if tn == "ORACLE" else (youden(pl[keep], s[keep]) if tn == "CATT" else t)
                    row[f"bacc_excl_label2_{tn}"] = bacc(y[keep], s[keep], t2)
            rows.append(row)
D = pd.DataFrame(rows)
S = D.groupby(["setting", "model"]).mean(numeric_only=True).drop(columns="vseed").round(3)
S.to_csv(os.path.join(OUT, "huca_subgroups.csv"))
pd.set_option("display.width", 250)
print(S.T.to_string())
counts = dict(pos_with_basal=int(((y == 1) & (basal == 1)).sum()), pos_without_basal=int(((y == 1) & (basal == 0)).sum()),
              neg_with_basal=int(((y == 0) & (basal == 1)).sum()), neg_without_basal=int(((y == 0) & (basal == 0)).sum()),
              label2=int((lab == 2).sum()), label2_with_basal=int(((lab == 2) & (basal == 1)).sum()))
json.dump(counts, open(os.path.join(OUT, "huca_subgroup_counts.json"), "w"), indent=1)
print(counts)
