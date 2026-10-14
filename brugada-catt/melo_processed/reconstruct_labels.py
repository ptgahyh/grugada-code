"""Khôi phục nhãn Brugada theo từng bệnh nhân cho bộ Melo et al. (Dryad 10.5061/dryad.s1rn8pkd9).

Bản phát hành không có file nhãn. Nhãn được suy ra từ đường ROC (auc_roc_All.xlsx)
và điểm dự đoán (holdout_reg.xlsx, lấy trung bình 7 mô hình):
sắp xếp bệnh nhân theo điểm giảm dần, mỗi đoạn dọc của ROC là ca dương, mỗi đoạn ngang là ca âm.

Cách chạy:  python reconstruct_labels.py "<đường dẫn tới Data/Results>"
"""
import sys
import numpy as np
import pandas as pd

R = sys.argv[1]


def load(p):
    return pd.read_excel(f"{R}/{p}", header=0).values.astype(float)


def recon(score, roc, n_pos, n_neg):
    dp = np.rint(np.diff(roc[:, 1]) * n_pos).astype(int)
    dn = np.rint(np.diff(roc[:, 0]) * n_neg).astype(int)
    order = np.argsort(-score, kind="stable")
    y = np.full(len(score), -1)
    k = 0
    for a, b in zip(dp, dn):
        idx = order[k:k + a + b]
        if a > 0 and b == 0:
            y[idx] = 1
        elif b > 0 and a == 0:
            y[idx] = 0          # đoạn chéo (điểm trùng nhau) giữ -1 = không xác định
        k += a + b
    return y


sets = {
    "claris": ("Training_Claris/holdout_reg.xlsx", "Training_Claris/auc_roc_All.xlsx", 596, 558, (509, 460)),
    "mortara": ("Validation_Mortara/CNN/holdout_reg.xlsx", "Validation_Mortara/CNN/auc_roc_All.xlsx", 137, 233, (102, 207)),
}
for name, (hf, rf, P, N, (tp_rep, tn_rep)) in sets.items():
    s = np.nanmean(load(hf), axis=1)
    y = recon(s, load(rf), P, N)
    tp = int(((s >= 0.5) & (y == 1)).sum())
    tn = int(((s < 0.5) & (y == 0)).sum())
    print(f"{name}: gán được {int((y >= 0).sum())}/{len(y)}, dương={int((y == 1).sum())}, "
          f"TP@0.5={tp} (báo cáo {tp_rep}), TN@0.5={tn} (báo cáo {tn_rep})")
    pd.DataFrame({"row": range(len(y)), "label_brs": y}).to_csv(f"labels_{name}.csv", index=False)
