"""(Bước 7) Kết luận có đổi không nếu nhãn Melo tái tạo sai một phần?

Kịch bản: nhãn "thật" của Claris/Mortara khác nhãn tái tạo ở p% bệnh nhân (lật ngẫu nhiên, 20 lần rút).
Mô hình giữ nguyên như đã huấn luyện (vì trong thực tế chúng đã được huấn luyện trên nhãn tái tạo); chỉ phép ĐÁNH GIÁ
trên miền đích Melo dùng nhãn "thật". Ngưỡng nguồn và nhãn giả của CATT vẫn tính như cũ (giống thực tế).
Đo trên 6 thiết lập có miền đích là Claris hoặc Mortara (các thiết lập có đích HUCA không bị ảnh hưởng vì nhãn HUCA không tái tạo):
  AUROC của CNN-V1V3, CBM5-Sign; balanced accuracy của CNN-V1V3 với SRC / CATT / ORACLE; mức tăng CATT − SRC.
Chạy: python label_sens.py <concepts_v5> <harmonized> <output>
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

V5, HD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
Y = {d: np.load(os.path.join(HD, f"{d}.npz"))["y"].astype(int) for d in ["huca", "claris", "mortara"]}
R = [json.loads(l) for f in sorted(glob.glob(os.path.join(V5, "v5_raw_*.jsonl"))) for l in open(f)]
R = [r for r in R if r["test"] != "huca"]


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return thr[np.argmax(tpr - fpr)]


def bacc(y, p):
    return float((p[y == 1].mean() + (~p)[y == 0].mean()) / 2)


rows = []
rng = np.random.default_rng(0)
for p in [0.0, 0.02, 0.05, 0.10]:
    for k in range(1 if p == 0 else 20):
        flips = {d: rng.random(len(Y[d])) < p for d in ["claris", "mortara"]}
        for r in R:
            yt = Y[r["test"]].copy(); yt[flips[r["test"]]] = 1 - yt[flips[r["test"]]]
            yv = np.array(r["y_val"])
            sv, st = np.array(r["models"]["CNN-V1V3"]["val"]), np.array(r["models"]["CNN-V1V3"]["tgt"])
            cv, ct = np.array(r["models"]["CBM5-Sign"]["val"]), np.array(r["models"]["CBM5-Sign"]["tgt"])
            t_src = youden(yv, sv)
            pl = (ct >= youden(yv, cv)).astype(int)
            t_catt = youden(pl, st) if 0 < pl.mean() < 1 else t_src
            rows.append(dict(p=p, draw=k, setting=r["setting"], name=f"{r['train']}->{r['test']}", vseed=r["vseed"],
                             auroc_cnn=roc_auc_score(yt, st), auroc_cbm=roc_auc_score(yt, ct),
                             SRC=bacc(yt, st >= t_src), CATT=bacc(yt, st >= t_catt), ORACLE=bacc(yt, st >= youden(yt, st))))
d = pd.DataFrame(rows)
d["gain"] = d.CATT - d.SRC
per = d.groupby(["p", "name"])[["auroc_cnn", "SRC", "CATT", "ORACLE", "gain"]].mean().round(3)
summ = d.groupby(["p", "draw", "setting"])[["auroc_cnn", "auroc_cbm", "SRC", "CATT", "ORACLE", "gain"]].mean().groupby(["p", "draw"]).mean()
out = summ.groupby("p").agg(["mean", "min"]).round(3)
print(out.to_string()); print(per.to_string())
out.to_csv(os.path.join(OUT, "label_sens_summary.csv")); per.to_csv(os.path.join(OUT, "label_sens_per_setting.csv"))
