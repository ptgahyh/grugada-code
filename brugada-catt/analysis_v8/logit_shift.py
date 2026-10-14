"""(Bước 8) Độ dịch logit của từng lớp khi chuyển miền, theo đơn vị độ tách lớp của chính mô hình (Mục 3.4).

Với mỗi (thiết lập, seed validation, mô hình), trên thang logit:
  tách lớp nguồn   sep   = mean(logit val dương) − mean(logit val âm)
  dịch âm tính     d_neg = mean(logit đích âm)   − mean(logit val âm)
  dịch dương tính  d_pos = mean(logit đích dương) − mean(logit val dương)
  báo cáo d_neg / sep và d_pos / sep (1.0 = âm tính đích nằm đúng chỗ dương tính nguồn).
Với đích HUCA: tính lại d_neg sau khi bỏ các ca âm tính có basal_pattern = 1 (23 ca), báo cáo chênh lệch
theo đơn vị logit và theo % của d_neg. Nhãn đích chỉ dùng để tách lớp khi PHÂN TÍCH.

Mô hình: CNN-V1V3, CBM (CBM3-Sign, QRS_w), CBM-sd15 (CBM5-Sign), lấy từ v5_raw_*.jsonl của compare_v5.py.

Chạy: python logit_shift.py <thư mục chứa v5_raw_*.jsonl> <harmonized> <thư mục HUCA có metadata.csv> <output>
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

RAW, HD, HU, OUT = sys.argv[1:5]
MODELS = {"CNN-V1V3": "CNN-V1V3", "CBM3-Sign": "CBM", "CBM5-Sign": "CBM-sd15"}
Y = {d: np.load(os.path.join(HD, f"{d}.npz"))["y"].astype(int) for d in ["huca", "claris", "mortara"]}
ids = np.load(os.path.join(HD, "huca.npz"))["ids"]
meta = pd.read_csv(os.path.join(HU, "metadata.csv")).set_index("patient_id")
BASAL = meta.loc[ids, "basal_pattern"].to_numpy().astype(bool)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


rows = []
for f in sorted(glob.glob(os.path.join(RAW, "v5_raw_*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        yv, yt = np.array(r["y_val"]), Y[r["test"]]
        for key, name in MODELS.items():
            lv, lt = logit(r["models"][key]["val"]), logit(r["models"][key]["tgt"])
            sep = lv[yv == 1].mean() - lv[yv == 0].mean()
            d_neg = lt[yt == 0].mean() - lv[yv == 0].mean()
            d_pos = lt[yt == 1].mean() - lv[yv == 1].mean()
            row = dict(setting=r["setting"], train=r["train"], test=r["test"], vseed=r["vseed"], model=name,
                       sep=sep, d_neg=d_neg, d_pos=d_pos, frac_neg=d_neg / sep, frac_pos=d_pos / sep)
            if r["test"] == "huca":
                keep = (yt == 0) & ~BASAL
                d_neg_excl = lt[keep].mean() - lv[yv == 0].mean()
                row.update(d_neg_excl_basal=d_neg_excl, change_logit=abs(d_neg_excl - d_neg),
                           change_pct=100 * abs(d_neg_excl - d_neg) / abs(d_neg))
            rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "logit_shift_raw.csv"), index=False)
per = df.groupby(["setting", "train", "test", "model"]).mean(numeric_only=True).drop(columns="vseed").round(3)
per.to_csv(os.path.join(OUT, "logit_shift_per_setting.csv"))
pd.set_option("display.width", 200)
print(per.to_string())

print("\nTóm tắt (trung bình 3 seed validation, khoảng qua các thiết lập):")
for label, mask in [("đích HUCA", per.index.get_level_values("test") == "huca"),
                    ("HUCA -> Melo (nguồn đơn)", per.index.get_level_values("train") == "huca")]:
    sub = per[mask]
    for name in MODELS.values():
        s = sub.xs(name, level="model")
        msg = f"  {label:26s} {name:9s} âm tính dịch {s.frac_neg.min():+.2f} đến {s.frac_neg.max():+.2f} lần tách lớp"
        if "change_logit" in s and s.change_logit.notna().any():
            msg += f"; bỏ âm tính có dạng nền: đổi tối đa {s.change_logit.max():.2f} logit ({s.change_pct.min():.0f}–{s.change_pct.max():.0f}%)"
        print(msg)
