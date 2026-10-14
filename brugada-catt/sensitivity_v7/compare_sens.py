"""(Bước 7) So sánh kết quả HUCA tính kiểu máy đo (sens_huca_*.jsonl) với HUCA gốc (concepts_v5), cùng thiết lập, cùng seed.
Chạy: python compare_sens.py <thư mục này> <concepts_v5>
"""
import glob
import json
import os
import sys

import pandas as pd

OD, V5 = sys.argv[1], sys.argv[2]
new = [json.loads(l) for f in glob.glob(os.path.join(OD, "sens_huca_*.jsonl")) for l in open(f)]
old5 = {(r["setting"], r["vseed"]): r for f in glob.glob(os.path.join(V5, "v5_raw_*.jsonl")) for r in map(json.loads, open(f))}
rc = pd.read_csv(os.path.join(V5, "recal_raw.csv"))
rc = rc[(rc.prev == "tự nhiên") & (rc.model == "CNN-V1V3")].set_index(["setting", "vseed", "method"]).bacc
rows = []
for r in new:
    k = (r["setting"], r["vseed"])
    rows.append(dict(name=f"{r['train']}->{r['test']}", setting=r["setting"], vseed=r["vseed"],
                     auroc_v13_old=old5[k]["models"]["CNN-V1V3"]["auroc"], auroc_v13_new=r["auroc"]["CNN-V1V3"],
                     auroc_cnn9_new=r["auroc"]["CNN9"],
                     auroc_cbm5_old=old5[k]["models"]["CBM5-Sign"]["auroc"], auroc_cbm5_new=r["auroc"]["CBM5-Sign"],
                     SRC_old=rc.loc[(k[0], k[1], "SRC")], SRC_new=r["bacc"]["SRC"],
                     CATT_old=rc.loc[(k[0], k[1], "CATT")], CATT_new=r["bacc"]["CATT"],
                     ORACLE_old=rc.loc[(k[0], k[1], "ORACLE")], ORACLE_new=r["bacc"]["ORACLE"]))
d = pd.DataFrame(rows)
per = d.groupby(["setting", "name"]).mean(numeric_only=True).drop(columns="vseed").round(3)
pd.set_option("display.width", 250)
print(per.to_string())
m = per.mean().round(3)
print("\nTB 7 thiết lập:"); print(m.to_string())
per.to_csv(os.path.join(OD, "sens_huca_per_setting.csv")); m.to_csv(os.path.join(OD, "sens_huca_summary.csv"))
