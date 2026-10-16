"""(Bước 9) CI của giá trị tệ nhất ở tỉ lệ 5%, có tính sai số chọn BỆNH NHÂN (bổ sung cho phần B của rebuttal_checks.py,
vốn chỉ lấy mẫu lại các lô 5% trên một tập bệnh nhân cố định).

Mỗi lần bootstrap: lấy mẫu lại bệnh nhân phân tầng theo lớp trong cohort đích, rồi rút 10 lô 5% (giữ mọi âm tính của mẫu,
rút dương tính) như v8; ngưỡng ước lượng lại trên từng lô. 200 lần bootstrap × 7 thiết lập held-out.

Chạy (mỗi thiết lập một tiến trình độc lập; multiprocessing.Pool bị treo khi fork sau khi đã nạp torch/OpenMP):
  python worst5_patient_boot.py <concepts_v5> <harmonized> <output> run <i>   -> worst5_boot_<i>.json
  python worst5_patient_boot.py <concepts_v5> <harmonized> <output> summary   -> worst_case_ci_5pct_patient_boot.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "rebuttal_checks.py")).read().split("# kiểm tra: điểm toàn cohort")[0]
exec(SRC)  # nạp dữ liệu R, Y, SETTINGS, HELD, METHODS và các hàm đặt ngưỡng (giống hệt rebuttal_checks.py)

B, NB = 200, 10


def run(i):
    rng = np.random.default_rng(100 + i)
    y = Y[SETTINGS[i][1]]; pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = []
    for b in range(B):
        bneg, bpos = rng.choice(neg, len(neg)), rng.choice(pos, len(pos))
        npos = max(2, int(round(0.05 / 0.95 * len(bneg))))
        res = [eval_setting(i, idx, idx) for idx in
               (np.concatenate([bneg, rng.choice(bpos, npos, replace=False)]) for _ in range(NB))]
        out.append({m: float(np.mean([r[m] for r in res])) for m in METHODS})
    return out


if __name__ == "__main__":
    if sys.argv[4] == "run":
        i = int(sys.argv[5])
        json.dump(run(i), open(os.path.join(OUT, f"worst5_boot_{i}.json"), "w"))
        sys.exit()
    res = {i: json.load(open(os.path.join(OUT, f"worst5_boot_{i}.json"))) for i in HELD}
    D = {m: np.array([[res[i][b][m] for i in HELD] for b in range(B)]) for m in METHODS}
    rows = []
    for m in METHODS:
        w = D[m].min(1); dw = D["CATT"].min(1) - w
        rows.append(dict(method=m, worst_boot_median=np.median(w), worst_lo=np.percentile(w, 2.5),
                         worst_hi=np.percentile(w, 97.5), diff_lo=np.percentile(dw, 2.5), diff_hi=np.percentile(dw, 97.5),
                         p_catt_worst_higher=float((dw > 0).mean())))
    df = pd.DataFrame(rows).round(3)
    df.to_csv(os.path.join(OUT, "worst_case_ci_5pct_patient_boot.csv"), index=False)
    print(df.to_string(index=False))
