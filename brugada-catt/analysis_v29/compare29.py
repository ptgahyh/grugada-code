"""Bước 29: bảng "cũ → mới" (pipeline gốc → pipeline lọc chung). Chạy: python analysis_v29/compare29.py
Cùng một hàm tính trên CSV/JSON cũ (analysis_vXX/) và mới (analysis_v29/vXX/), nên định nghĩa hai cột trùng nhau."""
import json
import os

import numpy as np
import pandas as pd

H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..")
rows = []


def flat(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{pre}{k}/"))
    elif isinstance(d, list):
        if all(isinstance(v, (int, float, bool)) or v is None for v in d) and len(d) <= 4:
            out[pre[:-1]] = d
        else:
            for k, v in enumerate(d):
                out.update(flat(v, f"{pre}{k}/"))
    else:
        out[pre[:-1]] = d
    return out


def add(step, key, old, new):
    rows.append(dict(step=step, key=key, old=old, new=new))


def jsons(step, old, new, only=None):
    if not (os.path.exists(old) and os.path.exists(new)):
        return
    a, b = flat(json.load(open(old))), flat(json.load(open(new)))
    for k in a:
        if k in b and (only is None or any(s in k for s in only)):
            add(step, k, a[k], b[k])


def v18(d):
    S = pd.read_csv(os.path.join(d, "est_brugada_single.csv")); o = {}
    for c in [c for c in S.columns if c.startswith("E")]:
        r = np.log(S[c] / S.sd_obs).replace([np.inf, -np.inf], np.nan).dropna()
        o[f"{c}: trung vị |log(ước lượng/SD quan sát)| từng mẫu đơn"] = round(float(r.abs().median()), 3)
        o[f"{c}: trung vị tỉ số"] = round(float(np.exp(r.median())), 3)
    return o


def v20(d):
    P = pd.read_csv(os.path.join(d, "smooth_brugada_spread.csv")); A = pd.read_csv(os.path.join(d, "smooth_brugada_acc.csv"))
    h = A[A.setting >= 2].groupby("setting").mean(numeric_only=True).mean()
    return {"S2a trung vị SD làm trơn/thường (đạt ≤ 0,85)": round(float((P["sd_s1.0"] / P.sd_raw).median()), 3),
            "S2b n=50: BA làm trơn − thường (đạt ≥ −0,005)": round(float(h["ba_n50_s1.0"] - h.ba_n50_raw), 4),
            "S2b n=100: BA làm trơn − thường (đạt ≥ −0,005)": round(float(h["ba_n100_s1.0"] - h.ba_n100_raw), 4),
            "S2c toàn cohort: BA làm trơn − thường (đạt |.| ≤ 0,010)": round(float(h["ba_full_s1.0"] - h.ba_full_raw), 4),
            "BA n=50 thường": round(float(h.ba_n50_raw), 4), "BA n=50 làm trơn": round(float(h["ba_n50_s1.0"]), 4),
            "P10 n=50 thường": round(float(h.p10_n50_raw), 4), "P10 n=50 làm trơn": round(float(h["p10_n50_s1.0"]), 4),
            "BA toàn cohort thường": round(float(h.ba_full_raw), 4), "BA toàn cohort làm trơn": round(float(h["ba_full_s1.0"]), 4)}


for step, fn in (("18c", v18), ("20", v20)):
    try:
        a, b = fn(os.path.join(R, f"analysis_v{step[:2]}")), fn(os.path.join(H, f"v{step[:2]}"))
        for k in a:
            add(step, k, a[k], b[k])
    except FileNotFoundError:
        pass
jsons("13", os.path.join(R, "analysis_v13", "brugada_unified_v2_summary.json"), os.path.join(H, "v13", "brugada_unified_v2_summary.json"))
jsons("13 (quy tắc dừng)", os.path.join(R, "analysis_v13", "stopping_rule_summary.json"), os.path.join(H, "v13", "stopping_rule_summary.json"))
jsons("16", os.path.join(R, "analysis_v16", "sens_summary.json"), os.path.join(H, "v16", "sens_summary.json"))
jsons("19", os.path.join(R, "analysis_v19", "summary_v19.json"), os.path.join(H, "v19", "summary_v19.json"))
jsons("19 (căn lại, quan sát sau)", os.path.join(R, "analysis_v19", "realign_posthoc.json"), os.path.join(H, "v19", "realign_posthoc.json"))
jsons("24", os.path.join(R, "analysis_v24", "summary_v24.json"), os.path.join(H, "v24", "summary_v24.json"), only=["T5", "T6", "brugada", "Brugada"])
jsons("25", os.path.join(R, "analysis_v25", "summary_v25.json"), os.path.join(H, "v25", "summary_v25.json"), only=["M1", "M2", "M3", "M4", "brugada", "Brugada"])
jsons("26", os.path.join(R, "analysis_v26", "summary_v26_brugada.json"), os.path.join(H, "v26", "summary_v26_brugada.json"))
D = pd.DataFrame(rows); D["changed"] = [str(a) != str(b) for a, b in zip(D.old, D.new)]
D["flip"] = [isinstance(a, bool) and isinstance(b, bool) and a != b for a, b in zip(D.old, D.new)]
D.to_csv(os.path.join(H, "compare29_all.csv"), index=False)
print(len(D), "dòng;", int(D.flip.sum()), "tiêu chí đổi trạng thái đạt/không đạt")
print(D[D.flip].to_string())
