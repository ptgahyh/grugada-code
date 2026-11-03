"""Bước 15a (KHÁM PHÁ, không định trước): thử ước lượng SD(t̂) bằng mẫu con trên dữ liệu RBBB của bước 14.

Mục đích: xem bản sửa có đáng đăng ký trước để kiểm định thật trên LVH (bước 15b) không.
Kết quả ở đây KHÔNG được dùng làm bằng chứng xác nhận, vì công thức được chọn sau khi đã thấy dữ liệu RBBB.

E1  Hệ quả 5c bản mẫu con: với mỗi (thiết lập, seed, mỏ neo, n), 20 mẫu cỡ n không nhãn; c từ sd_subsample (B = 100,
    b = n/2); dự đoán SD(t̂_n) = c·((1 − n/N)/n)^{1/3}; so với SD quan sát (cột sd_obs của p1_spread.csv bước 14).
E2  Quy tắc dừng hai giai đoạn bản mẫu con: mẫu thử 300 → n₁ theo công thức (Ṽ, Ã) như bước 14; thu n_h = max(300, ⌈n₁/2⌉);
    ước lượng c bằng mẫu con trên n_h; n₂ = n giải từ c; n cuối = max(n_h, n₂); 120 lần thu để đo SD đạt được.

Chạy: python explore_rbbb.py <scores bước 14> <p1_spread.csv bước 14> <out>
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common15 import logit, n_from_c, n_needed, pseudo_terms, sd_from_c, sd_subsample, youden

SC, P1, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
ANCH = {"A1 QRS_rms": "a1", "A2 RBBB-LR": "a2", "A3 CNNalt": "alt"}
FILES = sorted(glob.glob(os.path.join(SC, "scores_*_*.npz")))


def setup(fn):
    Z = np.load(fn); i, vs = map(int, os.path.basename(fn)[7:-4].split("_"))
    return i, vs, Z, Z["y_tgt"], logit(Z["f_tgt"])


def pseudo(Z, k):
    return (Z[f"{k}_tgt"] >= youden(Z["y_val"], Z[f"{k}_val"])).astype(int)


def usable(pl, y):
    a, b = pl[y == 0].mean(), 1 - pl[y == 1].mean()
    return 1 - a - b > 0.05 and pl.sum() >= 10 and (1 - pl).sum() >= 10


def e1_job(fn):
    rng = np.random.default_rng(151)
    i, vs, Z, y, f = setup(fn); N = len(y); rows = []
    for name, k in ANCH.items():
        pl = pseudo(Z, k)
        if not usable(pl, y):
            continue
        for n in (500, 1000, 2000):
            if n > 0.8 * N:
                continue
            preds = []
            for _ in range(20):
                idx = rng.choice(N, n, replace=False)
                c = sd_subsample(f[idx], pl[idx], rng)
                if c is not None:
                    preds.append(sd_from_c(c, n, N))
            rows.append(dict(setting=i, vseed=vs, anchor=name, n=n, sd_sub=float(np.median(preds)),
                             sd_sub_q10=float(np.percentile(preds, 10)), sd_sub_q90=float(np.percentile(preds, 90))))
    print("E1", i, vs, flush=True)
    return rows


def e2_job(fn):
    rng = np.random.default_rng(1515)
    i, vs, Z, y, f = setup(fn); N = len(y); M = 300; rows = []

    def achieved(n, pl):
        th = []
        for _ in range(120):
            idx = rng.choice(N, n, replace=False)
            if pl[idx].min() != pl[idx].max():
                th.append(youden(pl[idx], f[idx]))
        return float(np.std(th))

    for name, k in ANCH.items():
        pl = pseudo(Z, k)
        if not usable(pl, y):
            continue
        for eps in (0.15, 0.30):
            for rep in range(6):
                base = dict(setting=i, vseed=vs, anchor=name, eps=eps, rep=rep, N=N)
                pilot = rng.choice(N, M, replace=False)
                if pl[pilot].sum() < 5 or (1 - pl[pilot]).sum() < 5:
                    rows.append(dict(base, status="mẫu thử < 5 nhãn giả")); continue
                gg = np.linspace(np.percentile(f[pilot], 1), np.percentile(f[pilot], 99), 1201)
                _, V, A = pseudo_terms(f[pilot], pl[pilot], 1.0, gg)
                n1 = int(np.ceil(max(M, n_needed(V, A, eps, N))))
                nh = int(min(max(M, np.ceil(n1 / 2)), 0.8 * N))
                got = np.concatenate([pilot, rng.choice(np.setdiff1d(np.arange(N), pilot), nh - M, replace=False)])
                c = sd_subsample(f[got], pl[got], rng)
                n2 = int(np.ceil(max(nh, n_from_c(c, eps, N)))) if c is not None else n1
                if n2 > 0.8 * N:
                    rows.append(dict(base, status="không khả thi", n_plan=n2)); continue
                rows.append(dict(base, status="ok", n_plan=n2, sd=achieved(n2, pl)))
    print("E2", i, vs, flush=True)
    return rows


if __name__ == "__main__":
    with Pool(4) as p:
        e1 = pd.DataFrame(sum(p.map(e1_job, FILES), []))
        e2 = pd.DataFrame(sum(p.map(e2_job, FILES), []))
    e1.round(6).to_csv(os.path.join(OUT, "explore_e1.csv"), index=False)
    e2.round(6).to_csv(os.path.join(OUT, "explore_e2.csv"), index=False)
    D = pd.read_csv(P1).merge(e1, on=["setting", "vseed", "anchor", "n"])
    S = {}
    for col in ("sd_pred_1.0", "sd_plugin", "sd_sub"):
        r = np.log(D[col] / D.sd_obs)
        S[col] = dict(median_ratio=float(np.exp(r.median())), iqr=[float(np.exp(r.quantile(.25))), float(np.exp(r.quantile(.75)))],
                      spearman=float(spearmanr(D[col], D.sd_obs)[0]),
                      by_n={int(n): float(np.exp(g.median())) for n, g in r.groupby(D.n)},
                      by_anchor={a: float(np.exp(g.median())) for a, g in r.groupby(D.anchor)})
    ok = e2[e2.status == "ok"]
    S["E2"] = {f"eps={e}": dict(n_plans=int((e2.eps == e).sum()), feasible=len(d), frac_le_1_2eps=float(np.mean(d.sd <= 1.2 * e)),
                                frac_le_eps=float(np.mean(d.sd <= e)), median_n_plan=float(d.n_plan.median()),
                                safety_factor_80pct=float(np.percentile(np.maximum(1, (d.sd / e) ** 3), 80)),
                                by_anchor={a: float(np.mean(g.sd <= 1.2 * e)) for a, g in d.groupby("anchor")})
               for e, d in ok.groupby("eps")}
    json.dump(S, open(os.path.join(OUT, "explore_summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(S, indent=1, ensure_ascii=False))
