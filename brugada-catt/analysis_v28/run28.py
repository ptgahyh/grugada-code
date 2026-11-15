"""Bước 28: ngưỡng làm trơn, quy tắc cỡ mẫu (hệ số an toàn 1,5) và cổng trên RBBB/LVH. Xem PLAN_vi.md.

Chạy: python analysis_v28/run28.py <thư mục điểm> <thư mục ra> <rbbb|lvh|lvhmv> <a|b|c|summary>
  a: 28a (độ dao động)   b: 28b (quy tắc cỡ mẫu)   c: 28c + 28e (cổng, BA; bootstrap bệnh nhân)   summary: tổng hợp
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis_v24"))
from lib24 import c_const, estimators, kde_terms, n_needed_smooth, smooth_thr  # noqa: E402

SC, OUT, PROB, STAGE = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
ANCH = {"rbbb": ["a1", "a2", "alt"], "lvh": ["a1", "a2", "a3", "alt"], "lvhmv": ["a1", "a2", "a3", "alt"]}[PROB]
PRIMARY = {"rbbb": "a2", "lvh": "a3", "lvhmv": "a3"}[PROB]
NPROC = int(os.environ.get("NPROC", 2))
NBOOT = int(os.environ.get("NBOOT", 1000))
FILES = sorted(glob.glob(os.path.join(SC, "scores_*.npz")))
NS, EPS, ZS, MIN_PL = (500, 1000, 2000), (0.15, 0.30), (1, 2, 3), 5


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def youden(y, s):
    if y.min() == y.max():
        return None
    fpr, tpr, thr = roc_curve(y, s)
    return np.minimum(thr, s.max() + 1e-9)[np.argmax(tpr - fpr)]


def bacc(y, s, t):
    p = s >= t
    return 0.5 * (p[y == 1].mean() + (~p)[y == 0].mean())


def load(fn):
    i, vs = map(int, os.path.basename(fn)[7:-4].split("_")); Z = np.load(fn)
    f, y = logit(Z["f_tgt"]), Z["y_tgt"].astype(int)
    t_src = youden(Z["y_val"].astype(int), logit(Z["f_val"]))
    pls = {a: (Z[f"{a}_tgt"] >= youden(Z["y_val"].astype(int), Z[f"{a}_val"])).astype(float) for a in ANCH}
    return i, vs, f, y, t_src, pls


def ok_pl(pl):
    s = pl.sum()
    return MIN_PL <= s <= len(pl) - MIN_PL


def g2(f, pl):
    return estimators(f, pl, None, boot=False)["G2_sandwich"]


# ---------------------------------------------------------------- 28a
def job_a(fn):
    i, vs, f, y, t_src, pls = load(fn); N = len(f); rng = np.random.default_rng(2801 + 10 * i + vs); rows = []
    for a in ANCH:
        pl = pls[a]
        for n in NS:
            if n > 0.8 * N:
                continue
            th, tr, skipped = [], [], 0
            for _ in range(300):
                k = rng.choice(N, n, replace=False)
                if not ok_pl(pl[k]):
                    skipped += 1; continue
                th.append(smooth_thr(f[k], pl[k])); tr.append(youden(pl[k].astype(int), f[k]))
            if len(th) < 100:
                rows.append(dict(setting=i, vseed=vs, anchor=a, n=n, N=N, sample=-1, skipped=skipped)); continue
            sd_h, sd_r = float(np.std(th)), float(np.std(tr)); s = 0
            while s < 20:
                k = rng.choice(N, n, replace=False)
                if not ok_pl(pl[k]):
                    continue
                rows.append(dict(setting=i, vseed=vs, anchor=a, n=n, N=N, sample=s, skipped=skipped, sd_obs_smooth=sd_h, sd_obs_raw=sd_r,
                                 g2=g2(f[k], pl[k]) * np.sqrt(1 - n / N))); s += 1
    print("28a", i, vs, flush=True)
    return rows


# ---------------------------------------------------------------- 28b
def job_b(fn):
    i, vs, f, y, t_src, pls = load(fn); N = len(f); rng = np.random.default_rng(2851 + 10 * i + vs); rows = []; M = 300

    def achieved(pl, n):
        th = []
        for _ in range(120):
            k = rng.choice(N, n, replace=False)
            if ok_pl(pl[k]):
                th.append(smooth_thr(f[k], pl[k]))
        return float(np.std(th)) if len(th) >= 60 else np.nan

    for a in ANCH:
        pl = pls[a]
        for rep in range(6):
            k = rng.choice(N, M, replace=False); fp, pp = f[k], pl[k]
            for eps in EPS:
                base = dict(setting=i, vseed=vs, anchor=a, eps=eps, rep=rep, N=N)
                if not ok_pl(pp):
                    rows += [dict(base, sf=sf, status="mẫu thử < 5 nhãn giả") for sf in (1.5, 1.0)]; continue
                o = np.argsort(fp); V, A = kde_terms(fp[o], pp[o]); q = pp.mean()
                n0 = n_needed_smooth(V, A, c_const(fp), min(q, 1 - q), eps)
                for sf in (1.5, 1.0):
                    ninf = sf * n0
                    if not np.isfinite(ninf):
                        rows.append(dict(base, sf=sf, status="không ước lượng được")); continue
                    npl = int(max(M, np.ceil(ninf / (1 + ninf / N))))
                    if npl > 0.8 * N:
                        rows.append(dict(base, sf=sf, status="không khả thi", n_plan=npl)); continue
                    rows.append(dict(base, sf=sf, status="ok", n_plan=npl, sd=achieved(pl, npl)))
    print("28b", i, vs, flush=True)
    return rows


# ---------------------------------------------------------------- 28c + 28e
def thresholds(f, pl, with_raw_sd):
    """(t_c, σ̂ thường hoặc nan, t_h, σ̂ G2) trên một lô không nhãn."""
    tc = youden(pl.astype(int), f); e = estimators(f, pl, None, boot=False) if not with_raw_sd else None
    if with_raw_sd:
        o = np.argsort(f); V, A = kde_terms(f[o], pl[o]); sc = 0.5134 * (4 * V / (A ** 2 * len(f))) ** (1 / 3)
        e = estimators(f, pl, None, boot=False)
    else:
        sc = np.nan
    return tc, sc, e["t"], e["G2_sandwich"]


def gate_row(y, f, pl, t_src, with_raw_sd):
    tc, sc, th, sh = thresholds(f, pl, with_raw_sd)
    r = dict(ba_src=bacc(y, f, t_src), ba_oracle=bacc(y, f, youden(y, f)), ba_catt=bacc(y, f, tc), ba_catt_h=bacc(y, f, th),
             gap_smooth_over_sd=abs(th - t_src) / sh)
    if with_raw_sd:
        r["gap_raw_over_sd"] = abs(tc - t_src) / sc
    for z in ZS:
        op = int(abs(th - t_src) > z * sh); r[f"open_smooth_z{z}"] = op; r[f"ba_gate_smooth_z{z}"] = bacc(y, f, th if op else t_src)
        if with_raw_sd:
            op = int(abs(tc - t_src) > z * sc); r[f"open_raw_z{z}"] = op; r[f"ba_gate_raw_z{z}"] = bacc(y, f, tc if op else t_src)
    return r


def job_c(fn):
    i, vs, f, y, t_src, pls = load(fn); N = len(f); rng = np.random.default_rng(2871 + 10 * i + vs)
    point = [dict(setting=i, vseed=vs, anchor=a, N=N, **gate_row(y, f, pls[a], t_src, True)) for a in ANCH]
    # 28e: n = 500 không nhãn, mỏ neo chính
    pl = pls[PRIMARY]; br, bh = [], []
    for _ in range(300):
        k = rng.choice(N, 500, replace=False)
        if not ok_pl(pl[k]):
            br.append(bacc(y, f, t_src)); bh.append(bacc(y, f, t_src)); continue
        br.append(bacc(y, f, youden(pl[k].astype(int), f[k]))); bh.append(bacc(y, f, smooth_thr(f[k], pl[k])))
    point[ANCH.index(PRIMARY)].update(ba500_catt=np.mean(br), p10_500_catt=np.percentile(br, 10), ba500_catt_h=np.mean(bh),
                                      p10_500_catt_h=np.percentile(bh, 10))
    # bootstrap bệnh nhân phân tầng theo nhãn (như P4), mỏ neo chính, bản làm trơn (bản thường: chỉ ước lượng điểm, xem README)
    rb = np.random.default_rng(280 + i); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; boots = []
    for b in range(NBOOT):
        idx = np.concatenate([rb.choice(neg, len(neg)), rb.choice(pos, len(pos))])
        boots.append(dict(setting=i, vseed=vs, b=b, **gate_row(y[idx], f[idx], pl[idx], t_src, False)))
    print("28c", i, vs, flush=True)
    return point, boots


def frac_ok(d, eps):
    return float(np.mean(d.sd <= 1.2 * eps)) if len(d) else None


if __name__ == "__main__":
    if STAGE in "abc":
        with Pool(NPROC) as p:
            res = p.map({"a": job_a, "b": job_b, "c": job_c}[STAGE], FILES)
        if STAGE == "a":
            pd.DataFrame(sum(res, [])).round(6).to_csv(os.path.join(OUT, f"a_spread_{PROB}.csv"), index=False)
        elif STAGE == "b":
            pd.DataFrame(sum(res, [])).round(6).to_csv(os.path.join(OUT, f"b_plans_{PROB}.csv"), index=False)
        else:
            pd.DataFrame(sum([r[0] for r in res], [])).round(6).to_csv(os.path.join(OUT, f"c_gate_{PROB}.csv"), index=False)
            pd.DataFrame(sum([r[1] for r in res], [])).round(5).to_csv(os.path.join(OUT, f"c_boot_{PROB}.csv.gz"), index=False)
        sys.exit()
    S = {"problem": PROB, "primary_anchor": PRIMARY}
    # ---- 28a
    A = pd.read_csv(os.path.join(OUT, f"a_spread_{PROB}.csv")); S["28a"] = {}
    for a in ANCH:
        d = A[(A.anchor == a) & (A["sample"] >= 0)]
        if d.empty:
            continue
        r = np.log(d.g2 / d.sd_obs_smooth).replace([np.inf, -np.inf], np.nan).dropna()
        cells = d.groupby(["setting", "vseed", "n"]).first().reset_index()
        slope = lambda col: float(np.mean([np.polyfit(np.log(g.n), np.log(g[col]), 1)[0] for _, g in cells.groupby(["setting", "vseed"]) if len(g) >= 2]))
        S["28a"][a] = dict(R1_median_ratio=float(np.exp(r.median())), R1_median_abs_log=float(r.abs().median()), n_single=int(len(r)),
                           R1_by_n={int(n): float(np.exp(np.log(g.g2 / g.sd_obs_smooth).median())) for n, g in d.groupby("n")},
                           R2_median_sd_ratio=float((cells.sd_obs_smooth / cells.sd_obs_raw).median()), n_cells=int(len(cells)),
                           R3_slope_smooth=slope("sd_obs_smooth"), R3_slope_raw=slope("sd_obs_raw"),
                           skipped_subsamples=int(A[A.anchor == a].groupby(["setting", "vseed", "n"]).skipped.first().sum()),
                           cells_dropped=int((A[A.anchor == a]["sample"] == -1).sum()))
    p = S["28a"].get(PRIMARY, {})
    S["28a"]["R1_met"] = bool(p and 0.8 <= p["R1_median_ratio"] <= 1.25 and p["R1_median_abs_log"] <= 0.20)
    S["28a"]["R2_met"] = bool(p and p["R2_median_sd_ratio"] <= 0.80)
    # ---- 28b
    B = pd.read_csv(os.path.join(OUT, f"b_plans_{PROB}.csv")); S["28b"] = {}
    for a in ANCH:
        S["28b"][a] = {}
        for (sf, eps), d in B[B.anchor == a].groupby(["sf", "eps"]):
            ok = d[(d.status == "ok") & d.sd.notna()]
            S["28b"][a][f"sf={sf}, eps={eps}"] = dict(n_plans=int(len(d)), feasible=int(len(ok)), frac_le_1_2eps=frac_ok(ok, eps),
                                                      median_n_plan=float(ok.n_plan.median()) if len(ok) else None,
                                                      median_sd_over_eps=float((ok.sd / eps).median()) if len(ok) else None,
                                                      extra_factor_for_80pct=float(np.percentile(np.maximum(1, (ok.sd / (1.2 * eps)) ** 2.5), 80)) if len(ok) else None)
    c1 = [S["28b"][PRIMARY].get(f"sf=1.5, eps={e}", {}).get("frac_le_1_2eps") for e in EPS]
    S["28b"]["C1_met"] = bool(all(v is not None and v >= 0.8 for v in c1))
    # ---- 28c + 28e
    G = pd.read_csv(os.path.join(OUT, f"c_gate_{PROB}.csv")); Bt = pd.read_csv(os.path.join(OUT, f"c_boot_{PROB}.csv.gz")); S["28c"] = {}
    for a in ANCH:
        g = G[G.anchor == a]; m = g.groupby("setting").mean(numeric_only=True); e = {}
        e["ba"] = {c: round(float(m[c].mean()), 4) for c in ("ba_src", "ba_catt", "ba_catt_h", "ba_oracle")}
        e["broken_settings"] = [int(s) for s in m.index if m.loc[s, "ba_src"] < m.loc[s, "ba_oracle"] - 0.05]
        for kind, cc in (("raw", "ba_catt"), ("smooth", "ba_catt_h")):
            for z in ZS:
                e[f"{kind}_z{z}"] = dict(gate_minus_src=round(float((m[f"ba_gate_{kind}_z{z}"] - m.ba_src).mean()), 4),
                                         catt_minus_src=round(float((m[cc] - m.ba_src).mean()), 4), opened=int(g[f"open_{kind}_z{z}"].sum()), of=int(len(g)))
        e["gap_over_sd_by_setting"] = {k: {str(s): round(float(v), 2) for s, v in m[f"gap_{k}_over_sd"].items()} for k in ("raw", "smooth")}
        S["28c"][a] = e
    bm = Bt.groupby(["b", "setting"]).mean(numeric_only=True).groupby("b").mean()       # trung bình seed, rồi trung bình thiết lập
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    S["28c"]["bootstrap_primary_smooth"] = {**{f"gate_z{z} − SRC": ci(bm[f"ba_gate_smooth_z{z}"] - bm.ba_src) for z in ZS},
                                            "CATT_h − SRC": ci(bm.ba_catt_h - bm.ba_src), "CATT − SRC": ci(bm.ba_catt - bm.ba_src),
                                            "CATT_h − CATT": ci(bm.ba_catt_h - bm.ba_catt)}
    S["28c"]["G3_point"] = S["28c"][PRIMARY]["smooth_z2"]["gate_minus_src"]; S["28c"]["G3_met"] = bool(S["28c"]["G3_point"] >= -0.003)
    gp = G[G.anchor == PRIMARY]
    S["28e"] = {c: round(float(gp.groupby("setting")[c].mean().mean()), 4) for c in ("ba500_catt", "p10_500_catt", "ba500_catt_h", "p10_500_catt_h")}
    json.dump(S, open(os.path.join(OUT, f"summary_v28_{PROB}.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(S, indent=1, ensure_ascii=False))
