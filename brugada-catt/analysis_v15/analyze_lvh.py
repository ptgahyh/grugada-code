"""Bước 15b: phân tích P1–P4 và F1–F2 của PLAN_vi.md trên điểm LVH (scores_<thiết lập>_<seed>.npz của run_lvh.py).

P1–P4 giữ nguyên code bước 14 (analysis_v14/analyze_rbbb.py); chỉ đổi danh sách mỏ neo.
Thêm (định trước ở bước 15):
  F1  ước lượng SD(t̂) không nhãn bằng mẫu con (common15.sd_subsample), trên cùng 20 mẫu con của P2
  F2  quy tắc dừng hai giai đoạn, giai đoạn 2 dùng mẫu con thay cho KDE

Chạy: python analyze_lvh.py <thư mục điểm> <out> {p1|p3|p4|summary}
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.mixture import GaussianMixture

from common15 import logit, n_from_c, n_needed, pseudo_terms, sd_formula, sd_from_c, sd_subsample, youden

SC, OUT, STAGE = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
DOMS = ["ptb_cs", "ptb_at", "cn"]
SETTINGS = [(a, b) for a in DOMS for b in DOMS if a != b]
NAME = {i: f"{a}→{b}" for i, (a, b) in enumerate(SETTINGS)}
NEG_CONTROL = {i for i, (a, b) in enumerate(SETTINGS) if "cn" not in (a, b)}
ANCH = {"A1 Sokolow": "a1", "A2 Cornell": "a2", "A3 LVH-LR": "a3", "A4 CNNalt": "alt"}
FILES = sorted(glob.glob(os.path.join(SC, "scores_*_*.npz")))


def bacc(y, s, t):
    p = s >= t
    return 0.5 * (p[y == 1].mean() + (~p)[y == 0].mean())


def load(fn):
    Z = np.load(fn); i, vs = map(int, os.path.basename(fn)[7:-4].split("_"))
    return i, vs, Z


def pseudo(Z, k):
    g = Z[f"{k}_tgt"]; t = youden(Z["y_val"], Z[f"{k}_val"])
    return (g >= t).astype(int)


def usable(pl, y):
    a, b = pl[y == 0].mean(), 1 - pl[y == 1].mean()
    return 1 - a - b > 0.05 and pl.sum() >= 10 and (1 - pl).sum() >= 10


# ---------------------------------------------------------------- P1, P2, F1: độ phân tán của t̂
def p1_job(fn):
    rng = np.random.default_rng(15)
    i, vs, Z = load(fn); y = Z["y_tgt"]; f = logit(Z["f_tgt"]); N = len(y)
    g = np.linspace(np.percentile(f, 0.5), np.percentile(f, 99.5), 3001)
    gp = np.linspace(np.percentile(f, 0.5), np.percentile(f, 99.5), 1201)
    rows = []
    for name, k in ANCH.items():
        pl = pseudo(Z, k)
        if not usable(pl, y):
            rows.append(dict(setting=i, vseed=vs, anchor=name, skipped="1-a-b<=0.05 hoặc <10 nhãn giả")); continue
        terms = {m: pseudo_terms(f, pl, m, g) for m in (0.5, 1.0, 2.0)}
        for n in (500, 1000, 2000):
            if n > 0.8 * N:
                continue
            th, npos = [], []
            for _ in range(300):
                idx = rng.choice(N, n, replace=False)
                if pl[idx].min() == pl[idx].max():
                    continue
                th.append(youden(pl[idx], f[idx])); npos.append(pl[idx].sum())
            plug, sub = [], []
            for _ in range(20):
                idx = rng.choice(N, n, replace=False)
                if pl[idx].sum() < 5 or (1 - pl[idx]).sum() < 5:
                    continue
                _, Vp, Ap = pseudo_terms(f[idx], pl[idx], 1.0, gp)
                plug.append(sd_formula(Vp, Ap, n, N))
                c = sd_subsample(f[idx], pl[idx], rng)
                if c is not None:
                    sub.append(sd_from_c(c, n, N))
            th = np.array(th)
            row = dict(setting=i, vseed=vs, anchor=name, N=N, n=n, mean_pseudo_pos=float(np.mean(npos)),
                       q=pl.mean(), sd_obs=th.std(), sd_plugin=float(np.median(plug)) if plug else np.nan,
                       sd_sub=float(np.median(sub)) if sub else np.nan)
            for m in (0.5, 1.0, 2.0):
                _, V, A = terms[m]
                row[f"sd_pred_{m}"] = sd_formula(V, A, n, N)
            rows.append(row)
    print("P1", i, vs, flush=True)
    return rows


# ---------------------------------------------------------------- P3, F2: quy tắc dừng
def p3_job(fn):
    rng = np.random.default_rng(1515)
    i, vs, Z = load(fn); y = Z["y_tgt"]; f = logit(Z["f_tgt"]); N = len(y); M = 300
    rows = []

    def est(idx):
        ff, pp = f[idx], pl[idx]
        if pp.sum() < 5 or (1 - pp).sum() < 5:
            return None
        gg = np.linspace(np.percentile(ff, 1), np.percentile(ff, 99), 1201)
        _, V, A = pseudo_terms(ff, pp, 1.0, gg)
        return V, A

    def achieved(n):
        th = []
        for _ in range(120):
            idx = rng.choice(N, n, replace=False)
            if pl[idx].min() != pl[idx].max():
                th.append(youden(pl[idx], f[idx]))
        return float(np.std(th))

    def record(base, rule, n):
        if n > 0.8 * N:
            rows.append(dict(base, rule=rule, status="không khả thi", n_plan=n))
        else:
            rows.append(dict(base, rule=rule, status="ok", n_plan=n, sd=achieved(n)))

    for name, k in ANCH.items():
        pl = pseudo(Z, k)
        if not usable(pl, y):
            continue
        for eps in (0.15, 0.30):
            for rep in range(6):
                pilot = rng.choice(N, M, replace=False)
                e = est(pilot)
                base = dict(setting=i, vseed=vs, anchor=name, eps=eps, rep=rep, N=N)
                if e is None:
                    for rule in ("1 giai đoạn", "2 giai đoạn", "2 giai đoạn (mẫu con)"):
                        rows.append(dict(base, rule=rule, status="mẫu thử < 5 nhãn giả"))
                    continue
                n1 = int(np.ceil(max(M, n_needed(*e, eps, N))))
                record(base, "1 giai đoạn", n1)
                nh = int(min(max(M, np.ceil(n1 / 2)), 0.8 * N))
                got = np.concatenate([pilot, rng.choice(np.setdiff1d(np.arange(N), pilot), nh - M, replace=False)])
                e2 = est(got) or e
                record(base, "2 giai đoạn", int(np.ceil(max(nh, n_needed(*e2, eps, N)))))
                c = sd_subsample(f[got], pl[got], rng)
                record(base, "2 giai đoạn (mẫu con)", int(np.ceil(max(nh, n_from_c(c, eps, N)))) if c is not None else n1)
    print("P3", i, vs, flush=True)
    return rows


# ---------------------------------------------------------------- P4: CATT so với các phương pháp không nhãn
def gmm_thr(s):
    z = logit(s)[:, None]
    gm = GaussianMixture(2, random_state=0, n_init=1, max_iter=200).fit(z)
    m = gm.means_.ravel(); sd = np.sqrt(gm.covariances_.ravel()); lo, hi = np.argsort(m)
    grid = np.linspace(m[lo], m[hi], 400)
    d = norm.pdf(grid, m[hi], sd[hi]) - norm.pdf(grid, m[lo], sd[lo])
    k = np.where(np.diff(np.sign(d)) != 0)[0]
    return 1 / (1 + np.exp(-(grid[k[0]] if len(k) else (m[lo] + m[hi]) / 2)))


def otsu_thr(s):
    z = logit(s); h, e = np.histogram(z, 64); c = (e[:-1] + e[1:]) / 2
    w = np.cumsum(h); w2 = w[-1] - w
    m1 = np.cumsum(h * c) / np.maximum(w, 1); m2 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w2, 1)
    return 1 / (1 + np.exp(-c[np.argmax(w * w2 * (m1 - m2) ** 2)]))


CATT = {f"CATT({a.split()[0]})": k for a, k in ANCH.items()}
METHODS = ["SRC", "MED", "UPA", "GMM-2", "Otsu", *CATT, "ORACLE"]


def p4_eval(Z, idx):
    y, st, yv, sv = Z["y_tgt"][idx], Z["f_tgt"][idx], Z["y_val"], Z["f_val"]
    t_src = youden(yv, sv)
    q1v, mv, q3v = np.percentile(sv, [25, 50, 75]); q1t, mt, q3t = np.percentile(st, [25, 50, 75])
    T = {"SRC": t_src, "MED": mt + (t_src - mv) * (q3t - q1t) / max(q3v - q1v, 1e-9),
         "UPA": np.quantile(st, (sv < t_src).mean()), "GMM-2": gmm_thr(st), "Otsu": otsu_thr(st)}
    for lab, k in CATT.items():
        pl = (Z[f"{k}_tgt"][idx] >= youden(yv, Z[f"{k}_val"])).astype(int)
        t = youden(pl, st) if 0 < pl.mean() < 1 else None
        T[lab] = t if t is not None else t_src
    T["ORACLE"] = youden(y, st)
    return {m: bacc(y, st, T[m]) for m in METHODS}


def p4_job(i):
    Zs = [np.load(os.path.join(SC, f"scores_{i}_{vs}.npz")) for vs in (0, 1, 2)]
    y = Zs[0]["y_tgt"]; N = len(y); allidx = np.arange(N)
    point = pd.DataFrame([p4_eval(Z, allidx) for Z in Zs]).mean().to_dict()
    rng = np.random.default_rng(150 + i); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = []
    for _ in range(1000):
        idx = np.concatenate([rng.choice(neg, len(neg)), rng.choice(pos, len(pos))])
        boots.append(pd.DataFrame([p4_eval(Z, idx) for Z in Zs]).mean().to_dict())
    print("P4", i, flush=True)
    return i, point, boots


def ratio_stats(d, col):
    r = np.log(d[col] / d.sd_obs).dropna()
    return dict(median_pred_over_obs=float(np.exp(np.median(r))), iqr=[float(np.exp(np.percentile(r, 25))),
                float(np.exp(np.percentile(r, 75)))], spearman=float(spearmanr(d[col], d.sd_obs, nan_policy="omit")[0]),
                n=int(r.size), by_n={int(n): float(np.exp(g.median())) for n, g in r.groupby(d.n)})


def crit(s):
    return bool(0.8 <= s["median_pred_over_obs"] <= 1.25 and s["spearman"] >= 0.7)


if __name__ == "__main__":
    if STAGE == "p1":
        with Pool(4) as p:
            rows = sum(p.map(p1_job, FILES), [])
        pd.DataFrame(rows).round(6).to_csv(os.path.join(OUT, "p1_spread.csv"), index=False)
    elif STAGE == "p3":
        with Pool(4) as p:
            rows = sum(p.map(p3_job, FILES), [])
        pd.DataFrame(rows).round(6).to_csv(os.path.join(OUT, "p3_stopping.csv"), index=False)
    elif STAGE == "p4":
        with Pool(4) as p:
            res = p.map(p4_job, range(len(SETTINGS)))
        pt = pd.DataFrame({NAME[i]: point for i, point, _ in res}).T
        pt.round(4).to_csv(os.path.join(OUT, "p4_point.csv"))
        bt = pd.concat([pd.DataFrame(b).assign(setting=NAME[i]) for i, _, b in res])
        bt.round(5).to_csv(os.path.join(OUT, "p4_boot.csv"), index=False)
    else:
        S = {}
        D = pd.read_csv(os.path.join(OUT, "p1_spread.csv"))
        Dk = D.dropna(subset=["sd_obs"]) if "sd_obs" in D else D.iloc[:0]
        Dk = Dk[Dk.mean_pseudo_pos >= 5]
        S["P1"] = {"all": ratio_stats(Dk, "sd_pred_1.0"),
                   **{f"anchor {a}": ratio_stats(Dk[Dk.anchor == a], "sd_pred_1.0") for a in ANCH if (Dk.anchor == a).any()},
                   **{f"bandwidth x{m}": ratio_stats(Dk, f"sd_pred_{m}") for m in (0.5, 2.0)},
                   "skipped": D[D["skipped"].notna()][["setting", "vseed", "anchor"]].to_dict("records")
                   if "skipped" in D else []}
        y = np.log(Dk.sd_obs); nn = Dk.n / (1 - Dk.n / Dk.N); base = -np.log(nn) / 3
        S["P1"]["r2_log_formula"] = float(1 - np.var(y - np.log(Dk["sd_pred_1.0"])) / np.var(y))
        S["P1"]["r2_log_n_only"] = float(1 - np.var(y - base - np.mean(y - base)) / np.var(y))
        S["P1"]["criterion_met"] = crit(S["P1"]["all"])
        S["P2"] = ratio_stats(Dk, "sd_plugin"); S["P2"]["criterion_met"] = crit(S["P2"])
        S["F1"] = ratio_stats(Dk, "sd_sub"); S["F1"]["criterion_met"] = crit(S["F1"])
        S["F1"]["abs_log_error_median"] = {c: float(np.median(np.abs(np.log(Dk[c] / Dk.sd_obs)).dropna()))
                                          for c in ("sd_plugin", "sd_sub")}
        P = pd.read_csv(os.path.join(OUT, "p3_stopping.csv"))
        S["P3"] = {}
        for (rule, eps), d in P.groupby(["rule", "eps"]):
            ok = d[d.status == "ok"]
            S["P3"][f"{rule}, eps={eps}"] = dict(
                n_plans=len(d), feasible=len(ok), median_sd_over_eps=float(np.median(ok.sd / eps)),
                frac_le_eps=float(np.mean(ok.sd <= eps)), frac_le_1_2eps=float(np.mean(ok.sd <= 1.2 * eps)),
                median_n_plan=float(np.median(ok.n_plan)),
                safety_factor_80pct=float(np.percentile(np.maximum(1, (ok.sd / eps) ** 3), 80)) if len(ok) else None)
        for rule, key in (("2 giai đoạn", "criterion_met_two_stage"), ("2 giai đoạn (mẫu con)", "criterion_met_F2")):
            S["P3"][key] = bool(all(v["frac_le_1_2eps"] >= 0.8 for k, v in S["P3"].items()
                                    if isinstance(v, dict) and k.startswith(rule + ", ")))
        pt = pd.read_csv(os.path.join(OUT, "p4_point.csv"), index_col=0)
        bt = pd.read_csv(os.path.join(OUT, "p4_boot.csv"))
        broken = [s for s in pt.index if pt.loc[s, "SRC"] < pt.loc[s, "ORACLE"] - 0.05]
        S["P4"] = dict(point=pt.round(4).to_dict("index"), broken_settings=broken,
                       negative_control_settings=[NAME[i] for i in NEG_CONTROL])
        diffs = {}
        for c in ("CATT(A1)", "CATT(A2)", "CATT(A3)"):
            for m in ("SRC", "MED", "UPA", "GMM-2", "Otsu", "CATT(A4)"):
                for lab, sets in (("hỏng", broken), ("tất cả", list(pt.index))):
                    if not sets:
                        continue
                    v = np.mean([bt[bt.setting == s][c].to_numpy() - bt[bt.setting == s][m].to_numpy() for s in sets], 0)
                    p0 = np.mean([pt.loc[s, c] - pt.loc[s, m] for s in sets])
                    diffs[f"{c} − {m} ({lab})"] = [round(float(p0), 4), round(float(np.percentile(v, 2.5)), 4),
                                                    round(float(np.percentile(v, 97.5)), 4)]
        S["P4"]["pooled_differences_point_ci95"] = diffs
        S["P4"]["oracle_gap_broken"] = {s: {c: round(float(pt.loc[s, "ORACLE"] - pt.loc[s, c]), 4) for c in CATT}
                                        for s in broken}
        json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1, ensure_ascii=False)
        print(json.dumps(S, indent=1, ensure_ascii=False))
