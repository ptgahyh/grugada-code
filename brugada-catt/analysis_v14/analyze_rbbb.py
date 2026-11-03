"""Bước 14: phân tích P1–P4 của PLAN_vi.md trên điểm RBBB (scores_<thiết lập>_<seed>.npz của run_rbbb.py).

Ước lượng giữ nguyên bước 13: Ṽ, Ã, t̃ từ KDE băng thông Scott (hệ số m) của logit f trong từng lớp nhãn giả
(`pseudo_terms` như brugada_unified_v2.py); hiệu chỉnh quần thể hữu hạn (1 − n/N); quy tắc dừng
n = n_∞/(1 + n_∞/N), n_∞ = 4·SD(Z)³·Ṽ/(Ã²ε³) (như stopping_rule.py).

Chạy: python analyze_rbbb.py <thư mục điểm> <chernoff_samples.npy> <out> {p1|p3|p4|summary}
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm, spearmanr
from sklearn.metrics import roc_curve
from sklearn.mixture import GaussianMixture

SC, CHER, OUT, STAGE = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
ZS = np.load(CHER).astype(float)
SDZ = float(np.sqrt(0.26355))
DOMS = ["ptb_cs", "ptb_at", "cn"]
SETTINGS = [(a, b) for a in DOMS for b in DOMS if a != b]
NAME = {i: f"{a}→{b}" for i, (a, b) in enumerate(SETTINGS)}
NEG_CONTROL = {i for i, (a, b) in enumerate(SETTINGS) if "cn" not in (a, b)}
ANCH = {"A1 QRS_rms": "a1", "A2 RBBB-LR": "a2", "A3 CNNalt": "alt"}
FILES = sorted(glob.glob(os.path.join(SC, "scores_*_*.npz")))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def youden(y, s):
    if y.min() == y.max():
        return None
    fpr, tpr, thr = roc_curve(y, s)
    thr = np.minimum(thr, s.max() + 1e-9)
    return thr[np.argmax(tpr - fpr)]


def bacc(y, s, t):
    p = s >= t
    return 0.5 * (p[y == 1].mean() + (~p)[y == 0].mean())


def kde(x, m):
    return gaussian_kde(x, bw_method=lambda k: m * k.scotts_factor())


def pseudo_terms(f, pl, m, g):
    q = pl.mean(); dg = g[1] - g[0]
    d1, d0 = kde(f[pl == 1], m)(g), kde(f[pl == 0], m)(g)
    i = int(np.argmax(np.cumsum(d0 - d1))); i = min(max(i, 2), len(g) - 3)
    return g[i], d1[i] / q + d0[i] / (1 - q), abs(np.gradient(d0 - d1, dg)[i])


def n_needed(V, A, eps, N):
    n_inf = 4 * SDZ ** 3 * V / (A ** 2 * eps ** 3)
    return n_inf / (1 + n_inf / N)


def load(fn):
    Z = np.load(fn); i, vs = map(int, os.path.basename(fn)[7:-4].split("_"))
    return i, vs, Z


def pseudo(Z, k):
    g = Z[f"{k}_tgt"]; t = youden(Z["y_val"], Z[f"{k}_val"])
    return (g >= t).astype(int)


def usable(pl, y):
    a, b = pl[y == 0].mean(), 1 - pl[y == 1].mean()
    return 1 - a - b > 0.05 and pl.sum() >= 10 and (1 - pl).sum() >= 10


# ---------------------------------------------------------------- P1, P2: độ phân tán của t̂
def p1_job(fn):
    rng = np.random.default_rng(14)
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
            fpc = 1 - n / N
            th, npos = [], []
            for _ in range(300):
                idx = rng.choice(N, n, replace=False)
                if pl[idx].min() == pl[idx].max():
                    continue
                th.append(youden(pl[idx], f[idx])); npos.append(pl[idx].sum())
            plug = []
            for _ in range(20):
                idx = rng.choice(N, n, replace=False)
                if pl[idx].sum() < 5 or (1 - pl[idx]).sum() < 5:
                    continue
                _, Vp, Ap = pseudo_terms(f[idx], pl[idx], 1.0, gp)
                plug.append((4 * Vp * fpc / (Ap ** 2 * n)) ** (1 / 3) * SDZ)
            th = np.array(th)
            row = dict(setting=i, vseed=vs, anchor=name, N=N, n=n, mean_pseudo_pos=float(np.mean(npos)),
                       q=pl.mean(), sd_obs=th.std(), sd_plugin=float(np.median(plug)) if plug else np.nan)
            for m in (0.5, 1.0, 2.0):
                tt, V, A = terms[m]
                row[f"sd_pred_{m}"] = (4 * V * fpc / (A ** 2 * n)) ** (1 / 3) * SDZ
            rows.append(row)
    print("P1", i, vs, flush=True)
    return rows


# ---------------------------------------------------------------- P3: quy tắc dừng một và hai giai đoạn
def p3_job(fn):
    rng = np.random.default_rng(1414)
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
                    rows.append(dict(base, rule="1 giai đoạn", status="mẫu thử < 5 nhãn giả"))
                    rows.append(dict(base, rule="2 giai đoạn", status="mẫu thử < 5 nhãn giả")); continue
                n1 = int(np.ceil(max(M, n_needed(*e, eps, N))))
                if n1 > 0.8 * N:
                    rows.append(dict(base, rule="1 giai đoạn", status="không khả thi", n_plan=n1))
                else:
                    rows.append(dict(base, rule="1 giai đoạn", status="ok", n_plan=n1, sd=achieved(n1)))
                # hai giai đoạn: thu một nửa kế hoạch (gồm mẫu thử), ước lượng lại, thu nốt
                nh = int(min(max(M, np.ceil(n1 / 2)), 0.8 * N))
                rest = rng.choice(np.setdiff1d(np.arange(N), pilot), nh - M, replace=False)
                e2 = est(np.concatenate([pilot, rest])) or e
                n2 = int(np.ceil(max(nh, n_needed(*e2, eps, N))))
                if n2 > 0.8 * N:
                    rows.append(dict(base, rule="2 giai đoạn", status="không khả thi", n_plan=n2))
                else:
                    rows.append(dict(base, rule="2 giai đoạn", status="ok", n_plan=n2, sd=achieved(n2)))
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


METHODS = ["SRC", "MED", "UPA", "GMM-2", "Otsu", "CATT(A1)", "CATT(A2)", "CATT(A3)", "ORACLE"]


def p4_eval(Z, idx):
    y, st, yv, sv = Z["y_tgt"][idx], Z["f_tgt"][idx], Z["y_val"], Z["f_val"]
    t_src = youden(yv, sv)
    q1v, mv, q3v = np.percentile(sv, [25, 50, 75]); q1t, mt, q3t = np.percentile(st, [25, 50, 75])
    T = {"SRC": t_src, "MED": mt + (t_src - mv) * (q3t - q1t) / max(q3v - q1v, 1e-9),
         "UPA": np.quantile(st, (sv < t_src).mean()), "GMM-2": gmm_thr(st), "Otsu": otsu_thr(st)}
    for lab, k in (("CATT(A1)", "a1"), ("CATT(A2)", "a2"), ("CATT(A3)", "alt")):
        pl = (Z[f"{k}_tgt"][idx] >= youden(yv, Z[f"{k}_val"])).astype(int)
        t = youden(pl, st) if 0 < pl.mean() < 1 else None
        T[lab] = t if t is not None else t_src
    T["ORACLE"] = youden(y, st)
    return {m: bacc(y, st, T[m]) for m in METHODS}


def p4_job(i):
    Zs = [np.load(os.path.join(SC, f"scores_{i}_{vs}.npz")) for vs in (0, 1, 2)]
    y = Zs[0]["y_tgt"]; N = len(y); allidx = np.arange(N)
    point = pd.DataFrame([p4_eval(Z, allidx) for Z in Zs]).mean().to_dict()
    rng = np.random.default_rng(140 + i); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
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
                n=int(r.size))


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
        Dk = D.dropna(subset=["sd_obs"])
        Dk = Dk[Dk.mean_pseudo_pos >= 5]
        S["P1"] = {"all": ratio_stats(Dk, "sd_pred_1.0"),
                   **{f"anchor {a}": ratio_stats(Dk[Dk.anchor == a], "sd_pred_1.0") for a in ANCH},
                   **{f"bandwidth x{m}": ratio_stats(Dk, f"sd_pred_{m}") for m in (0.5, 2.0)},
                   "skipped": D[D.get("skipped").notna()][["setting", "vseed", "anchor"]].to_dict("records")
                   if "skipped" in D else []}
        y = np.log(Dk.sd_obs); nn = Dk.n / (1 - Dk.n / Dk.N); base = -np.log(nn) / 3
        S["P1"]["r2_log_formula"] = float(1 - np.var(y - np.log(Dk["sd_pred_1.0"])) / np.var(y))
        S["P1"]["r2_log_n_only"] = float(1 - np.var(y - base - np.mean(y - base)) / np.var(y))
        S["P1"]["criterion_met"] = bool(0.8 <= S["P1"]["all"]["median_pred_over_obs"] <= 1.25 and S["P1"]["all"]["spearman"] >= 0.7)
        S["P2"] = ratio_stats(Dk, "sd_plugin")
        S["P2"]["criterion_met"] = bool(0.8 <= S["P2"]["median_pred_over_obs"] <= 1.25 and S["P2"]["spearman"] >= 0.7)
        P = pd.read_csv(os.path.join(OUT, "p3_stopping.csv"))
        S["P3"] = {}
        for (rule, eps), d in P.groupby(["rule", "eps"]):
            ok = d[d.status == "ok"]
            k80 = float(np.percentile(np.maximum(1, (ok.sd / eps) ** 3), 80)) if len(ok) else None
            S["P3"][f"{rule}, eps={eps}"] = dict(n_plans=len(d), feasible=len(ok), median_sd_over_eps=float(np.median(ok.sd / eps)),
                                                 frac_le_eps=float(np.mean(ok.sd <= eps)), frac_le_1_2eps=float(np.mean(ok.sd <= 1.2 * eps)),
                                                 median_n_plan=float(np.median(ok.n_plan)), safety_factor_80pct=k80)
        S["P3"]["criterion_met_two_stage"] = bool(all(v["frac_le_1_2eps"] >= 0.8 for k, v in S["P3"].items()
                                                      if isinstance(v, dict) and k.startswith("2 giai đoạn")))
        pt = pd.read_csv(os.path.join(OUT, "p4_point.csv"), index_col=0)
        bt = pd.read_csv(os.path.join(OUT, "p4_boot.csv"))
        broken = [s for s in pt.index if pt.loc[s, "SRC"] < pt.loc[s, "ORACLE"] - 0.05]
        S["P4"] = dict(point=pt.round(4).to_dict("index"), broken_settings=broken,
                       negative_control_settings=[NAME[i] for i in NEG_CONTROL])
        diffs = {}
        for c in ("CATT(A1)", "CATT(A2)"):
            for m in ("SRC", "MED", "UPA", "GMM-2", "Otsu", "CATT(A3)"):
                for lab, sets in (("hỏng", broken), ("tất cả", list(pt.index))):
                    if not sets:
                        continue
                    v = np.mean([bt[bt.setting == s][c].to_numpy() - bt[bt.setting == s][m].to_numpy() for s in sets], 0)
                    p0 = np.mean([pt.loc[s, c] - pt.loc[s, m] for s in sets])
                    diffs[f"{c} − {m} ({lab})"] = [round(float(p0), 4), round(float(np.percentile(v, 2.5)), 4),
                                                    round(float(np.percentile(v, 97.5)), 4)]
        S["P4"]["pooled_differences_point_ci95"] = diffs
        json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1, ensure_ascii=False)
        print(json.dumps(S, indent=1, ensure_ascii=False))
