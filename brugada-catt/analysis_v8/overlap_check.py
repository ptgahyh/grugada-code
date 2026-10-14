"""(Bước 8) Có bệnh nhân nào xuất hiện ở cả Claris và Mortara (cùng trung tâm, hai máy) không?

Nếu có, các thiết lập Claris ↔ Mortara sẽ lạc quan (rò rỉ bệnh nhân giữa nguồn và đích).
Cách kiểm tra (không cần định danh): với mỗi nhịp đại diện, tìm láng giềng gần nhất (tương quan Pearson trên 9 chuyển
đạo × 75 mẫu) ở cohort kia, rồi so phân bố đó với phân bố láng giềng gần nhất TRONG cùng cohort (bỏ chính nó).
Một bệnh nhân đo trên hai máy sẽ cho cặp chéo có tương quan cao bất thường, vượt hẳn mức "láng giềng tự nhiên".
Để có mốc chuẩn cho "cùng bệnh nhân, khác bản ghi": các cặp trùng gần như hoàn toàn trong Claris (đã biết, r > 0.999)
và nhịp HUCA so với chính nó khi tính lại theo kiểu máy đo (bước 7, r trung vị 0.966).
Chạy: python overlap_check.py <harmonized> <sensitivity_v7> <out>
"""
import json
import os
import sys

import numpy as np

HD, S7, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
D = {d: np.load(os.path.join(HD, f"{d}.npz")) for d in ["huca", "claris", "mortara"]}


def unit(x):
    v = x.reshape(len(x), -1).astype(np.float64)
    v = v - v.mean(1, keepdims=True)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


U = {d: unit(D[d]["X"]) for d in D}
Yd = {d: D[d]["y"].astype(int) for d in D}


def nn_cross(a, b):
    C = U[a] @ U[b].T
    j = C.argmax(1)
    return C.max(1), j


def nn_within(a):
    C = U[a] @ U[a].T
    np.fill_diagonal(C, -1)
    return C.max(1), C.argmax(1)


res = {}
q = [5, 25, 50, 75, 95, 99, 100]
for a in ["claris", "mortara", "huca"]:
    r, _ = nn_within(a)
    res[f"within_{a}"] = dict(zip(map(str, q), np.percentile(r, q).round(4).tolist()))
r_mc, j_mc = nn_cross("mortara", "claris")
r_cm, j_cm = nn_cross("claris", "mortara")
r_hm, _ = nn_cross("huca", "claris")
res["mortara_to_claris"] = dict(zip(map(str, q), np.percentile(r_mc, q).round(4).tolist()))
res["claris_to_mortara"] = dict(zip(map(str, q), np.percentile(r_cm, q).round(4).tolist()))
res["huca_to_claris"] = dict(zip(map(str, q), np.percentile(r_hm, q).round(4).tolist()))
# mutual nearest neighbours across devices with very high correlation
mutual = [(i, j) for i, j in enumerate(j_mc) if j_cm[j] == i]
res["n_mutual_nn_pairs"] = len(mutual)
for thr in (0.95, 0.97, 0.98, 0.99):
    hi = [(i, j) for i, j in mutual if r_mc[i] >= thr]
    agree = np.mean([Yd["mortara"][i] == Yd["claris"][j] for i, j in hi]) if hi else float("nan")
    res[f"mutual_pairs_r>={thr}"] = dict(n=len(hi), label_agreement=round(float(agree), 3) if hi else None)
# label agreement expected by chance
pm, pc = Yd["mortara"].mean(), Yd["claris"].mean()
res["chance_label_agreement"] = round(float(pm * pc + (1 - pm) * (1 - pc)), 3)
# reference: same patient, different median-beat computation (HUCA original vs device-style, step 7)
hp = os.path.join(S7, "huca_hires.npz")
if os.path.exists(hp):
    H = unit(np.load(hp)["X"])
    same = (H * U["huca"]).sum(1)
    res["reference_same_patient_recomputed_HUCA"] = dict(zip(map(str, q), np.percentile(same, q).round(4).tolist()))
# how many Mortara beats are closer to a Claris beat than any Claris beat is to its own within-Claris neighbour (p99)?
p99 = np.percentile(nn_within("claris")[0], 99)
res["mortara_with_cross_nn_above_claris_within_p99"] = int((r_mc > p99).sum())
json.dump(res, open(os.path.join(OUT, "overlap_check.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
