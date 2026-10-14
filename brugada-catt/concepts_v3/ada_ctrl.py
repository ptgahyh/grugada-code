"""Ada-ctrl: chuẩn hóa khái niệm theo QUẦN THỂ ÂM TÍNH (control) của từng miền.

Lý do: Ada thường (thống kê trên cả lô) bị lệch khi tỉ lệ Brugada trong lô thay đổi.
Nếu chuẩn hóa theo nhóm âm tính thì thống kê không phụ thuộc tỉ lệ bệnh.
  - Khi huấn luyện: nhãn có sẵn -> thống kê của từng miền tính trên ca âm tính của miền đó.
  - Ở miền mới (không có nhãn): ước lượng nhóm âm tính giả (pseudo-control) lặp:
      bước 0: thống kê trên cả lô -> chấm điểm
      bước t: lấy các ca có điểm < ngưỡng validation (tối thiểu 30% lô có điểm thấp nhất) làm âm tính giả
              -> tính lại thống kê -> chấm điểm lại   (3 vòng)
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from cbm_lib import CBM2, loc_scale


class CBM2Ctrl(CBM2):
    def norm_ctrl(self, C, dom, ctrl, freeze=False, fit=False):
        out = torch.empty_like(C)
        for g in np.unique(dom):
            m = torch.tensor(dom == g)
            if freeze and g in self.stats:
                mu, sd = self.stats[g]
            else:
                mu, sd = loc_scale(C[m & torch.tensor(ctrl)], self.robust)
                if fit:
                    self.stats[g] = (mu.detach(), sd.detach())
            out[m] = (C[m] - mu) / sd
        return out

    def forward_ctrl(self, x, dom, ctrl, freeze=False, fit=False):
        return self.norm_ctrl(self.concepts(x), dom, ctrl, freeze, fit) @ self.weights() + self.b


def train_ctrl(xtr, ytr, dtr, xva, yva, dva, seed, signs, epochs=300, patience=40, l1=1e-3):
    torch.manual_seed(seed); g = np.random.default_rng(seed)
    model = CBM2Ctrl(signs, robust=False)
    opt = torch.optim.Adam(model.parameters(), lr=2e-2)
    pw = torch.tensor((ytr == 0).sum() / max(1, (ytr == 1).sum()), dtype=torch.float32)
    xt, yt, xv = torch.tensor(xtr), torch.tensor(ytr, dtype=torch.float32), torch.tensor(xva)
    ctrl = ytr == 0
    best, state, wait = -1, None, 0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        logit = model.forward_ctrl(torch.roll(xt, int(g.integers(-1, 2)), dims=2), dtr, ctrl, fit=True)
        R = torch.stack([F.binary_cross_entropy_with_logits(logit[torch.tensor(dtr == d)], yt[torch.tensor(dtr == d)], pos_weight=pw)
                         for d in np.unique(dtr)])
        (R.mean() + l1 * model.weights().abs().sum()).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            model.forward_ctrl(xt, dtr, ctrl, fit=True)
            auc = roc_auc_score(yva, model.forward_ctrl(xv, dva, None, freeze=True).numpy())
        if auc > best:
            best, state, wait = auc, (copy.deepcopy(model.state_dict()), copy.deepcopy(model.stats)), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state[0]); model.stats = state[1]; model.eval()
    return model


def predict_val(models, x, dom):
    with torch.no_grad():
        return np.mean([torch.sigmoid(m.forward_ctrl(torch.tensor(x), dom, None, freeze=True)).numpy() for m in models], 0)


def predict_batch(models, x, thr, iters=3, min_frac=0.3):
    """miền mới, không nhãn: ước lượng nhóm âm tính giả rồi chuẩn hóa theo nó."""
    xt = torch.tensor(x); dom = np.array(["__b__"] * len(x))
    ctrl = np.ones(len(x), bool)
    with torch.no_grad():
        for _ in range(iters + 1):
            p = np.mean([torch.sigmoid(m.forward_ctrl(xt, dom, ctrl)).numpy() for m in models], 0)
            new = p < thr
            k = int(np.ceil(min_frac * len(x)))
            if new.sum() < k:
                new = np.zeros(len(x), bool); new[np.argsort(p)[:k]] = True
            ctrl = new
    return p
