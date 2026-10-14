"""ResNet1D và hàm huấn luyện, tách từ concepts_v4/arch_v4.py (giữ nguyên) để dùng lại ở bước 6."""
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


class Res(nn.Module):
    def __init__(self, c_in, c_out, k):
        super().__init__()
        self.a = nn.Sequential(nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.BatchNorm1d(c_out), nn.ReLU(),
                               nn.Conv1d(c_out, c_out, k, padding=k // 2), nn.BatchNorm1d(c_out))
        self.s = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return torch.relu(self.a(x) + self.s(x))


class ResNet1D(nn.Module):
    def __init__(self, n_leads):
        super().__init__()
        self.f = nn.Sequential(Res(n_leads, 32, 7), nn.MaxPool1d(2), Res(32, 64, 5), nn.MaxPool1d(2), Res(64, 64, 3),
                               nn.AdaptiveAvgPool1d(1))
        self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.h(self.f(x)).squeeze(1)


def train_resnet(xtr, ytr, xva, yva, seed, epochs=60, patience=12):
    TG = torch.Generator(); torch.manual_seed(seed); TG.manual_seed(seed); g = np.random.default_rng(seed)
    net = ResNet1D(xtr.shape[1]); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ytr == 0).sum() / max(1, (ytr == 1).sum()), dtype=torch.float32))
    xt, yt, xv = torch.tensor(xtr), torch.tensor(ytr, dtype=torch.float32), torch.tensor(xva)
    best, state, wait = -1, None, 0
    for _ in range(epochs):
        net.train(); perm = g.permutation(len(xt))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            xb = torch.roll(xt[b], int(g.integers(-2, 3)), dims=2)
            xb = xb * (1 + 0.1 * (torch.rand(xb.shape[0], xb.shape[1], 1, generator=TG) - 0.5) * 2) + 0.05 * torch.randn(xb.shape, generator=TG)
            opt.zero_grad(); lossf(net(xb), yt[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            auc = roc_auc_score(yva, torch.sigmoid(net(xv)).numpy())
        if auc > best:
            best, state, wait = auc, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(state); net.eval()
    return net
