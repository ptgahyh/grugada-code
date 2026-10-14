"""(Bước 5) Lớp khái niệm v5 = v3 (điểm J từ V1–V3) nhưng thay khái niệm độ rộng QRS.

Vấn đề của QRS_w (bước 3–4): đó là độ rộng tuyệt đối (ms) của năng lượng đạo hàm QRS ở V1–V3. Độ rộng này
phụ thuộc dải thông của máy đo và cách máy tính nhịp trung vị, nên dịch chuyển mạnh giữa các miền ngay
cả ở nhóm âm tính.

Hai ứng viên thay thế (xem qrs_test5.py, README_vi.md mục B):
  "rel"   QRS_rel = log( W(V1–V3) / W(V4–V6) ). Ứng viên CHỌN TRƯỚC theo giả thuyết "chậm dẫn truyền khu trú
          ở thất phải": tỉ số triệt tiêu phần giãn do thiết bị. KẾT QUẢ: bền với thiết bị nhưng mất gần hết
          khả năng phân biệt (AUROC 0.52–0.55), vì ở cả 3 bộ dữ liệu Brugada làm QRS rộng ở MỌI chuyển đạo
          (chậm dẫn truyền lan tỏa). Giữ lại để báo cáo kết quả âm tính.
  "lp15"  QRS_w15 = độ rộng theo ngưỡng (như QRS_w) nhưng đo trên bản sao đã lọc thông thấp 15 Hz.
          Bền với thiết bị nhưng BỊ LOẠI: sau lọc, độ rộng theo ngưỡng bị đáp ứng xung của bộ lọc chi phối
          (~80 ms bất kể QRS thật), phantom giãn QRS cho Spearman −0.38.
  "sd15"  QRS_sd15 = thời lượng RMS (mô-men bậc hai, ms) của năng lượng đạo hàm V1–V3 sau lọc 15 Hz:
              e_t = relu(ê_t − 0.1),  ê = năng lượng chuẩn hóa theo đỉnh trong [−80, +150] ms quanh đỉnh QRS
              QRS_sd15 = sqrt( Σ e_t (t − μ)² / Σ e_t ),  μ = Σ e_t t / Σ e_t
          Lý do: tích chập với bộ lọc CỘNG phương sai của bộ lọc vào phương sai của đường bao, nên với một
          sàn dải thông chung (15 Hz), khác biệt dải thông giữa các máy (> 15 Hz) bị loại, còn chênh lệch thời
          lượng thật được giữ (gần đúng, dạng cộng). ĐƯỢC CHỌN cho bước 5 bằng quy tắc KHÔNG dùng nhãn hay dữ
          liệu miền khác: trong 8 biến thể (có/không lọc × lũy thừa 1/2 × nền 0/0.1), chọn biến thể có phantom
          đơn điệu (Spearman ≥ 0.95) và ít nhạy nhất với nhiễu loạn thiết bị mô phỏng (lọc 20/30 Hz, làm trơn).
  "abs"   QRS_w như v3 (đối chứng).
Mọi chế độ vẫn bất biến afin theo từng chuyển đạo và khả vi.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import firwin

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "concepts_v4"))
import concepts3  # noqa: E402
from concepts3 import ANCHOR, MS, LEADS_USED, PER_LEAD, sabs, soft_max  # noqa: E402

QRS_NAME = {"abs": "QRS_w", "rel": "QRS_rel", "lp15": "QRS_w15", "sd15": "QRS_sd15"}


def concept_names(qrs="sd15"):
    return [f"{c}_{l}" for l in LEADS_USED for c in PER_LEAD] + [QRS_NAME[qrs]]


CONCEPT_NAMES = concept_names("sd15")


def soft_width(x_group, lo=ANCHOR - 8, hi=ANCHOR + 15, level=0.3, temp=0.05):
    """x_group: (B, L, T). Độ rộng mềm (ms) của năng lượng đạo hàm chuẩn hóa theo thang từng chuyển đạo."""
    d = x_group[:, :, 1:] - x_group[:, :, :-1]
    s = x_group.std(-1, keepdim=True) + 1e-6
    e = sabs(d / s).sum(1)
    peak = soft_max(e[:, ANCHOR - 6:ANCHOR + 7], 0.05)
    en = e / (peak.unsqueeze(1) + 1e-6)
    return torch.sigmoid((en[:, lo:hi] - level) / temp).sum(1) * MS


def rms_duration(x_group, lo=ANCHOR - 8, hi=ANCHOR + 15, floor=0.1):
    """Thời lượng RMS (ms) của năng lượng đạo hàm chuẩn hóa (khả vi, bất biến afin từng chuyển đạo)."""
    d = x_group[:, :, 1:] - x_group[:, :, :-1]
    s = x_group.std(-1, keepdim=True) + 1e-6
    e = sabs(d / s).sum(1)[:, lo:hi]
    e = e / (soft_max(e, 0.02).unsqueeze(1) + 1e-6)
    e = F.softplus((e - floor) * 50.0) / 50.0                 # relu mềm
    t = torch.arange(lo, hi, dtype=e.dtype)
    w = e / (e.sum(1, keepdim=True) + 1e-9)
    mu = (w * t).sum(1, keepdim=True)
    return torch.sqrt((w * (t - mu) ** 2).sum(1) + 1e-6) * MS


class BrugadaConcepts(concepts3.BrugadaConcepts):
    def __init__(self, learn=True, qrs="sd15", fc=15.0, ntap=21):
        super().__init__(learn=learn)
        self.qrs = qrs
        self.register_buffer("fir", torch.tensor(firwin(ntap, fc, fs=100.0), dtype=torch.float32).view(1, 1, -1))

    def lowpass(self, x):
        B, L, T = x.shape
        p = self.fir.shape[-1] // 2
        xp = F.pad(x.reshape(B * L, 1, T), (p, p), mode="replicate")
        return F.conv1d(xp, self.fir.to(x.dtype)).reshape(B, L, T)

    def forward(self, x, return_aux=False):
        C, aux = super().forward(x, return_aux=True)
        if self.qrs == "abs":
            return (C, aux) if return_aux else C
        if self.qrs == "rel":
            w13, w46 = soft_width(x[:, 3:6, :]), soft_width(x[:, 6:9, :])
            q = torch.log((w13 + 5.0) / (w46 + 5.0))
        elif self.qrs == "lp15":
            q = soft_width(self.lowpass(x[:, 3:6, :])) / 100.0
        else:
            q = rms_duration(self.lowpass(x[:, 3:6, :])) / 10.0
        C = torch.cat([C[:, :-1], q[:, None]], 1)
        return (C, aux) if return_aux else C
