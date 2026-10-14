"""Lớp khái niệm Brugada khả vi (differentiable Brugada concept layer).

Đầu vào: nhịp trung vị đã chuẩn hóa, tensor (B, 9, 75), 100 Hz, thứ tự chuyển đạo I, II, III, V1..V6,
đỉnh năng lượng QRS nằm ở mẫu ANCHOR = 28 (xem harmonized/harmonize.py).

Ý tưởng chính
-------------
1. Điểm mốc mềm (soft fiducials), tính bằng phép toán khả vi:
   - nền (isoelectric): trung bình có trọng số Gauss quanh đoạn PR, tâm học được
   - điểm J toàn cục: "lần đầu tiên" năng lượng QRS (tổng |đạo hàm| 9 chuyển đạo) rơi dưới ngưỡng θ,
     dùng soft-first-crossing:  w_t = g_t · Π_{s<t} (1 − g_s),  g_t = σ((θ − ê_t)/τ)
     J = Σ_t t·w_t  (+ khối dư dồn về cuối cửa sổ) + δ_J
   - giá trị tại vị trí lẻ (sub-sample) lấy bằng nội suy tuyến tính → đạo hàm theo vị trí tồn tại
2. Khái niệm (concepts) đều là TỈ SỐ hoặc THỜI GIAN, nên bất biến với mọi biến đổi afin theo từng
   chuyển đạo x → a·x + b (a > 0). Vì vậy dùng được cho dữ liệu đã z-score (Melo) lẫn dữ liệu mV (HUCA),
   và chặn được shortcut biên độ.
   Với mỗi chuyển đạo V1, V2, V3 (Jl = điểm "cất cánh" của chuyển đạo: soft-argmax của x − nền
   trong [J − 10 ms, J + 50 ms]; các khái niệm ST đo từ Jl):
     J_elev   = (x(Jl) − nền) / QRSpp
     ST80     = (x(Jl + Δ) − nền) / QRSpp,  Δ học được trong [50, 110] ms (khởi tạo 80 ms)
     Corrado  = (STJ − ST80) / (|STJ| + |ST80| + ε·QRSpp)      ∈ (−1, 1): >0 là ST đi xuống (dạng vòm)
     slope    = a1 / QRSpp  từ khớp bậc hai trên [Jl, Jl+120 ms]
     curv60   = a2 / QRSpp  từ khớp bậc hai trên [Jl, Jl+60 ms]  (v2: cửa sổ ngắn, bắt dạng vòm sớm)
     beta_rel = góc (độ) giữa sườn lên r' [Jl−30, Jl] và sườn xuống [Jl, Jl+30] ms, trong hệ đơn vị
                (biên độ / QRSpp) × (thời gian / 100 ms) → bất biến thang; góc rộng ~ dạng vòm
     r_prime  = softmax(x − nền) trên 60 ms cuối QRS / QRSpp
     tri_w    = độ rộng (ms) của phức bộ r'–ST tại độ cao tương đối h·đỉnh  (h học được)
     T        = trung bình (x − nền) ở [J+140, J+280] ms / QRSpp
   Toàn cục: QRS_w = độ rộng (ms) của năng lượng QRS chuẩn hóa trên ngưỡng 0.3
   (v3: điểm J và QRS_w chỉ tính từ năng lượng của V1–V3, không dùng chuyển đạo chi và V4–V6)
3. Đầu phân loại tuyến tính trên khái niệm đã chuẩn hóa → hệ số đọc được như y văn lâm sàng.
"""
import math

import torch
import torch.nn as nn

ANCHOR = 28
FS = 100
MS = 1000.0 / FS
LEADS_USED = {"V1": 3, "V2": 4, "V3": 5}
PER_LEAD = ["J_elev", "ST80", "Corrado", "slope", "curv60", "beta_rel", "r_prime", "tri_w", "T"]
CONCEPT_NAMES = [f"{c}_{l}" for l in LEADS_USED for c in PER_LEAD] + ["QRS_w"]


def bounded(raw, lo, hi):
    """tham số không ràng buộc -> khoảng [lo, hi]."""
    return lo + (hi - lo) * torch.sigmoid(raw)


def inv_bounded(v, lo, hi):
    p = (v - lo) / (hi - lo)
    return math.log(p / (1 - p))


def interp(x, pos):
    """x: (B, T); pos: (B, K) vị trí thực -> (B, K) giá trị nội suy tuyến tính, khả vi theo pos."""
    T = x.shape[-1]
    pos = pos.clamp(0, T - 1.0001)
    i0 = pos.floor().long()
    w = pos - i0.to(pos.dtype)
    v0 = torch.gather(x, 1, i0)
    v1 = torch.gather(x, 1, i0 + 1)
    return v0 * (1 - w) + v1 * w


def soft_max(x, tau, dim=-1):
    return tau * torch.logsumexp(x / tau, dim=dim)


def sabs(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)


class BrugadaConcepts(nn.Module):
    def __init__(self, learn=True):
        super().__init__()
        # tham số mốc (khởi tạo theo giá trị lâm sàng, có giới hạn)
        self.p_theta = nn.Parameter(torch.tensor(inv_bounded(0.15, 0.03, 0.5)), requires_grad=learn)    # ngưỡng J
        self.p_dj = nn.Parameter(torch.tensor(0.0), requires_grad=learn)                                 # lệch J, ±20 ms
        self.p_base = nn.Parameter(torch.tensor(inv_bounded(-8.0, -15.0, -5.0)), requires_grad=learn)   # tâm nền (mẫu)
        self.p_st = nn.Parameter(torch.tensor(inv_bounded(8.0, 5.0, 11.0)), requires_grad=learn)         # vị trí ST80 (mẫu sau J)
        self.p_h = nn.Parameter(torch.tensor(inv_bounded(0.5, 0.05, 0.8)), requires_grad=learn)          # độ cao tam giác tương đối (v2: nới biên dưới)
        self.tau_g = 0.05
        # ma trận giả nghịch đảo cho khớp bậc hai trên 13 điểm J..J+120 ms
        u = torch.linspace(0, 1, 13)
        V = torch.stack([torch.ones_like(u), u, u * u], 1)
        self.register_buffer("pinv", torch.linalg.pinv(V))        # (3, 13)
        self.register_buffer("k13", torch.arange(13.0))
        u7 = torch.linspace(0, 1, 7)
        V7 = torch.stack([torch.ones_like(u7), u7, u7 * u7], 1)
        self.register_buffer("pinv7", torch.linalg.pinv(V7))      # (3, 7) cho [Jl, Jl+60 ms]
        self.register_buffer("k7", torch.arange(7.0))
        self.register_buffer("k_up", torch.arange(-3.0, 0.0 + 1e-6, 0.5))
        self.register_buffer("k_dn", torch.arange(0.0, 3.0 + 1e-6, 0.5))
        self.register_buffer("k_tri", torch.arange(-6.0, 12.0 + 1e-6, 0.25))   # J-60 ms .. J+120 ms, bước 2.5 ms
        self.register_buffer("k_rp", torch.arange(-6.0, 0.0 + 1e-6, 0.5))
        self.register_buffer("k_T", torch.arange(12.0, 24.0 + 1e-6, 1.0))      # 120..240 ms sau điểm cất cánh
        self.register_buffer("k_to", torch.arange(-1.0, 5.0 + 1e-6, 0.25))     # cửa sổ tìm điểm cất cánh

    # ------------------------------------------------------------ mốc
    def fiducials(self, x):
        B, L, T = x.shape
        xv = x[:, 3:6, :]                                             # v3: chỉ V1, V2, V3
        d = xv[:, :, 1:] - xv[:, :, :-1]
        s = xv.std(-1, keepdim=True) + 1e-6                           # thang của từng chuyển đạo
        e = sabs(d / s).sum(1)                                        # (B, T-1) năng lượng QRS, bất biến thang
        qrs_peak = soft_max(e[:, ANCHOR - 6:ANCHOR + 7], 0.05)        # ~ max quanh đỉnh
        en = e / (qrs_peak.unsqueeze(1) + 1e-6)
        theta = bounded(self.p_theta, 0.03, 0.5)
        lo, hi = ANCHOR + 1, ANCHOR + 15
        g = torch.sigmoid((theta - en[:, lo:hi]) / self.tau_g)      # "đã xuống dưới ngưỡng"
        surv = torch.cumprod(torch.cat([torch.ones(B, 1, dtype=x.dtype), 1 - g[:, :-1]], 1), 1)
        w = g * surv
        resid = (1 - w.sum(1)).clamp(min=0)
        t_idx = torch.arange(lo, hi, dtype=x.dtype) + 1.0           # đạo hàm giữa t và t+1 -> J tại t+1
        J = (w * t_idx).sum(1) + resid * t_idx[-1]
        J = J + 2.0 * torch.tanh(self.p_dj)                          # ±2 mẫu = ±20 ms
        # độ rộng QRS mềm (ms)
        win = en[:, ANCHOR - 8:ANCHOR + 15]
        qrs_w = torch.sigmoid((win - 0.3) / 0.05).sum(1) * MS
        return J, qrs_w

    def base(self, x):
        c = ANCHOR + bounded(self.p_base, -15.0, -5.0)
        t = torch.arange(x.shape[-1], dtype=x.dtype)
        w = torch.exp(-0.5 * ((t - c) / 1.5) ** 2)
        w = w / w.sum()
        return (x * w).sum(-1)                                        # (B, L)

    # ------------------------------------------------------------ khái niệm
    def forward(self, x, return_aux=False):
        B = x.shape[0]
        J, qrs_w = self.fiducials(x)
        base = self.base(x)
        st_off = bounded(self.p_st, 5.0, 11.0)
        h = bounded(self.p_h, 0.05, 0.8)
        feats = []
        for name, li in LEADS_USED.items():
            xl = x[:, li, :]
            b = base[:, li:li + 1]
            sc = xl.std(-1, keepdim=True) + 1e-6                     # nhiệt độ softmax theo thang chuyển đạo
            tau = 0.02 * sc
            seg = xl[:, ANCHOR - 6:ANCHOR + 9]
            pp = (soft_max(seg / tau, 1.0) + soft_max(-seg / tau, 1.0)) * tau[:, 0] + 0.02 * sc[:, 0]   # đỉnh-đỉnh QRS (mềm)
            # điểm "cất cánh" (high take-off) riêng từng chuyển đạo: soft-argmax của (x − nền)
            # trong [J − 10 ms, J + 50 ms]; STJ là giá trị tại đó
            P = J[:, None] + self.k_to[None]
            yto = interp(xl, P) - b
            wto = torch.softmax(yto / tau, dim=1)
            xJ = (wto * yto).sum(1)
            Jl = (wto * P).sum(1)
            x80 = interp(xl, (Jl + st_off)[:, None])[:, 0] - b[:, 0]
            J_elev = xJ / pp
            ST80 = x80 / pp
            corr = (xJ - x80) / (sabs(xJ, 1e-3 * pp) + sabs(x80, 1e-3 * pp) + 0.05 * pp)
            ys = interp(xl, Jl[:, None] + self.k13[None]) - b            # (B, 13)
            coef = ys @ self.pinv.T                                     # (B, 3): a0, a1, a2
            slope = coef[:, 1] / pp
            y7 = interp(xl, Jl[:, None] + self.k7[None]) - b
            curv60 = (y7 @ self.pinv7.T)[:, 2] / pp
            # góc beta tương đối: độ dốc theo đơn vị QRSpp / 100 ms (10 mẫu)
            def lsq_slope(k):
                yy = interp(xl, Jl[:, None] + k[None]) - b
                kc = k - k.mean()
                return (yy * kc).sum(1) / (kc * kc).sum() * 10.0 / pp
            s_up = lsq_slope(self.k_up)
            s_dn = lsq_slope(self.k_dn)
            beta_rel = (math.pi - torch.atan(s_up) - torch.atan(-s_dn)) * 180.0 / math.pi / 100.0
            yr = interp(xl, J[:, None] + self.k_rp[None]) - b
            r_prime = soft_max(yr / tau, 1.0) * tau[:, 0] / pp
            yt = interp(xl, Jl[:, None] + self.k_tri[None]) - b
            peak = soft_max(yt / tau, 1.0) * tau[:, 0]
            lvl = h * peak
            tri_w = torch.sigmoid((yt - lvl[:, None]) / (0.03 * pp[:, None])).sum(1) * (0.25 * MS) / 100.0
            yT = interp(xl, Jl[:, None] + self.k_T[None]) - b
            Tm = yT.mean(1) / pp
            feats += [J_elev, ST80, corr, slope, curv60, beta_rel, r_prime, tri_w, Tm]
        feats.append(qrs_w / 100.0)
        C = torch.stack(feats, 1)
        if return_aux:
            return C, dict(J=J, qrs_w=qrs_w, base=base)
        return C

    def fiducial_values(self):
        return dict(theta=float(bounded(self.p_theta, 0.03, 0.5).detach()),
                    dJ_ms=float(2.0 * torch.tanh(self.p_dj) * MS),
                    base_ms=float(bounded(self.p_base, -15.0, -5.0) * MS),
                    st_offset_ms=float(bounded(self.p_st, 5.0, 11.0) * MS),
                    tri_height=float(bounded(self.p_h, 0.05, 0.8)))


class ConceptBottleneck(nn.Module):
    """Khái niệm -> chuẩn hóa (BatchNorm không affine) -> tuyến tính."""

    def __init__(self, learn_fiducials=True):
        super().__init__()
        self.concepts = BrugadaConcepts(learn=learn_fiducials)
        self.norm = nn.BatchNorm1d(len(CONCEPT_NAMES), affine=False, momentum=None)  # trung bình tích lũy
        self.head = nn.Linear(len(CONCEPT_NAMES), 1)

    def forward(self, x):
        return self.head(self.norm(self.concepts(x))).squeeze(1)
