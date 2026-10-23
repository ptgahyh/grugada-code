# Bước 10: kiểm tra độ vững cho các kết quả mới (23/10/2026)

Các kết quả thêm vào trong tuần 17–21/10 (1–2 chuyển đạo, cửa sổ đầu ST, smartwatch mô phỏng, chỉ đổi máy, mạng Melo cài lại) ban đầu chỉ chạy với **một** seed chia validation. Bước này chạy lại chúng đúng giao thức của bài:
- 9 thiết lập ngoài miền;
- 3 seed validation;
- ensemble 3 seed;
- quyết định thiết kế chỉ dựa trên 2 thiết lập phát triển;
- kiểm định bootstrap theo bệnh nhân, có hiệu chỉnh Holm.

| Script | Câu hỏi | Đầu ra |
|---|---|---|
| `window_select.py` | Cửa sổ "đầu ST" có còn tốt khi chọn **chỉ** trên 2 thiết lập phát triển không? | `window_dev_summary.csv`, `window_choice.json`, `window_heldout_summary.csv` |
| `leads_3seed.py` | 1–2 chuyển đạo, 3 seed | `leads_3seed_summary.csv`, `leads_3seed_raw.csv` |
| `watch_3seed.py` | Smartwatch áp ngực mô phỏng, 3 seed | `watch_3seed_summary.csv` |
| `device_3seed.py` | Chỉ đổi máy, 3 lần lặp | `device_3seed_summary.csv` |
| `melo_sweep.py` | Mạng Melo cài lại có bị yếu do siêu tham số không? | `melo_sweep_stage1.csv`, `melo_sweep_pick.json`, `melo_sweep_summary.csv` |
| `screening.py` | Góc nhìn sàng lọc lâm sàng (LR, PPV/NPV) | `screening_summary.csv` |
| `bootstrap_tests.py` | 8 so sánh mới, bootstrap theo bệnh nhân + Holm | `bootstrap_tests.csv` |

Điểm số từng bệnh nhân lưu trong `*_scores.npz` (bị `.gitignore` chặn, tạo lại bằng script).
Kiểm tra tái lập: CNN-V1V3 huấn luyện lại trên thiết lập 0–1 trùng điểm của bài, chênh tối đa 5.4e-7. Ở thiết lập 2–8, `leads_3seed.py` dùng lại đúng điểm đã lưu của bài.

## 1. Cửa sổ "đầu ST": kết luận tuần trước KHÔNG đứng vững

Tuần trước, cửa sổ QRS+50..+130 ms được chọn **sau khi** đã xem cả 9 thiết lập, nên có thiên lệch chọn lựa. Tuần này làm lại đúng quy trình:
- 15 cửa sổ neo theo điểm J mềm của lớp khái niệm (dịch −20..+20 ms, dài 60/90/120 ms).
- Quy tắc chọn được cố định trước khi chạy: balanced accuracy với ngưỡng nguồn (SRC) cao nhất trên 2 thiết lập phát triển.

**Trên 2 thiết lập phát triển:** cả 15 cửa sổ đều có SRC cao hơn cả nhịp (0.628–0.724 so với 0.581). Cửa sổ được chọn là **J−20 ms, dài 60 ms**.

**Trên 7 thiết lập held-out** (3 seed):

| Cửa sổ | AUROC | SRC TB / tệ nhất | CATT | Oracle |
|---|---|---|---|---|
| Cả nhịp | 0.876 | 0.701 / 0.548 | **0.799** | 0.820 |
| J−20 ms, 60 ms (chọn trên dev) | 0.849 | 0.704 / 0.596 | 0.764 | 0.789 |
| J−20 ms, 90 ms (lân cận) | 0.859 | 0.719 / 0.625 | 0.776 | 0.796 |
| J−10 ms, 60 ms (lân cận) | 0.854 | 0.709 / 0.614 | 0.776 | 0.796 |
| QRS+50 ms, 90 ms (tuần trước, chọn sau khi xem kết quả) | 0.864 | 0.748 / 0.710 | 0.779 | 0.802 |

Kết luận:
- Cửa sổ chọn đúng quy trình **không cải thiện ngưỡng trung bình** (0.701 → 0.704); trường hợp tệ nhất chỉ tăng nhẹ (0.548 → 0.596).
- Cửa sổ của tuần trước trông tốt hơn chỉ vì được chọn sau khi xem kết quả. **Bỏ luận điểm "90 ms đầu ST làm ngưỡng bền".**
- Điều còn giữ được: chỉ 60–90 ms quanh điểm J đã giữ AUROC 0.85–0.86. Thông tin Brugada tập trung quanh điểm J và đầu ST, khớp với lớp khái niệm.
- Thu hẹp đầu vào không thay được CATT.

## 2. 1–2 chuyển đạo (3 seed)

| Đầu vào | AUROC 9 thiết lập (TB / tệ nhất) | SRC held-out | CATT (CBM) | CATT (nhãn ST_V2) | Oracle |
|---|---|---|---|---|---|
| V1 | 0.860 / 0.811 | 0.739 | 0.794 | 0.795 | 0.815 |
| V2 | 0.846 / 0.757 | 0.733 | 0.795 | 0.787 | 0.807 |
| V3 | 0.727 / 0.596 | 0.662 | 0.682 | 0.681 | 0.699 |
| V1+V2 | 0.843 / 0.783 | 0.722 | 0.786 | 0.783 | 0.800 |
| V1–V3 | 0.865 / 0.819 | 0.701 | 0.799 | 0.789 | 0.820 |
| I | 0.682 / 0.583 | 0.632 | 0.652 | 0.649 | 0.671 |
| II | 0.799 / 0.742 | 0.689 | 0.743 | 0.747 | 0.757 |
| 9 chuyển đạo | 0.812 / 0.741 | 0.650 | 0.734 | 0.736 | 0.758 |

- Một chuyển đạo V1 hoặc V2 gần bằng V1–V3 và hơn hẳn 9 chuyển đạo, khớp với kết quả một seed.
- Hệ thống chỉ dùng V2 (CNN-V2 cộng nhãn giả ST_V2): **0.733 → 0.787** (oracle 0.807).

## 3. Smartwatch áp ngực mô phỏng (3 nguồn Melo × 3 seed, đích HUCA)

| Đầu vào ở HUCA | AUROC | SRC | CATT | Oracle |
|---|---|---|---|---|
| V2 thật | 0.867 | 0.656 | 0.767 | 0.797 |
| Đồng hồ tại V2 (tham chiếu tay phải) | 0.859 | 0.693 | 0.772 | 0.790 |
| Đồng hồ tại V2 (tham chiếu tay trái) | 0.859 | 0.639 | 0.764 | 0.793 |
| V1 thật | 0.877 | 0.682 | 0.765 | 0.824 |
| Đồng hồ tại V1 (tay phải) | 0.860 | 0.752 | 0.751 | 0.811 |
| Đồng hồ tại V1 (tay trái) | 0.795 | 0.648 | 0.706 | 0.760 |
| Đeo cổ tay (chuyển đạo I) | 0.748 | 0.656 | 0.690 | 0.714 |

## 4. Chỉ đổi máy (3 lần lặp)

Dịch lớp âm (đơn vị khoảng tách lớp nguồn), trung bình ± SD:

| Biến thể | A. Trong HUCA | B. Melo → HUCA |
|---|---|---|
| Gốc | −0.00 ± 0.06 | **0.94 ± 0.14** |
| Lọc thông thấp 25 Hz | 0.14 ± 0.03 | 0.98 |
| Lượng tử + nhiễu | −0.02 ± 0.08 | 0.93 |
| Lọc thông cao nhân quả 0.67 Hz | 0.23 ± 0.05 | 1.24 |
| Lọc thông cao nhân quả 1 Hz | 0.33 ± 0.09 | 1.35 |
| Kết hợp | **0.35 ± 0.09** | 1.27 |

Máy giải thích tối đa khoảng 1/3 độ lệch liên viện, chủ yếu qua lọc thông cao nhân quả làm méo đoạn ST.

## 5. Góc nhìn sàng lọc (CNN-V1V3, 7 thiết lập held-out)

| Cách đặt ngưỡng | Độ nhạy | Độ đặc hiệu | LR+ | LR− | PPV ở 1% | NPV ở 1% |
|---|---|---|---|---|---|---|
| SRC | 0.845 | 0.558 (tệ nhất 0.098) | 1.91 | 0.28 | 1.9% | 99.7% |
| **CATT** | 0.776 | 0.823 | **4.38** | 0.27 | **4.2%** | 99.7% |
| Oracle | 0.805 | 0.835 | 4.87 | 0.23 | 4.7% | 99.8% |
| Độ nhạy tại độ đặc hiệu 0.90 / 0.95 (cần nhãn đích) | 0.692 / 0.540 | | | | | |

- CATT tăng LR+ hơn gấp đôi, tức giảm một nửa báo động giả.
- Ở tỉ lệ bệnh cộng đồng, PPV vẫn thấp. Nên định vị bài là công cụ loại trừ hoặc sàng lọc nhóm nguy cơ cao (người nhà bệnh nhân), không phải sàng lọc toàn dân.
- Ngưỡng ở đây được tính trên cohort ở tỉ lệ bệnh tự nhiên; PPV/NPV chỉ tính lại theo công thức Bayes.

## 6. Mạng Melo cài lại: có bị yếu do siêu tham số?

MELO_PLACEHOLDER

## 7. Kiểm định bootstrap theo bệnh nhân (2000 lần, Holm)

BOOT_PLACEHOLDER

## Chạy lại

Mất khoảng 2 giờ trên 2 CPU, chạy hai luồng song song.

```
python analysis_v10/leads_3seed.py harmonized concepts_v5 analysis_v10
python analysis_v10/window_select.py harmonized concepts_v5 analysis_v10
python analysis_v10/watch_3seed.py <thư mục HUCA> harmonized analysis_v10
python analysis_v10/device_3seed.py <thư mục HUCA> harmonized analysis_v10
python analysis_v10/melo_sweep.py harmonized concepts_v5 analysis_v10
python analysis_v10/screening.py harmonized concepts_v5 analysis_v10
python analysis_v10/bootstrap_tests.py harmonized concepts_v5 analysis_v10 2000
```
