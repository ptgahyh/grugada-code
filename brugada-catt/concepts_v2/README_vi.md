# Concepts v2: sửa khái niệm và thêm cơ chế bất biến giữa miền

## Thay đổi so với v1

- **`curv` → `curv60`:** khớp bậc hai trên cửa sổ ngắn [Jl, Jl+60 ms]. Trên ECG giả lập, khả năng phân biệt vòm và yên ngựa tăng từ AUROC 0.77 lên 0.88–0.91.
- **Thêm `beta_rel`:** góc giữa sườn lên và sườn xuống quanh điểm cất cánh, đo theo đơn vị QRSpp/100 ms nên bất biến thang. Trên giả lập chỉ đạt AUROC 0.62–0.65.
  - Trên dữ liệu thật, chiều tác động nhất quán ở cả 3 miền: AUROC đơn biến 0.30–0.44.
  - Chiều này ngược với kỳ vọng của góc β lâm sàng. **Không được gọi nó là góc β lâm sàng.**
- **Độ cao tam giác h:** nới khoảng từ [0.2, 0.8] thành [0.05, 0.8], vì ở v1 h chạm biên dưới.
- **Tổng cộng 28 khái niệm.**

## Bốn cơ chế bất biến (`run_v2.py`)

| Biến thể | Ý tưởng | Dùng nhãn miền test? |
|---|---|---|
| CBM2-Ada | Chuẩn hóa khái niệm theo từng miền. Ở miền mới dùng thống kê của chính miền đó, giống AdaBN | Không, chỉ cần một lô bệnh nhân chưa gắn nhãn |
| CBM2-Sign | Chỉ giữ khái niệm có chiều tác động giống nhau trên **mọi miền train** (\|AUROC−0.5\| > 0.03); trọng số bị ràng buộc dấu: w = s·softplus(v) | Không |
| CBM2-SignAda | Kết hợp hai cơ chế trên | Không |
| CBM2-VREx | Phạt phương sai rủi ro giữa các miền train (chỉ áp dụng LODO) | Không |

## Cách đánh giá

- **Chỉ đánh giá ngoài miền:** 6 cặp chéo + 3 LODO.
- **Ổn định theo seed:** mỗi mô hình chạy với 3 seed chia validation. Điểm cuối là trung bình 3 lần chạy.
- **So sánh cặp:** bootstrap 2000 lần, hiệu chỉnh Holm trên 9 thiết lập.
- **Ngưỡng:** chọn bằng Youden trên validation lấy từ dữ liệu train.

## Kết quả

| Mô hình | AUROC TB (9) | AUROC TB LODO (3) | Balanced acc TB | Thắng/thua/hòa với CNN (Holm) |
|---|---|---|---|---|
| ST_V2 | 0.808 | 0.808 | 0.729 | — |
| CNN | 0.816 | 0.818 | 0.642 | — |
| CBM1 (v1, 1 seed) | 0.821 | 0.837 | — | 2 / 1 / 6 |
| CBM2 | 0.813 | 0.823 | 0.685 | 2 / 1 / 6 |
| CBM2-Ada | **0.827** | **0.848** | 0.720 | 2 / 1 / 6 |
| CBM2-Sign | 0.825 | 0.830 | 0.730 | 2 / 1 / 6 |
| **CBM2-SignAda** | 0.826 | 0.832 | **0.733** | **3 / 1 / 5** |
| CBM2-VREx | — | 0.836 | 0.711 (LODO) | 1 / 0 / 2 |

Độ lệch AUROC giữa các seed nhỏ, trung bình 0.005–0.012.

## Đọc kết quả

1. **Khái niệm mới tự nó không giúp** (CBM2 0.813 so với CBM1 0.821). Phần cải thiện đến từ các **cơ chế bất biến**.
2. **Vấn đề ngưỡng khi đổi miền gần như được giải quyết:** balanced accuracy của CBM2-SignAda là 0.733, so với 0.642 của CNN.
   - Ví dụ Mortara → HUCA, CNN: độ nhạy 0.995 nhưng độ đặc hiệu chỉ 0.150.
   - Cùng cặp đó, SignAda: độ nhạy 0.818, độ đặc hiệu 0.699.
3. **Cặp Mortara → HUCA**, điểm yếu nhất của v1, thu hẹp từ −0.119 xuống −0.037 so với CNN và không còn có ý nghĩa thống kê. AUROC tăng 0.765 → 0.841.
4. **CBM2-SignAda thắng CNN có ý nghĩa ở 3 thiết lập** (HUCA → Claris, Mortara → Claris, HUCA + Mortara → Claris).
   - **Thua có ý nghĩa ở 1 thiết lập:** HUCA + Claris → Mortara, chênh −0.054.
   - HUCA → Mortara thua −0.056 nhưng không còn ý nghĩa sau hiệu chỉnh Holm.
   - **Mortara vẫn là miền khó cho mô hình khái niệm.**
5. **Cổng dấu tự động loại các khái niệm không ổn định** (curv60, tri_w, ST80, V3) đúng ở những thiết lập có HUCA trong tập train. Chiều được giữ lại khớp sinh lý: ST đi xuống, T âm, J chênh lên, QRS rộng. Xem `v2_sign_gate.json`.

## Hạn chế cần ghi rõ

- **Ada cần một lô bệnh nhân của miền mới** để tính thống kê. Tỉ lệ Brugada khác nhau giữa các miền (HUCA 20%, Mortara 37%, Claris 52%) làm lệch các thống kê này. Bài báo nên thêm thí nghiệm thay đổi tỉ lệ bệnh ở miền test.
- **Chỉ có 3 miền**, nên LODO chỉ có 3 điểm dữ liệu.
- **Validation của HUCA rất nhỏ** (khoảng 11 ca dương), nên ngưỡng dao động giữa các seed.

## File

| File | Nội dung |
|---|---|
| `concepts2.py` | Lớp khái niệm v2 |
| `phantom_test2.py`, `phantom_report.json` | Kiểm chứng giả lập |
| `run_v2.py` | Chạy thí nghiệm |
| `analyze_v2.py` | Phân tích kết quả |
| `v2_raw.json` | Điểm dự đoán từng bệnh nhân, từng seed |
| `v2_results.csv`, `v2_paired_holm.csv`, `v2_summary.csv`, `v2_sign_gate.json` | Bảng kết quả |
| `fig_v2_ood.png` | Hình tổng hợp |

Chạy lại:
```
python phantom_test2.py .
python run_v2.py <harmonized> .        # khoảng 40 phút
python analyze_v2.py . <concepts_v1> <harmonized>
```
