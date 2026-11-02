# Bước 12: thử công thức của bước 11 trên dữ liệu khác (03/11/2026)

**Câu hỏi:** hai công thức mới của bước 11 có đúng ngoài dữ liệu Brugada không?
- **Công thức n_eff:** CATT với n ECG không nhãn tương đương ngưỡng Youden với n_eff = n·π(1−π)(1−α−β)²/[q(1−q)] mẫu có nhãn.
- **Công thức độ lệch Γ:** độ lệch ngưỡng ≈ Γ/(D′ − Γ′), Γ đo lỗi của mỏ neo dồn ở ranh giới quyết định.

| Script | Việc làm | Đầu ra |
|---|---|---|
| `heart_test.py` | Bộ dữ liệu thứ hai, khác hẳn: UCI Heart Disease, 3 bệnh viện | `heart_raw.csv`, `heart_points.csv`, `heart_neff.csv`, `heart_summary.json` |
| `brugada_neff.py` | Kiểm trực tiếp luật n_eff trên 3 cohort Brugada | `brugada_neff.csv`, `brugada_neff_summary.json` |
| `sim_matched.py` | Cùng quy trình đo, trong mô phỏng biết trước đáp án | `sim_matched.json` |
| `make_fig_v12.py` | Hình 2 panel | `fig_v12.png/.pdf` |
| `common12.py` | Hàm dùng chung (Youden, n_eff, Γ) | |

## 1. Dữ liệu thứ hai: UCI Heart Disease

- Dữ liệu công khai, kinh điển, **nhiều bệnh viện**: Cleveland (Mỹ), Viện Tim mạch Hungary (Budapest), VA Long Beach (Mỹ). Kết cục: bệnh mạch vành trên chụp mạch (num > 0). Tệp lấy từ một bản sao UCI trên GitHub; md5 của tệp Switzerland trùng một bản sao độc lập khác.
- Bỏ Switzerland: chỉ 8/123 người âm tính, không ước lượng được ngưỡng Youden.
- Bỏ các dòng thiếu kết quả test gắng sức: Cleveland 303, Hungary 293, VA 144 (từ 200).
- **Thiết kế giống bài Brugada** (9 thiết lập: 6 cặp + 3 bỏ-một-viện; 10 seed validation):
  - Mô hình đích f: gradient boosting trên biến lâm sàng và lúc nghỉ (tuổi, giới, kiểu đau ngực, huyết áp, cholesterol, đường huyết, ECG nghỉ, nhịp tim tối đa).
  - Mỏ neo "góc nhìn khác": **EX** (độ chênh ST khi gắng sức, phép đo cố định, giống ST_V2) và **EX-LR** (hồi quy logistic trên test gắng sức, giống CBM).
  - Mỏ neo "phụ thuộc": **LR-same** và **RF-same** (mô hình khác trên cùng biến của f, giống LogReg9 và CNNalt).
- Lưu ý: dữ liệu này khó hơn và nhỏ hơn: f chỉ đạt AUROC 0.70; ngưỡng không sập nhiều giữa các viện; CATT tăng balanced accuracy từ 0.637 lên 0.646–0.658 (oracle 0.679).

## 2. Luật n_eff: kết quả

**Cách đo (không cần giả định gì về phân phối):** với mỗi thiết lập và mỏ neo, rút n ECG không nhãn từ viện đích, chạy CATT; song song rút m mẫu **có nhãn thật**, chọn ngưỡng Youden; tìm m\* sao cho hai cách cho cùng tổn thất (đánh giá trên toàn viện đích). Công thức đúng nếu m\*/n ≈ n_eff/n.

| Nguồn | n_eff ≤ 20 | 20–40 | 40–80 | > 80 |
|---|---|---|---|---|
| Mô phỏng, mỏ neo độc lập | 1.63 | 1.46 | 1.11 | 1.02 |
| Brugada (ECG, 3 cohort) | 2.12 | 1.87 | 1.41 | 1.17 |
| UCI Heart Disease (3 viện) | 3.76 | 1.65 | 1.29 | — |

(Trung vị của quan sát/dự đoán; 1 = công thức đúng tuyệt đối.)

**Đọc kết quả:**
- Ở cả ba nguồn, tỉ số **tiến về 1 khi n_eff tăng**: công thức đúng khi cỡ mẫu đủ lớn (n_eff > 80: mô phỏng 1.02, Brugada 1.17).
- Khi n_eff nhỏ, công thức **thận trọng**: CATT thực tế tốt hơn dự đoán khoảng 1.5–2 lần. Mô phỏng cho thấy đây là tính chất của mẫu nhỏ (công thức là tiệm cận), không phải lỗi của công thức. Về mặt ứng dụng, thận trọng là chiều an toàn: bệnh viện thu theo công thức sẽ thu dư chứ không thiếu.
- Với bộ gán nhãn phụ thuộc, công thức n_eff không áp dụng: lỗi tương quan làm giảm phương sai nhưng thêm thiên lệch (Mệnh đề 2).

## 3. Công thức độ lệch Γ trên UCI Heart Disease

36 điểm (9 thiết lập × 4 mỏ neo, trung bình 10 seed):
- **Mỏ neo phụ thuộc:** xếp hạng độ lệch đúng (Spearman **0.92**), công thức phóng đại độ lớn khi lệch lớn, giống trên Brugada.
- **Mỏ neo "góc nhìn khác":** độ lệch dự đoán **không** khớp quan sát (Pearson −0.13). Lý do: độ lệch thật nhỏ, viện đích chỉ 144–303 người, f yếu (đường Youden rất phẳng) nên nhiễu lấy mẫu lớn hơn thiên lệch cần đo.
- Toàn bộ 36 điểm: Pearson 0.43, Spearman 0.52 (yếu hơn nhiều so với Brugada: 0.94 và 0.975).

## 4. Kết luận trung thực

| Công thức | Brugada (ECG) | UCI Heart (dữ liệu khác) | Mô phỏng |
|---|---|---|---|
| n_eff (luật cỡ mẫu) | Đúng khi n_eff lớn; thận trọng khi nhỏ | Cùng xu hướng | Khớp đến 2% khi n_eff > 80 |
| Γ (độ lệch do lỗi ở ranh giới) | Khớp tốt (r = 0.94) | Chỉ khớp xếp hạng ở mỏ neo phụ thuộc | Khớp chính xác (Mệnh đề 1) |

- **Luật n_eff là công thức vững nhất:** cùng một hành vi ở mô phỏng, ECG Brugada và một bộ dữ liệu y khoa hoàn toàn khác. Nên đặt nó làm "công thức của bài".
- **Công thức Γ** giải thích tốt khi thiên lệch lớn hơn nhiễu (Brugada, mỏ neo phụ thuộc); trên dữ liệu nhỏ, yếu thì không đo được.
- UCI Heart Disease không phải ECG, nên nó chứng minh tính **tổng quát** của công thức, không thay thế được một bộ ECG thứ hai.

## 5. Bước tiếp theo đề xuất: một bộ ECG nhiều nguồn

Kiểm mạnh nhất là một bài toán ECG khác trên dữ liệu nhiều nguồn công khai, ví dụ block nhánh phải (RBBB) trong bộ PhysioNet/CinC Challenge 2021 (Georgia, Chapman-Shaoxing, Ningbo, PTB-XL, CPSC):
- f: CNN trên tín hiệu ECG; mỏ neo: tiêu chí độ rộng QRS (≥ 120 ms) đo bằng lớp khái niệm.
- Hàng nghìn ECG mỗi nguồn, nên n_eff lớn, đúng vùng công thức được dự đoán chính xác.
- Cần tải dữ liệu từ PhysioNet về máy (môi trường cloud bị chặn PhysioNet).

## 6. Chạy lại

```
python analysis_v12/heart_test.py analysis_v12        # khoảng 10 phút
python analysis_v12/brugada_neff.py analysis_v11 analysis_v12   # cần analysis_v11/scores
python analysis_v12/sim_matched.py analysis_v12
python analysis_v12/make_fig_v12.py analysis_v12
```
