# Phân tích ECG hội chứng Brugada (Brugada-HUCA)

Trích xuất đặc trưng ECG 12 chuyển đạo và đánh giá mối liên quan với đột tử trên bộ dữ liệu
[Brugada-HUCA](https://physionet.org/content/brugada-huca/1.0.0/) (PhysioNet): 363 bệnh nhân,
100 Hz, 12 giây, trong đó có 11 ca đột tử.

## Cài đặt và chạy

```bash
pip install -r requirements.txt

python "0.tai du lieu.py"                 # tải dữ liệu (~8 MB) vào data/
python "1.doc du lieu.py"                 # thống kê nhãn, danh sách ca đột tử
python "4.trich xuat dac trung.py"        # trích đặc trưng -> Final_Research_Data.csv
python "5.ve bieu do va thong ke.py"      # thống kê -> ket_qua_thong_ke.csv, figures/
python "3.trung binh các nhip tim.py" --all   # (tùy chọn) hình kiểm tra từng ca -> qc_plots/

python -m pytest tests/                   # kiểm thử bằng ECG giả lập
```

Dữ liệu được đọc theo thứ tự: biến môi trường `BRUGADA_DIR`, rồi thư mục `data/`, rồi đường dẫn ổ `D:\` cũ.

| File | Nội dung |
|---|---|
| `brugada_utils.py` | Xử lý tín hiệu và trích đặc trưng (dùng chung) |
| `brugada_stats.py` | AUC + CI bootstrap, hồi quy logistic Firth, FDR |
| `2.thuat toan.py` | Mô tả thuật toán; chạy thử 1 bệnh nhân: `python "2.thuat toan.py" 428031` |
| `3.trung binh các nhip tim.py` | Vẽ nhịp trung bình + các mốc đo để kiểm tra bằng mắt |
| `tests/` | ECG giả lập với hình dạng biết trước và các bài kiểm thử |

## Phương pháp xử lý tín hiệu

1. Lọc Butterworth bậc 3, 0.5–40 Hz, cả 12 chuyển đạo.
2. Tìm đỉnh R trên DII (QRS ở V1 thường âm, dễ bắt nhầm sóng T).
3. Cắt nhịp (R −400 → +600 ms); loại nhịp lạc có tương quan QRS < 0.9 với nhịp mẫu; lấy **nhịp trung vị**.
4. **Nội suy lên 1000 Hz** (giới hạn băng tần, hợp lệ vì tín hiệu đã lọc < 40 Hz) để đo với độ phân giải 1 ms.
5. **Điểm bắt đầu QRS và điểm J**: tốc độ biến thiên không gian (tổng |đạo hàm| 12 chuyển đạo),
   ngưỡng 8% đỉnh, lấy vùng liền mạch quanh đỉnh. Ngưỡng chọn để trung vị QRS ở người khỏe ≈ 88 ms.
6. Mọi biên độ đo so với mức đẳng điện ngay trước QRS.
7. **Đỉnh/kết thúc T**: đỉnh = cực trị sau J+80 ms; kết thúc = tiếp tuyến tại điểm dốc nhất của nhánh xuống
   cắt mức nền sau T. Sóng T < 0.1 mV → không đo (NaN).

## Đặc trưng

| Cột | Định nghĩa | Tham khảo |
|---|---|---|
| `V1_J`, `V2_J` | Biên độ tại điểm J (mV) | |
| `V1_Slope`, `V2_Slope` | Độ dốc ST từ J đến J+80 ms (mV/s) | |
| `V1_T_Amp`, `V2_T_Amp` | Giá trị thấp nhất vùng ST-T, J+80 → R+450 ms (mV) | |
| `Type1_Like` | Dạng type 1 (coved) ở V1 hoặc V2: ST tại J+20 ms ≥ 0.2 mV, ST đi xuống, T âm ≤ −0.1 mV, không có sóng T dương xen giữa | Priori 2013 (HRS/EHRA/APHRS) |
| `QRS` | Độ rộng QRS toàn bộ 12 chuyển đạo (ms) | Ohkubo 2011 |
| `aVR_R`, `aVR_Sign` | Sóng R cuối ở aVR (mV); dấu aVR khi ≥ 0.3 mV | Babai Bigi 2007 |
| `S_I_Amp`, `S_I_Dur`, `S_I_Sign` | Sóng S ở DI; dương khi ≥ 0.1 mV và/hoặc ≥ 40 ms | Calò 2016 |
| `fQRS_Count`, `fQRS` | Số đỉnh+đáy trong QRS (V1–V3); phân mảnh khi ≥ 4 | Morita 2008 |
| `TpTe` | Tpeak–Tend ở V5 (ms) | Castro Hevia 2006 |
| `QTc` | QT (chuyển đạo có T rõ hơn giữa DII và V5), hiệu chỉnh Bazett (ms) | |
| `ER` | Tái cực sớm: J ≥ 0.1 mV ở ≥ 2 chuyển đạo dưới hoặc ≥ 2 chuyển đạo bên | Kamakura 2009 |
| `HR`, `RR_CV` | Nhịp tim; hệ số biến thiên RR | |
| `n_beats`, `n_used` | Số nhịp cắt được / số nhịp dùng sau khi loại nhịp lạc | |
| `quality_ok` | ≥ 6 nhịp dùng được, ≥ 70% số nhịp được giữ, `RR_CV` ≤ 0.15 | |

Ghi chú: các ngưỡng của `Type1_Like` đã được chỉnh khi đối chiếu với nhãn `brugada` (không dùng nhãn đột tử),
để đạt độ đặc hiệu cao ở người khỏe (1/287 dương tính).

## Thống kê (`5.ve bieu do va thong ke.py`)

1. **Kiểm chứng phép đo**: Brugada so với người khỏe — các đặc trưng phải khác biệt rõ nếu đo đúng.
2. **Phân tích chính**: đột tử so với sống, toàn bộ bệnh nhân.
3. **Phân nhóm**: chỉ bệnh nhân Brugada (`brugada` = 1, 2).
4. **Độ nhạy**: chỉ ca có `quality_ok = 1`.
5. **Hiệu chỉnh**: Firth 2 biến (đặc trưng + chẩn đoán Brugada) cho đặc trưng có ý nghĩa ở (2).

Liên tục: trung vị [IQR], Mann-Whitney, AUC (CI 95% bootstrap phân tầng, 2000 lần), OR Firth mỗi 1 SD.
Nhị phân: n/N (%), Fisher, OR Firth. Giá trị q: FDR Benjamini-Hochberg. Hồi quy Firth (Firth 1993) dùng vì
chỉ có 11 biến cố; không nên đưa quá 1–2 biến vào một mô hình.

## Kết quả hiện tại

- **Kiểm chứng**: hầu hết đặc trưng khác biệt rõ giữa Brugada và người khỏe; `Type1_Like` dương ở 30% Brugada
  so với 0.3% người khỏe; QRS 100 so với 88 ms.
- **Đột tử so với sống** (11 so với 352), có ý nghĩa sau FDR:
  - **Tpeak–Tend V5** 70 so với 61 ms, AUC 0.78 [0.65–0.88], q = 0.04; vẫn có ý nghĩa sau hiệu chỉnh Brugada
    (OR 1.98 mỗi SD [1.18–3.33]).
  - **QRS** 105 so với 90 ms, AUC 0.77 [0.66–0.87], q = 0.04; yếu đi sau hiệu chỉnh Brugada
    (OR 1.45 [0.97–2.15]) — một phần do bệnh nhân Brugada có QRS rộng hơn.
  - Cả hai giữ nguyên khi chỉ phân tích ca chất lượng tốt.
- Trong nhóm Brugada (5 đột tử), không đặc trưng nào có ý nghĩa — cỡ mẫu quá nhỏ.

## Hạn chế

- Chỉ 11 ca đột tử: kết quả mang tính thăm dò, cần kiểm chứng trên dữ liệu độc lập.
- Tần số lấy mẫu 100 Hz và lọc 40 Hz làm mất chi tiết tần số cao: `fQRS` kém tin cậy.
- `ER` là định nghĩa giản lược (chỉ dựa biên độ J, không xét khấc/trượt cuối QRS).
- Đo tự động: nên rà soát hình trong `qc_plots/` cho các ca quan trọng (đặc biệt 11 ca đột tử).

## Tài liệu tham khảo

- Priori SG et al. HRS/EHRA/APHRS expert consensus statement on inherited primary arrhythmia syndromes. Heart Rhythm, 2013.
- Babai Bigi MA et al. aVR sign as a risk factor for life-threatening arrhythmic events in patients with Brugada syndrome. Heart Rhythm, 2007.
- Calò L et al. A new electrocardiographic marker of sudden death in Brugada syndrome: the S-wave in lead I. J Am Coll Cardiol, 2016.
- Morita H et al. Fragmented QRS as a marker of conduction abnormality and a predictor of prognosis of Brugada syndrome. Circulation, 2008.
- Castro Hevia J et al. Tpeak-Tend and Tpeak-Tend dispersion as risk factors for ventricular tachycardia/ventricular fibrillation in patients with the Brugada syndrome. J Am Coll Cardiol, 2006.
- Ohkubo K et al. Prolonged QRS duration in lead V2 and risk of life-threatening ventricular arrhythmia in patients with Brugada syndrome. Int Heart J, 2011.
- Kamakura S et al. Long-term prognosis of probands with Brugada-pattern ST-elevation in leads V1–V3. Circ Arrhythm Electrophysiol, 2009.
- Firth D. Bias reduction of maximum likelihood estimates. Biometrika, 1993.
