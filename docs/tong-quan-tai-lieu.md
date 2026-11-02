# Tổng quan tài liệu: dấu hiệu ECG chẩn đoán và tiên lượng đột tử trong hội chứng Brugada

*Soạn tháng 10/2026 cho dự án Grugada. Số liệu lấy từ tóm tắt hoặc bản toàn văn công khai; mục nào chỉ dựa trên tóm tắt/bài tổng hợp được ghi rõ. Trước khi trích dẫn vào bài báo, hãy mở bản gốc để kiểm tra lại con số.*

## 1. Bộ dữ liệu đang dùng

**Brugada-HUCA** (Costa Cortez N, Garcia Iglesias D; PhysioNet v1.0.0, công bố 02/2026)
- 363 bản ghi ECG 12 chuyển đạo, 12 giây, Bệnh viện Universitario Central de Asturias (Tây Ban Nha), thu hồi cứu.
- Thu bằng hệ EP Tracer ở 1 kHz (README GitHub); bản phát hành có header **100 Hz, 1200 mẫu** (đã kiểm tra cả 363 file).
- 76 Brugada / 287 bình thường. Metadata có `basal_pattern` (ECG nền có kiểu Brugada bệnh lý: 0/1), `brugada` (0/1, và thực tế có 7 ca mang giá trị 2 chưa được giải thích), `sudden_death` (11 ca).
- Giới hạn do tác giả nêu: một trung tâm; nhãn nhị phân, không phân biệt type 1 tự phát hay do thuốc.
- Công trình dùng bộ này: một tóm tắt CinC 2026 (multi-scale hybrid transformer) báo AUROC 0.87, độ nhạy 0.67, độ đặc hiệu 0.92 cho phân loại Brugada vs bình thường. → Đây là mốc so sánh nếu bạn làm phân loại.

## 2. Chẩn đoán

| Nguồn | Nội dung chính |
|---|---|
| Antzelevitch C et al. *J-Wave syndromes expert consensus conference report.* Heart Rhythm 2016;13:e295–e324 / Europace 2017;19:665–694 (hội nghị Thượng Hải) | Đề xuất **Shanghai score**: type 1 tự phát 3.5 điểm; type 1 do sốt 3; type 2/3 chuyển type 1 khi test thuốc 2; đột biến gen 0.5 … ≥ 3.5 = chắc chắn/có thể Brugada. Type 1 chỉ lộ ra khi test thuốc chẹn kênh Na cần thêm một tiêu chí khác. |
| Zeppenfeld K et al. *2022 ESC Guidelines for ventricular arrhythmias and prevention of SCD.* Eur Heart J 2022 (doi 10.1093/eurheartj/ehac262) | Type 1 do thuốc đơn độc (không có dấu hiệu khác) không đủ chẩn đoán. *(Nội dung khuyến cáo cụ thể: xem bản gốc, chưa kiểm tra toàn văn.)* |
| Chevallier S et al. JACC 2011 | Phân biệt Brugada type 2/3 với block nhánh phải không hoàn toàn: **góc β > 58°** (nhạy 79%, đặc hiệu 83%), góc α > 50°. Mẫu nhỏ (38 BN). |
| Serra G et al. Europace 2014 | **Đáy tam giác** đo ở 5 mm dưới điểm cất cánh ≥ 4 mm (160 ms): nhạy 95.6%, đặc hiệu 85% phân biệt type 2 với vận động viên có r'. |

**Liên hệ với dự án:** độ dốc ST (`V1_Slope`, `V2_Slope`) và biên độ J là phiên bản tự động, thô hơn của các tiêu chí này. Ở 100 Hz, đo góc β và đáy tam giác chỉ có độ phân giải 10 ms, nhưng là hướng mở rộng đặc trưng hợp lý.

## 3. Tiên lượng đột tử / rối loạn nhịp nguy hiểm

### Yếu tố lâm sàng (nền tảng)
| Nghiên cứu | Thiết kế | Kết quả |
|---|---|---|
| **FINGER** – Probst V et al. Circulation 2010;121:635–643 | 1,029 BN, 11 trung tâm, 4 nước châu Âu; theo dõi trung vị ~32 tháng | Tỷ lệ biến cố/năm: 7.7% (sau ngừng tim), 1.9% (ngất), **0.5% (không triệu chứng)**. Dự báo: triệu chứng + type 1 tự phát. Giới, tiền sử gia đình, gây được rối loạn nhịp khi thăm dò điện sinh lý, đột biến SCN5A: **không** dự báo. |
| **PRELUDE** – Priori SG et al. JACC 2012 | 308 BN, 10 trung tâm Ý; theo dõi trung vị 34 tháng; 14 biến cố (4.5%) | Gây được VT/VF **không** dự báo. Dự báo: ngất + type 1 tự phát (HR 4.20), thời gian trơ thất < 200 ms (HR 3.91), **QRS phân mảnh** (HR 4.94). *(HR từ tóm tắt.)* |
| **BRUGADA-RISK** – Honarbakhsh S et al. JACC Clin Electrophysiol 2021 | 1,110 BN, 16 trung tâm, 8 nước; theo dõi trung vị 5.33 năm; 114 biến cố (10.3%, 1.5%/năm) | 4 yếu tố: ngất nghi do loạn nhịp (HR 3.71), type 1 tự phát (HR 3.80), tái cực sớm (HR 3.42), type 1 ở chuyển đạo ngoại vi (HR 2.33). Ở 5 năm: nhạy 71.2%, đặc hiệu 80.2%. Kiểm định chéo theo quốc gia. |

### Dấu hiệu ECG
| Dấu hiệu | Nguồn | Kết quả | Có trong dự án? |
|---|---|---|---|
| **Sóng S ở DI** (thời gian ≥ 40 ms, diện tích ≥ 1 mm²) | Calò L et al. JACC 2016 (347 BN type 1 tự phát, 4 trung tâm Ý) | Dự báo độc lập mạnh nhất (HR ~38 cho ≥ 40 ms). Có S ở DI: 97% BN VF/SCD so với 56% BN không triệu chứng. Giá trị chủ yếu là **âm tính**: không có S lớn → nguy cơ rất thấp. Tác giả yêu cầu cohort độc lập xác nhận. | ✅ `I_S_ms`, `I_S_amp` |
| **Dấu hiệu aVR** (R ≥ 0.3 mV hoặc R/q ≥ 0.75 ở aVR) | Babai Bigi MA et al. Heart Rhythm 2007 | R ở aVR lớn hơn ở BN có biến cố tái phát; phản ánh chậm dẫn truyền thất phải. | ✅ `aVR_R_amp` |
| aVR + S ở DI (chậm dẫn truyền RVOT) | Phân tích gộp, Front Cardiovasc Med 2022 (11 cohort, 2,575 BN) | RR gộp 1.87 cho biến cố lớn; AUC: aVR 0.77, S ở DI 0.69. | ✅ |
| **QRS phân mảnh (fQRS)** | Morita H et al. Circulation 2008;118:1697–1704 (115 BN) | fQRS ở 85% nhóm VF vs 34% không triệu chứng; có mô hình thực nghiệm. Phân tích gộp 8 nghiên cứu (1,637 BN): RR ~3.6 cho VF. | ❌ Cần tần số lấy mẫu cao; 100 Hz không đủ phát hiện các khía nhỏ trong QRS. |
| QRS rộng, block AV độ 1, Tpeak-Tend, tái cực sớm vùng dưới-bên | Nhiều phân tích gộp (tổng hợp trong Front Cardiovasc Med 2022) | Đều liên quan đến biến cố lớn; mức bằng chứng khác nhau. *(Chỉ đọc tóm tắt bài tổng hợp.)* | ✅ QRS · ❌ PR, Tpeak-Tend, tái cực sớm ngoại vi (có thể thêm) |

**Khuyến cáo thực hành hiện hành:** theo một tổng quan 2025, hướng dẫn ESC và AHA/ACC chỉ công nhận ngất (và một phần thăm dò điện sinh lý) là yếu tố dự báo đã được kiểm chứng; các dấu hiệu ECG ở trên vẫn đang được nghiên cứu.

## 4. Học máy / học sâu trên ECG Brugada
- Liao et al. JACC Clin Electrophysiol 2022: CNN phát hiện type 1 trên ECG 12 chuyển đạo và Holter 24 giờ, hiệu năng tương đương bác sĩ tim mạch đọc lại. Gợi ý Holter + học sâu để sàng lọc trước test procainamide.
- Tóm tắt CinC 2026 trên chính Brugada-HUCA (xem mục 1).
- Một số luận văn thạc sĩ (Politecnico di Torino, 2026) về phát hiện type 2/type 3, chưa qua bình duyệt.

## 5. Phương pháp xử lý tín hiệu và thống kê
| Chủ đề | Nguồn | Áp dụng trong dự án |
|---|---|---|
| Lọc thông cao và méo ST | Kligfield P et al. AHA/ACCF/HRS recommendations, Part I, Circulation 2007 | Chặn thấp chuẩn 0.05 Hz; **được nâng lên ≤ 0.67 Hz nếu bộ lọc số tuyến tính, pha bằng 0**. Dự án dùng 0.5 Hz + `filtfilt` (pha bằng 0) → phù hợp. *(Câu chữ theo bản tóm tắt; nên kiểm tra bản gốc.)* |
| Cỡ mẫu cho mô hình dự báo | Riley RD et al. BMJ 2020;368:m441; gói R `pmsampsize` | Cỡ mẫu hiệu dụng ≈ min(số biến cố, số không biến cố). Với **11 ca đột tử**, không đủ cho mô hình đa biến dù chỉ 1–2 biến. Dự án chỉ báo cáo đơn biến, thăm dò. |
| Hiệu chỉnh nhiều phép kiểm định | Benjamini Y, Hochberg Y. JRSS-B 1995 | q-value BH trong `5.ve bieu do va thong ke.py`. |
| Báo cáo | TRIPOD (2015) / TRIPOD+AI (2024) cho mô hình dự báo; STROBE cho nghiên cứu quan sát | Dùng làm checklist khi viết bài. |

## 6. Khoảng trống và hướng đi cho dự án
1. **Chẩn đoán (Brugada vs bình thường)** là câu hỏi có đủ dữ liệu (76 vs 287). Kết quả dự án đã khớp y văn: J/ST ở V1–V2, QRS rộng, S ở DI dài. Bước tiếp: mô hình hồi quy logistic ít biến hoặc cây quyết định với kiểm định chéo, so với mốc AUROC 0.87 của CinC 2026.
2. **Tiên lượng đột tử**: 11 ca chỉ cho phép mô tả. Hướng mạnh nhất là **kiểm tra giả thuyết có sẵn** (S ở DI ≥ 40 ms, aVR ≥ 0.3 mV) thay vì dò tìm đặc trưng mới, vì như vậy không cần hiệu chỉnh cho 14 phép kiểm định.
3. Đặc trưng có thể thêm ở 100 Hz: khoảng PR (block AV độ 1), góc β / đáy tam giác ở V1–V2, J-point ở chuyển đạo dưới-bên (tái cực sớm), Tpeak-Tend (cần cẩn trọng vì điểm cuối T khó xác định).
4. Đối chiếu điểm J tự động với đo tay của bác sĩ trên khoảng 30–50 ca để báo cáo sai số đo.

## Nguồn đã tra cứu
- PhysioNet Brugada-HUCA: https://physionet.org/content/brugada-huca/1.0.0/ · GitHub README của tác giả: https://github.com/NahuelCostaCortez/brugada-huca · Zenodo: https://zenodo.org/records/15685916
- Calò 2016 (toàn văn): https://iris.unito.it/retrieve/e27ce42a-6510-2581-e053-d805fe0acbaa/BrS_JACC%20text%20accepted_4aperto.pdf · ACC Journal Scan: https://www.acc.org/Latest-in-Cardiology/Journal-Scans/2016/03/21/14/44/A-New-Electrocardiographic-Marker-of-Sudden-Death-in-Brugada
- BRUGADA-RISK (toàn văn): https://discovery.ucl.ac.uk/10122863/24/Lambiase_1-s2.0-S2405500X20308197-main.pdf
- PRELUDE: https://www.qxmd.com/r/22192666
- FINGER: https://hal.archives-ouvertes.fr/hal-00910144
- Morita 2008 fQRS: https://researchhub.com/paper/7743993/fragmented-qrs-as-a-marker-of-conduction-abnormality-and-a-predictor-of-prognosis-of-brugada-syndrome · Phân tích gộp fQRS: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5600929/
- Phân tích gộp aVR / S ở DI: https://pmc.ncbi.nlm.nih.gov/articles/PMC9247269
- J-wave consensus (Thượng Hải): https://pmc.ncbi.nlm.nih.gov/articles/PMC5834028 · https://pubmed.ncbi.nlm.nih.gov/27423412/
- ESC 2022 tóm tắt: https://www.acc.org/latest-in-cardiology/ten-points-to-remember/2022/09/02/14/23/2022-esc-guidelines-for-vas-esc-2022 · Tổng quan điểm nguy cơ 2025: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12902935/
- Chevallier 2011 / Serra 2014: https://pmc.ncbi.nlm.nih.gov/articles/PMC4040869 · https://pmc.ncbi.nlm.nih.gov/articles/PMC6931854
- Học sâu: CinC 2026 https://cinc.org/2026/Program/accepted/545.html · Liao 2022 https://pesquisa.bvsalud.org/nicaragua/resource/es/mdl-35981788
- Lọc ECG (AHA 2007): https://vivo.weill.cornell.edu/display/pubid17317378
- Riley 2020 / pmsampsize: https://cran.r-universe.dev/pmsampsize/doc/manual.html
