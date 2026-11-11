# Bước 19: ngưỡng trôi vì thành phần dạng sóng hay vì cùng dạng sóng mà khác điểm số?

Kế hoạch `PLAN_vi.md` được commit và đẩy lên lúc 04:01 UTC ngày 11/11/2026 (6eceb3b), trước khi gom cụm hay tính kết quả.
Không có sai lệch so với kế hoạch. Mục "Quan sát sau" ở cuối **không** thuộc kế hoạch. Số liệu: `summary_v19.json`.

## Chạy lại

```
python analysis_v19/cluster_v19.py <harmonized> <analysis_v8/scores> <metadata.csv của HUCA> analysis_v19 <thư mục riêng>   # ~2 phút
python analysis_v19/make_fig_v19.py analysis_v19
python analysis_v19/realign_posthoc.py <harmonized> <analysis_v8/scores> analysis_v19                                        # quan sát sau
```

Nhãn cụm từng bệnh nhân nằm ở thư mục riêng, không đẩy lên kho.

## Kết quả theo tiêu chí (K = 5 cụm, lớp âm, 5 thiết lập hỏng)

| Mã | Nội dung | Kết quả | Kết luận định trước |
|---|---|---|---|
| D1 | Tỉ phần "trong cụm" của độ dịch trung bình logit | **0,97** (KTC 95% 0,93–1,03); K = 3: 1,00; K = 8: 0,97 | ≥ 2/3 và cận dưới > 1/2: **cùng dạng sóng, khác điểm số** |
| D2 | Ngưỡng "chỉ sửa thành phần" lấp bao nhiêu khoảng cách tới oracle | **4%** (BA 0,567 → 0,575; oracle 0,796). CATT trên cùng điểm số: 91% (0,774) | ≤ 20%: thành phần **không** giải thích việc hỏng ngưỡng |
| D3 | Thành phần cụm giữa các bộ (TV, lớp âm) | HUCA–Claris 0,67 (0,62–0,73); HUCA–Mortara 0,43 (0,37–0,50); Claris–Mortara 0,24 (0,17–0,32) | **Đạt**: hai viện khác nhau hơn hai máy cùng viện |
| D4 | Lỗi ở ngưỡng nguồn có dồn vào dạng sóng nào không | Độ giàu theo cụm 0,77–1,24; tỉ lệ lỗi 63–100% ở mọi cụm | mô tả: lỗi trải đều mọi dạng sóng |

Độ dịch trung bình logit của lớp âm ở 5 thiết lập hỏng lớn (trung bình |Δ| = 2,3 logit), ở 4 thiết lập không hỏng nhỏ (0,26).

| Thiết lập | HUCA→Claris | HUCA→Mortara | Claris→HUCA | Mortara→HUCA | Claris+Mortara→HUCA |
|---|---|---|---|---|---|
| Δ (logit, lớp âm) | −1,30 | −1,26 | +3,39 | +2,41 | +3,02 |
| phần thành phần | −0,02 | 0,00 | −0,04 | **+0,58** | −0,20 |
| phần trong cụm | −1,28 | −1,26 | +3,43 | +1,83 | +3,22 |

Lớp dương (phụ, validation nguồn nhỏ): tỉ phần trong cụm 1,00 (0,95–1,07) ở K = 5; 0,89 ở K = 3.

## Mô tả cụm (K = 5; đánh số theo tỉ lệ Brugada tăng dần)

| Cụm | Tỉ phần lớp âm: HUCA / Claris / Mortara | Tỉ lệ Brugada: HUCA / Claris / Mortara | HUCA: ECG nền bệnh lý |
|---|---|---|---|
| 1 | 51% / 2% / 8% | 0,15 / 0,22 / 0,30 | 8% |
| 2 | 17% / 57% / 50% | 0,19 / 0,39 / 0,23 | 12% |
| 3 | 20% / 8% / 20% | 0,31 / 0,45 / 0,42 | 19% |
| 4 | 11% / 4% / 11% | 0,24 / 0,62 / 0,48 | 17% |
| 5 | 1% / 28% / 11% | 0,40 / 0,67 / 0,59 | 20% (5 bản ghi) |

Hình `fig_v19_clusters.png`: cụm 3–5 có r′ và đoạn J–ST cao ở V1–V2, tỉ lệ Brugada cao hơn ở cả ba bộ. Cụm 1 và 2 khác nhau chủ
yếu ở **thời điểm đáy QRS** so với điểm căn (xem dưới).

## Đọc kết quả

1. **Ngưỡng hỏng không phải vì hai viện có tỉ lệ dạng sóng khác nhau.** Trong cùng một cụm dạng sóng, điểm số đã lệch
   1,3–3,4 logit. Sửa riêng thành phần chỉ lấp 4% khoảng cách. Lỗi ở ngưỡng nguồn trải đều mọi cụm.
2. **Thứ thay đổi là thang điểm của cả cohort**, không phải một nhóm bệnh nhân. Điều này khớp các bước trước: AUROC giữ, ngưỡng
   trôi; và đây là loại lỗi mà một phép đặt lại ngưỡng (CATT) sửa được.
3. Ngoại lệ một phần: Mortara→HUCA có 24% độ dịch đến từ thành phần (0,58 trên 2,41).
4. **Phân bố dạng sóng giữa hai viện khác nhau rõ** (TV 0,67), giữa hai máy cùng viện thì ít hơn (0,24). Trả lời câu hỏi của
   thầy: ở mức dạng sóng, các bộ dữ liệu **không** giống nhau; dù vậy mô hình vẫn nhìn cùng một vùng (bước 17).

## Quan sát sau (không có trong kế hoạch)

- Ở V1, đáy sóng S nằm tại **0 ms** so với điểm căn ở HUCA nhưng **+20 ms** ở Claris và Mortara (trung vị; V2: +10 so với +20 ms).
  Tức là cùng quy tắc "căn theo đỉnh năng lượng QRS" nhưng điểm căn rơi vào chỗ khác nhau của phức bộ QRS ở hai nguồn.
- Giả thuyết (chưa kiểm): `harmonize.py` tính năng lượng trên tín hiệu **mV** ở HUCA nhưng trên nhịp **đã z-score từng chuyển đạo**
  ở Melo, nên trọng số các chuyển đạo khác nhau. Ngoài ra code chỉ lọc 0,5–40 Hz cho HUCA; nhịp Melo chỉ được lấy mẫu lại.
  Câu "all ECGs were band-pass filtered" trong bài cần sửa cho đúng.
- Gom cụm lại sau khi căn từng bản ghi theo đáy S của V1 (`realign_posthoc.py`): TV lớp âm HUCA–Claris giảm từ 0,67 xuống **0,38**,
  Claris–Mortara từ 0,24 xuống 0,05. Khoảng một nửa khác biệt "dạng sóng" giữa hai viện là do lệch căn thời gian.
  Tỉ phần trong cụm của D1 không đổi (0,97).
- Chưa biết độ lệch 20 ms này đóng góp bao nhiêu vào độ dịch điểm số của CNN (mạng có tăng cường dịch ±20 ms khi huấn luyện).
  Kiểm được bằng cách dịch nhịp HUCA 1–2 mẫu rồi chấm điểm lại; cần mô hình, tức cần PyTorch.

## Hạn chế

- Điểm số phía nguồn chỉ có trên tập validation (HUCA: 54 bản ghi, 11 dương). Ở HUCA→Claris có 9,5% tỉ trọng đích lớp âm rơi vào
  cụm không có bản ghi validation nguồn (lớp dương: tới 34%), khi đó phần đó được tính vào "trong cụm". K = 3 không có cụm vắng và
  cho cùng kết luận.
- Gom cụm k-means trên nhịp đã z-score: cụm là mô tả thống kê, không phải phân loại lâm sàng (type 1, type 2, RBBB…).
- "Trong cụm" không tách được nguyên nhân: máy đo, vị trí điện cực, cách tạo nhịp đại diện, lệch căn thời gian, hay khác biệt
  quần thể mà 5–8 cụm không bắt được.
- Chưa làm đường đáp ứng của f theo tham số dạng sóng (cần PyTorch).
