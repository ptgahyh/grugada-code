# Bước 18: kế hoạch (viết và commit TRƯỚC khi chạy)

Ngày viết: 11/11/2026. Chưa có mô phỏng hay phép tính nào của bước này được chạy trước khi commit tệp này.
Ghi chú: thư mục `analysis_v17` (bước 17) hiện chỉ có ở một bản cục bộ chưa đẩy lên; bước này đặt ở `analysis_v18`.

Mục đích: trả lời hai nhận xét phương pháp về Hệ quả 2 của bài.

- **18a.** Hệ số quần thể hữu hạn (1 − n/N)^(1/3) trong công thức (8), (9) là heuristic. Kiểm xem số mũ 1/3 có đúng không.
- **18b.** Ước lượng nửa mẫu F1 (b = n/2) không có chứng minh. So nó với các ước lượng **có** chứng minh cho ước lượng
  căn bậc ba, trong mô phỏng biết SD thật.

Cả hai phần chỉ dùng mô hình mô phỏng Gauss của bài (mục 2.7, `analysis_v13/unified_sim.py`):
π = 0,2; f | y=0 ~ N(0, 1); f | y=1 ~ N(1,6; 1,6²); mỏ neo g | y ~ N(0 hoặc 2,5; 1), tương quan trong lớp ρ với f;
nhãn giả ỹ = 1[g ≥ 1,25 + độ dời]. Ngưỡng t̂ = argmax Youden theo nhãn giả (quy tắc f ≥ t), tính trên các điểm mẫu.
Bốn cấu hình: ρ ∈ {0; 0,6} × độ dời ∈ {0; −1,25}. Hạt giống cố định: 1801 (18a), 1802 (18b), 1803 (Brugada).

## 18a. Hệ số quần thể hữu hạn

**Thiết kế.** Với mỗi cấu hình và mỗi N ∈ {400, 1200, 4000}: sinh 200 kho, mỗi kho N bản ghi độc lập từ quần thể.
Trong mỗi kho, với mỗi tỉ lệ n/N ∈ {0,1; 0,2; 0,35; 0,5; 0,8}, lấy 200 mẫu con cỡ n **không hoàn lại** và tính t̂.

- SD_kho(n, N) = căn của trung bình (qua 200 kho) phương sai của t̂ trong từng kho. Đây đúng là "SD quan sát" của bài.
- SD_iid(n) = SD của t̂ qua 2000 mẫu độc lập cỡ n lấy thẳng từ quần thể (tương ứng N = ∞).
- R = SD_kho / SD_iid. Công thức của bài dự đoán R = (1 − n/N)^(1/3).

Mẫu con mà một lớp nhãn giả rỗng thì bỏ; báo cáo số lần bỏ.

**Tiêu chí (định trước).**

| Mã | Nội dung | Đạt khi |
|---|---|---|
| A1 | Tỉ số R / (1 − n/N)^(1/3) | trung vị qua tất cả ô (4 cấu hình × 3 N × 5 tỉ lệ = 60 ô) ∈ [0,90; 1,10] **và** trung vị theo từng tỉ lệ n/N đều ∈ [0,85; 1,15] |
| A2 | Số mũ γ khớp bình phương tối thiểu của log R theo log(1 − n/N), không hệ số chặn | mô tả: báo cáo γ và KTC 95% (bootstrap theo ô); nói rõ KTC có chứa 1/3 và 1/2 hay không |

Mô tả thêm (không có tiêu chí): R theo N; căn trung bình bình phương của (t̂_n − t̂_N) so với SD_kho.

Nếu A1 không đạt: báo cáo đúng như vậy, đề xuất số mũ thực nghiệm, và **không** sửa lùi các kết quả đã đăng ký trước ở
RBBB/LVH (chỉ thêm một phân tích độ nhạy ghi rõ là "quan sát sau").

## 18b. F1 so với các ước lượng có chứng minh

**Đại lượng cần ước lượng.** SD của t̂_n khi n bản ghi lấy độc lập từ quần thể (N = ∞). SD thật = SD của t̂ qua 2000
mẫu độc lập cho mỗi ô. Ô: 4 cấu hình × n ∈ {100, 300, 1000, 3000} = 16 ô. Mỗi ô 200 mẫu; mỗi ước lượng tính một lần
trên mỗi mẫu, không dùng nhãn thật.

**Các ước lượng.**

| Mã | Ước lượng | Cách tính | Cơ sở |
|---|---|---|---|
| E0 | Plug-in KDE | 0,5134·(4Ṽ/(Ã²n))^(1/3), Ṽ và Ã từ KDE Scott như `common15.pseudo_terms` | Định lý 1 |
| E1 | **F1 nửa mẫu** | B = 100 mẫu con cỡ n/2 không hoàn lại; SD(t̂_b) (hệ số quy đổi bằng 1 khi N = ∞), như `common15.sd_subsample` | heuristic của bài |
| E2 | Mẫu con b = n^(2/3) | B = 200 mẫu con không hoàn lại cỡ b = round(n^(2/3)); SD(t̂_b)·(b/n)^(1/3) | Politis, Romano, Wolf 1999; Delgado và cs. 2001 |
| E3 | Bootstrap định hình lại | B = 200; t* = argmax {(J*_n − J_n)(t) − ½·Ã·(t − t̂)²}, Ã từ KDE như E0; SD(t*) | Cattaneo, Jansson, Nagasawa 2020 |
| E4 | Bootstrap số | B = 200; t* = argmax {J_n(t) + ε·√n·(J*_n − J_n)(t)}, ε = n^(−1/3); SD(t*)·ε^(−2/3)·n^(−1/3) | Hong và Li 2020 |
| E5 | Bootstrap thường | B = 200; SD(t*) | đối chứng (đã biết là không nhất quán) |

J*_n là đường Youden theo nhãn giả tính trên mẫu bootstrap (lấy lại cặp (f, ỹ) có hoàn lại). Lần lấy lại mà một lớp rỗng thì bỏ.
Trích dẫn E3, E4 viết theo trí nhớ, phải kiểm lại với bài gốc trước khi đưa vào bản thảo; nếu chi tiết thủ tục khác bài gốc
thì ghi rõ đây là "biến thể theo ý tưởng của…".

**Thước đo.** Với mỗi mẫu: tỉ số ước lượng / SD thật của ô. Báo cáo trung vị, khoảng tứ phân vị, trung vị |log tỉ số|,
gộp và theo từng n.

**Tiêu chí (định trước).**

| Mã | Nội dung | Đạt khi |
|---|---|---|
| B1 | Từng ước lượng có "dùng được" không | trung vị gộp ∈ [0,80; 1,25] **và** trung vị theo từng n ∈ [0,80; 1,25] (cùng dải như bước 14–15) |
| B2 | F1 có kém các ước lượng có chứng minh không | trung vị \|log tỉ số\| của F1 ≤ (giá trị nhỏ nhất trong E2, E3, E4) + 0,03 |
| B3 | Độ nhạy của E4 theo bước ε | mô tả: lặp lại với ε = n^(−1/4) và n^(−0,4) |

Diễn giải định trước:
- B1(F1) đạt và B2 đạt: giữ F1 làm ước lượng chính, thêm E2–E4 làm đối chiếu có lý thuyết.
- B2 không đạt: đổi ước lượng chính sang ước lượng tốt nhất trong E2–E4 **cho các phân tích sau này**; kết quả đăng ký trước
  ở bước 15 giữ nguyên.
- B1(F1) không đạt: báo cáo đúng như vậy và rút câu "gần như không lệch" trong mục 2.3 của bài.

## 18b-Brugada (mô tả, không có tiêu chí vì không có SD thật)

Dùng điểm số của bước 8 (`analysis_v8/scores/scores_{thiết lập}_{seed}.npz`, không nằm trong kho vì có nhãn): f = logit của
`cnn_tgt`, mỏ neo CBM-sd15 (`cbm5_tgt`, ngưỡng Youden trên `cbm5_val`). 9 thiết lập × 3 seed, dùng **toàn bộ** cohort đích
(n = N), không dùng nhãn đích. Tính E0–E5 và báo cáo tỉ số từng ước lượng so với F1 (trung vị, khoảng). Sai khác so với
bước 15c: dùng điểm số bước 8 thay cho bước 11 (bước 11 không có ở máy chạy), nên chỉ có một mỏ neo.

## Không làm trong bước này

Không huấn luyện lại mô hình, không đụng tới RBBB/LVH, không đổi bất kỳ con số nào đã báo cáo.

## Tệp sẽ tạo

`fpc_sim.py`, `fpc_sim.csv`, `est_sim.py`, `est_sim.csv`, `est_brugada.py`, `est_brugada.csv` (chỉ số tổng hợp theo
thiết lập, không có dữ liệu từng bệnh nhân), `summary_v18.json`, `README_vi.md`.
