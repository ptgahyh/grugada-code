# Bước 15: LVH (bài toán ECG thứ ba) và bản sửa công thức không nhãn

Kế hoạch `PLAN_vi.md` được commit lúc 05:40 UTC ngày 03/11/2026 (commit ce6c1ad), trước khi trích nhịp LVH hay huấn luyện
mô hình LVH. Chuẩn hóa chạy 05:40–05:43, huấn luyện 05:43–06:08, phân tích 06:08–06:20 (UTC). Không có sai lệch so với kế hoạch.

## Dữ liệu

| Miền | N | LVH (+) | Bị loại |
|---|---|---|---|
| `ptb_cs` | 12 843 | 941 (7,3%) | 0 do ít nhịp |
| `ptb_at` | 8 352 | 810 (9,7%) | 1 do ít nhịp |
| `cn` | 38 242 | 637 (1,7%) | 1 487 do ít nhịp; 2 header hỏng |

Loại theo định nghĩa nhãn: PTB-XL bỏ 381 bản ghi LVH likelihood < 50 và 222 bản ghi VCLVH-only; cn bỏ 5 399 bản ghi LVHV-only
và 22 bản ghi không rõ giới tính (`results/harmonize_log.json`).

AUROC trên miền đích (khoảng qua 3 seed):

| Thiết lập | f | A4 CNNalt | A1 Sokolow | A2 Cornell | A3 LVH-LR |
|---|---|---|---|---|---|
| cs→at | 0,88 | 0,88 | 0,88 | 0,68 | 0,91 |
| cs→cn | 0,93–0,94 | 0,94 | 0,97 | 0,78 | 0,98 |
| at→cs | 0,91 | 0,90–0,91 | 0,92 | 0,70 | 0,95 |
| at→cn | 0,94 | 0,93 | 0,97 | 0,78 | 0,98 |
| cn→cs | 0,88 | 0,88 | 0,92 | 0,70 | 0,94 |
| cn→at | 0,83–0,84 | 0,84 | 0,88 | 0,68 | 0,90 |

## Kết quả theo tiêu chí định trước

| | Tiêu chí | Kết quả | |
|---|---|---|---|
| **P1** Định lý 5a | trung vị ∈ [0,8; 1,25], Spearman ≥ 0,7 | **1,06** (IQR 1,01–1,16), Spearman **0,98** | **ĐẠT** |
| **P2** Hệ quả 5c (KDE) | như trên | 1,11 (IQR 1,01–1,24), Spearman 0,97 | **ĐẠT** |
| **F1** Hệ quả 5c (mẫu con) | như trên | **1,00** (IQR 0,92–1,08), Spearman 0,97 | **ĐẠT** |
| F1 so với P2 | sai số \|log\| trung vị của F1 < P2 | 0,078 so với 0,132 | **ĐẠT** |
| **P3** dừng 2 giai đoạn (KDE) | ≥ 80% đạt SD ≤ 1,2ε | 89% (ε = 0,15), 90% (ε = 0,30) | **ĐẠT** |
| **F2** dừng 2 giai đoạn (mẫu con) | như trên | 76%, 79% | **KHÔNG ĐẠT** |
| **H1** CATT > SRC ở thiết lập hỏng | cần ≥ 1 thiết lập hỏng | không có thiết lập hỏng | **không kiểm được** |

Chi tiết:
- P1: R² trên thang log 0,97 (mô hình chỉ theo n: 0,11). Theo mỏ neo: Sokolow 1,11; Cornell 1,01; LVH-LR 1,06; CNNalt 1,10.
  Băng thông ×0,5: 0,94; ×2: 1,17. Không cấu hình nào bị bỏ. Số nhãn giả dương trung bình mỗi mẫu con thấp nhất là 25.
- P3: một giai đoạn 83% / 87%; hệ số an toàn cho 80% là 1,4 (một giai đoạn) và 1,1–1,2 (hai giai đoạn).
  Theo mỏ neo, hai giai đoạn ε = 0,15: Sokolow 80%, Cornell 77%, LVH-LR 89%, CNNalt 100%.
- F2: hệ số an toàn 2,0 / 1,8. Kém hơn bản KDE trên LVH, dù tốt hơn bản KDE trên RBBB (bước 15a: 61–72% so với 54–58%).

### P4: không có thiết lập "hỏng" theo định nghĩa balanced accuracy

| | SRC | MED | UPA | GMM-2 | Otsu | CATT(A1) | CATT(A2) | CATT(A3) | CATT(A4) | ORACLE |
|---|---|---|---|---|---|---|---|---|---|---|
| cs→at | 0,801 | 0,797 | 0,800 | 0,758 | 0,753 | 0,773 | 0,771 | 0,790 | **0,801** | 0,803 |
| cs→cn | **0,872** | 0,855 | 0,857 | 0,781 | 0,760 | 0,775 | 0,857 | 0,842 | 0,865 | 0,874 |
| at→cs | 0,823 | **0,824** | 0,823 | 0,756 | 0,744 | 0,781 | 0,792 | 0,810 | 0,817 | 0,828 |
| at→cn | **0,866** | 0,841 | 0,844 | 0,761 | 0,748 | 0,775 | 0,820 | 0,822 | 0,861 | 0,868 |
| cn→cs | 0,781 | 0,596 | 0,756 | 0,787 | 0,790 | **0,801** | 0,794 | 0,800 | 0,797 | 0,803 |
| cn→at | 0,742 | 0,581 | 0,711 | 0,755 | 0,758 | 0,763 | 0,753 | **0,764** | 0,761 | 0,766 |

Khoảng cách SRC − ORACLE lớn nhất là −0,024 (cn→at), nên H1 không kiểm được. Hiệu gộp trên 6 thiết lập
(KTC 95% bootstrap): CATT(A3) − SRC = −0,010 (−0,012; −0,005); CATT(A3) − Otsu = +0,046 (+0,042; +0,050);
CATT(A3) − GMM-2 = +0,039; CATT(A3) − UPA = +0,006 (+0,004; +0,011); CATT(A1) − SRC = −0,036; CATT(A2) − SRC = −0,016.
Mỏ neo phụ thuộc A4 có khoảng cách tới oracle **nhỏ hơn** các mỏ neo sinh lý, tức ngược với dự đoán mô tả của Mệnh đề 1–2.

## Quan sát sau (post hoc, không thuộc kế hoạch)

1. **Ngưỡng có hỏng, nhưng balanced accuracy không thấy.** Ở cn→PTB, ngưỡng nguồn lệch khỏi oracle khoảng 1,6–1,9 logit.
   Độ nhạy của SRC là 0,71 và 0,63, so với 0,81 và 0,71 của oracle. CATT(A1/A3) đưa độ nhạy về 0,80–0,82 và 0,74–0,77.
   Balanced accuracy gần như phẳng quanh đỉnh, nên mất 8–10 điểm độ nhạy chỉ làm BA giảm khoảng 0,02.
   Định nghĩa "hỏng" theo BA (giữ nguyên từ bước 14) vì vậy không bắt được loại hỏng này. Nhận xét này cần kiểm định trước
   ở dữ liệu mới, không được dùng làm bằng chứng xác nhận.
2. **CATT hại khi đích có tỉ lệ bệnh thấp hơn nhiều.** Ở PTB→cn (LVH 1,7%), mỏ neo gắn nhãn giả dương cho 11–21% bản ghi,
   phần lớn là âm tính có điện thế cao. Những bản ghi này cũng có điểm f cao, tức sai số của mỏ neo và của f tương quan,
   vi phạm giả định độc lập có điều kiện. Ngưỡng CATT vì vậy bị kéo xuống quá thấp: độ nhạy 0,95–0,98, độ đặc hiệu 0,57–0,73.
   Đây đúng là chế độ hỏng mà Mệnh đề 1–2 cảnh báo, và nó xuất hiện ở mỏ neo sinh lý chứ không ở CNNalt.

## Tổng hợp ba bộ dữ liệu (tỉ số dự đoán/quan sát của SD(t̂))

| | Brugada (bước 13) | RBBB (bước 14) | LVH (bước 15) |
|---|---|---|---|
| Định lý 5a (Ṽ, Ã trên cả đích) | 1,09 | 0,80 | 1,06 |
| Hệ quả 5c, KDE | 1,08 | 0,78 | 1,11 |
| Hệ quả 5c, mẫu con (F1) | **0,98 (xác nhận, F1-B)** | 0,95 (phát triển) | **1,00 (xác nhận)** |
| Dừng 2 giai đoạn KDE, % đạt SD ≤ 1,2ε | chưa kiểm | 54–58% | 89–90% |
| Dừng 2 giai đoạn mẫu con, % đạt | chưa kiểm | 61–72% | 76–79% |

Kết luận:
- Định lý 5a cho thứ hạng rất tốt ở cả ba bài toán. Hằng số sai từ −20% (RBBB) đến +10%. Cách giải thích "nhóm nhãn giả
  dương quá nhỏ" cho RBBB là chẩn đoán sau khi có kết quả, nên chỉ được nêu như giả thuyết.
- Bản mẫu con F1 là cách ước lượng không nhãn tốt nhất ở cả hai bộ đã chạy, và đạt tiêu chí trên LVH, nơi nó được kiểm
  định xác nhận. Nên đưa F1 vào bài làm ước lượng chính.
- Quy tắc dừng: chưa bản nào đạt 80% ở cả hai bộ. Bản KDE hai giai đoạn đạt trên LVH nhưng không đạt trên RBBB. Nên trình bày
  là quy tắc lập kế hoạch kèm tỉ lệ đạt thực nghiệm, không phải bảo đảm.
- Giả thuyết chính (CATT sửa ngưỡng hỏng theo BA) vẫn chỉ có bằng chứng từ Brugada.

## F1-B: F1 trên Brugada (kiểm ngoài mẫu thứ hai, chạy 03/11/2026)

`brugada_subsample_check.py`, code cố định từ commit kế hoạch. 561 điểm (9 thiết lập × 3 seed × 7 mỏ neo × 3 cỡ mẫu).

| Ước lượng SD(t̂) | Dự đoán/quan sát (trung vị) | Khoảng tứ phân vị | Spearman | Trung vị \|log sai số\| |
|---|---|---|---|---|
| Định lý 5a, Ṽ và Ã trên cả cohort | 1,09 | 0,96–1,26 | 0,92 | 0,134 |
| Hệ quả 5c, KDE, không nhãn | 1,08 | 0,93–1,31 | 0,86 | 0,168 |
| **F1, mẫu con, không nhãn** | **0,98** | 0,89–1,11 | **0,93** | **0,114** |

**Tiêu chí F1 đạt** (trung vị trong [0,8; 1,25], Spearman ≥ 0,7). F1 cũng tốt hơn KDE theo so sánh định trước (0,114 < 0,168).

**Theo mỏ neo (F1):**

| Mỏ neo | CBM | CBM-sd15 | CNN9 | CNNalt | LogReg9 | ResNet | ST_V2 |
|---|---|---|---|---|---|---|---|
| Dự đoán/quan sát | 0,98 | 0,97 | 1,00 | 0,97 | 0,98 | 0,94 | 1,07 |

- **ST_V2:** KDE dự đoán dư 1,17–1,23; F1 giảm còn 1,07, nằm trong dải chung.
- Như vậy phần dư ở ST_V2 nằm ở việc ước lượng hằng số (Ṽ, Ã), không nằm ở tốc độ n^{−1/3} của Định lý 5a.

## Định vị khi viết bài (rà soát sau F1-B)

**F1 không phải phương pháp mới.** Mẫu con m-out-of-n là cách suy luận chuẩn cho ước lượng hội tụ tốc độ căn bậc ba, vì bootstrap
thông thường không nhất quán với loại ước lượng này. Cần trích dẫn:
- Kim & Pollard (1990), *Ann. Statist.* 18:191–219 (tiệm cận căn bậc ba);
- Politis, Romano & Wolf (1999), *Subsampling*, Springer;
- Delgado, Rodríguez-Poo & Wolf (2001), *Economics Letters* 73:241–250 (mẫu con cho ước lượng maximum score);
- Abrevaya & Huang (2005), *Econometrica* 73:1175–1204 (bootstrap thất bại với maximum score).

Lý thuyết mẫu con đòi b/n → 0. Ở đây b = n/2, và bước quy đổi dùng dạng quần thể hữu hạn (1 − b/n)/b của Định lý 5a.
Bước này là **heuristic**, chỉ được bảo đảm bằng thực nghiệm (LVH 1,00; Brugada 0,98; RBBB 0,95 ở dữ liệu phát triển).

**Vai trò còn lại của hằng số 4Ṽ/Ã².** F1 chỉ dùng tốc độ n^{−1/3} và hệ số quần thể hữu hạn, không dùng hằng số.
Nhưng hằng số vẫn cần ở **giai đoạn lập kế hoạch**: khi mới có mẫu thử 300 bản ghi, quá ít để lấy mẫu con.
Trên LVH, bản lập kế hoạch dùng hằng số (KDE, hai giai đoạn) đạt 89–90%, còn bản thay bằng mẫu con chỉ đạt 76–79%.
Cách phân vai đề xuất: hằng số dùng để **lập kế hoạch** cỡ mẫu; F1 dùng để **báo cáo độ chính xác** của ngưỡng đã tính.

**Đóng góp lý thuyết, viết khiêm tốn:** áp dụng Kim–Pollard cho ngưỡng Youden theo nhãn giả; nhận diện hằng số Ṽ/Ã² theo
các đại lượng của bài toán (mật độ hai lớp nhãn giả tại t̃, hệ số quần thể hữu hạn); chỉ ra n_eff là trường hợp riêng.

**Điều kiện về số ca nhãn giả dương** (RBBB dự đoán thiếu khi mỗi mẫu có khoảng 15–25 ca) là giả thuyết sinh ra sau khi có
kết quả, chưa được chứng minh hay kiểm định.

**Không dùng tiêu chí độ nhạy để cứu phần phương pháp trong bài này.** Ngưỡng theo độ nhạy cố định là ước lượng phân vị, hội tụ
tốc độ n^{−1/2}, nên Định lý 5a không áp dụng được. Quan sát về độ nhạy ở cn→PTB (mục "Quan sát sau") chỉ ghi lại làm gợi ý
cho một nghiên cứu riêng, có kế hoạch riêng.
