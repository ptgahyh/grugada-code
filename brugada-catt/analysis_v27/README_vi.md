# Bước 27: kết quả chính trên Brugada với bộ lọc chung cho mọi nguồn

Kế hoạch `PLAN_vi.md` được commit và đẩy lên lúc 10:51 UTC ngày 14/11/2026 (a86f461), trước mọi phép chạy. Không có sai lệch so
với kế hoạch. Số liệu: `summary_v27.json`; bảng tổng hợp của bước 8 chạy lại nằm trong `results/`.

Thay đổi duy nhất: nhịp Melo (Claris, Mortara) qua lọc thông thấp Butterworth bậc 2, 40 Hz, hai chiều ở 200 Hz trước khi hạ mẫu,
tức cùng thành phần thông thấp mà HUCA đã qua. Huấn luyện lại toàn bộ (bước 5: CNN, CBM, Hybrid; bước 8: CNN, AdaBN, TENT, DANN,
CORAL), 9 thiết lập × 3 seed. Điểm CNN giữa hai lần huấn luyện khớp (sai khác lớn nhất 5·10⁻⁷).

## Chạy lại

```
python analysis_v27/harmonize_v27.py harmonized <melo_processed> <h27>
python concepts_v5/compare_v5.py <h27> <v5_27> 0 2        # và chunk 1; ~17 phút trên 2 nhân
python analysis_v8/v8_core.py <h27> <v5_27> <v8_27> 0 2  # và chunk 1; ~15 phút
python analysis_v8/v8_analyze.py <v8_27> analysis_v27/results
```

## Kết quả theo tiêu chí

| Mã | Nội dung | Pipeline cũ | Lọc chung |
|---|---|---|---|
| **U1** | CATT − SRC, gộp 7 thiết lập held-out (KTC 95%) | +0,098 (0,084–0,109) | **+0,097 (0,085–0,111): đạt** |
| U2 | Thiết lập hỏng | 0, 1, 2, 4, 6 | **0, 1, 2, 4, 6** (giữ nguyên) |
| U4 | AUROC CNN-V1V3, 9 thiết lập | 0,865 | 0,868 |
| U5 | BA held-out: SRC / CATT / ORACLE | 0,701 / 0,799 / 0,820 | 0,703 / 0,800 / 0,823 |

**U3: CATT so với các phương pháp không nhãn khác (gộp held-out):**

| So với | Tỉ lệ bệnh tự nhiên, cũ | Tỉ lệ bệnh tự nhiên, lọc chung | Tỉ lệ bệnh 5%, lọc chung |
|---|---|---|---|
| MED | +0,026 (0,013–0,037) | +0,012 (0,003–0,022) | +0,044 (0,025–0,060) |
| UPA | +0,013 (0,003–0,022) | +0,015 (0,004–0,025) | +0,049 (0,030–0,068) |
| GMM-2 | +0,015 (0,002–0,027) | **+0,006 (−0,004 – 0,015)** | +0,018 (0,004–0,032) |
| Otsu | +0,016 (0,002–0,027) | **+0,004 (−0,004 – 0,015)** | +0,021 (0,006–0,035) |

## Đọc kết quả

1. **Kết luận chính không phụ thuộc vào bất đối xứng tiền xử lý.** Với cùng bộ lọc thông thấp, ngưỡng nguồn vẫn hỏng ở đúng 5 thiết
   lập, và CATT vẫn sửa được gần hết (cách oracle 0,02).
2. **Bộ lọc chung chỉ thu hẹp một phần khác biệt phổ.** Tỉ lệ năng lượng ≥ 15 Hz ở V1–V3: Claris 0,204 → 0,186, Mortara 0,230 → 0,209,
   HUCA 0,147. Phần khác biệt còn lại nằm ở bản chất nhịp (trung vị các nhịp đã lọc so với nhịp đại diện của máy), không phải ở bộ lọc.
   Vì vậy bộ lọc 40 Hz gần như không đổi gì, trong khi làm trơn mạnh hơn ở bước 23 (σ = 6 ms) lấp được khoảng 20%.
3. **Điểm phải nói thẳng:** với pipeline lọc chung, ở tỉ lệ bệnh tự nhiên CATT **không còn hơn GMM-2 và Otsu một cách có ý nghĩa**
   (+0,006 và +0,004, KTC chứa 0). Hai phương pháp này chỉ dùng điểm của chính CNN, không cần mỏ neo. Lợi thế của CATT chỉ còn rõ ở
   **tỉ lệ bệnh 5%** (+0,018 và +0,021), vì GMM-2 và Otsu giả định điểm số có hai cụm rõ, điều mất đi khi ca dương hiếm.
4. Hệ quả cho bài: không viết "CATT tốt hơn các phương pháp không nhãn". Viết đúng là: CATT ngang các phương pháp không giám sát ở tỉ
   lệ bệnh cao, hơn chúng ở tỉ lệ bệnh thấp (gần thực tế sàng lọc hơn), và **có lý thuyết về độ dao động, cỡ mẫu và cổng** mà các phương
   pháp kia không có.

## Hạn chế

- Chỉ thay thành phần thông thấp; thông cao 0,5 Hz không áp dụng được cho nhịp đơn.
- Các bước 13–26 trên Brugada vẫn dùng điểm số pipeline cũ. Nếu chọn pipeline lọc chung làm chính, cần chạy lại các phân tích đó
  (điểm số mới đã có ở thư mục riêng của phiên này).
