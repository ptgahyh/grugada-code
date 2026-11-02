# Mệnh đề 5 viết lại sau phản biện: định lý, xấp xỉ, hệ quả

Mệnh đề 5 được tách thành ba phần, ghi rõ phần nào đã chứng minh và phần nào chỉ có bằng chứng số.

## Ký hiệu

- **Dữ liệu ở nguồn đích:** n mẫu độc lập (f_i, ỹ_i).
  - f là điểm số của mô hình đích (logit của CNN).
  - ỹ = 1[g ≥ c] là nhãn giả của mỏ neo.
  - Ngưỡng c của mỏ neo lấy từ tập validation ở nguồn gốc, nên coi là cố định khi xét dữ liệu đích.
- **Đặc trưng của lớp nhãn giả:** q = P(ỹ = 1); p̃₁, p̃₀ là mật độ của f trong lớp nhãn giả dương và âm; F̃₁, F̃₀ là hàm phân phối tương ứng.
- **Đường Youden theo nhãn giả:**
  - Quần thể: J̃(t) = F̃₀(t) − F̃₁(t). Đây là TPR − FPR theo nhãn giả, cộng thêm một hằng số.
  - Thực nghiệm: J̃ₙ(t), tính trên n mẫu.
- **Hai ngưỡng:**
  - Ngưỡng CATT quần thể: t̃ = argmax J̃.
  - Ngưỡng CATT ước lượng: t̂ = argmax J̃ₙ.
- **Hai đại lượng chính:**
  - Ṽ = p̃₁(t̃)/q + p̃₀(t̃)/(1−q).
  - Ã = |J̃″(t̃)| = |p̃₀′(t̃) − p̃₁′(t̃)|.
- **Phân phối Chernoff:** Z = argmax_h {W(h) − h²}, với W là chuyển động Brown hai phía. Var Z = 0.26355, SD Z = 0.5134 (Groeneboom & Wellner 2001).

## Định lý 5a (phân phối của ngưỡng CATT): đã chứng minh

**Điều kiện:**
- (C1) J̃ có cực đại duy nhất, tách biệt, tại t̃ nằm trong miền xác định.
- (C2) p̃₁ và p̃₀ khả vi liên tục trong một lân cận của t̃.
- (C3) Ã > 0.
- (C4) 0 < q < 1.

**Kết luận:**

  n^{1/3} (t̂ − t̃) → (4Ṽ / Ã²)^{1/3} · Z   (hội tụ theo phân phối)

Nói cách khác, khi n đủ lớn, t̂ ≈ t̃ + a·Z với a = (4Ṽ/(Ã²n))^{1/3}.

**Chứng minh (khung):**
1. **Quá trình cục bộ.** Đặt M_n(h) = n^{2/3}[J̃ₙ(t̃ + h n^{−1/3}) − J̃ₙ(t̃)].
   - Phần kỳ vọng: −(Ã/2)h² + o(1), do khai triển Taylor bậc hai quanh cực đại (C1–C3).
   - Phần ngẫu nhiên: tổng của [ỹ/q − (1−ỹ)/(1−q)]·1[f nằm giữa t̃ và t̃ + δ]. Phương sai của nó xấp xỉ (δ/n)·[p̃₁/q + p̃₀/(1−q)] = δṼ/n.
   - Kết hợp hai phần: M_n hội tụ yếu về √Ṽ·W(h) − (Ã/2)h².
2. **Ảnh hưởng của q ước lượng.** Thay q bằng q̂ = n₁/n chỉ thêm sai số cỡ O(n^{−1/2}·n^{−1/3}). Ở thang n^{−2/3}, sai số này không đáng kể.
3. **Định lý argmax.** Kim & Pollard (1990), Định lý 1.1, áp dụng được vì:
   - lớp hàm chỉ thị nửa đường thẳng là lớp VC;
   - bao hàm thoả điều kiện cube-root.
   Đổi biến h = (4Ṽ/Ã²)^{1/3}·s, giới hạn trở thành tỉ lệ với argmax_s {W(s) − s²} = Z.
4. **So với y văn.** Hằng số trùng dạng của Banerjee & McKeague (2007) cho split point, và của Kosorok (2008) §14.5 cho ước lượng argmax kiểu "Manski".

**Trường hợp riêng: mỏ neo độc lập có điều kiện.** Khi đó t̃ = t\*, p̃₁(t\*) = p̃₀(t\*) = φ, và Ã = k|J″| với k = π₀π₁(1−α−β)/[q(1−q)] (Menon 2015). Thay vào:
- Ṽ = φ/[q(1−q)];
- a³ = 4φ/(J″²·π₀π₁·n_eff).

Đây đúng là luật n_eff (Mệnh đề 3), nên luật n_eff là một hệ quả của Định lý 5a.

## Xấp xỉ 5b (tổn thất kỳ vọng): heuristic, có bằng chứng số

  E[BA(t\*) − BA(t̂)] ≈ E_Z[ BA(t\*) − BA(t̃ + a·Z) ]

**Phần chứng minh được:**
- Khi t̃ = t\*, khai triển bậc hai cho ¼|J″|·a²·Var Z. Đây là công thức cũ, đúng tiệm cận nếu thêm điều kiện khả tích đều.
- Khi t̃ ≠ t\*, giới hạn bậc nhất là thiên lệch BA(t\*) − BA(t̃).

**Phần chưa chứng minh:**
- Dùng đường BA thật thay cho parabol.
- Gộp thiên lệch và nhiễu ở bậc hai khi t̃ ≠ t\*. Ở đây E[t̂ − t̃] chỉ là o(n^{−1/3}); ta không có tốc độ chính xác hơn.

Vì vậy 5b là **xấp xỉ**: tốt hơn công thức cũ ở mẫu nhỏ là phát hiện thực nghiệm (mô phỏng Gauss, mô phỏng ngoài Gauss, Brugada).

## Hệ quả 5c (độ bất định không cần nhãn và quy tắc dừng): điểm mạnh thực hành

t̃, Ṽ và Ã chỉ phụ thuộc vào phân phối chung của (f, ỹ). Vì vậy **nguồn mới ước lượng được độ bất định của ngưỡng CATT trước khi có nhãn nào**:

  SD(t̂) ≈ 0.513·(4Ṽ/(Ã²n))^{1/3}

**Quy tắc dừng:** muốn SD(t̂) ≤ ε (trên thang logit), cần

  n ≥ 4·(0.513)³·Ṽ/(Ã²ε³) ≈ 0.54·Ṽ/(Ã²ε³)

Lưu ý:
- Luật n_eff không làm được việc này, vì nó cần α, β, tức là cần nhãn thật.
- Điểm yếu: Ã là độ cong, ước lượng bằng KDE hội tụ chậm và phụ thuộc băng thông. Kết quả kiểm độ nhạy nằm trong README.

## Hiệu chỉnh quần thể hữu hạn: heuristic

Khi lấy n mẫu không hoàn lại từ một cohort N người, phương sai của quá trình thực nghiệm nhân với (1 − n/N) (Hájek). Vì a tỉ lệ với căn bậc ba của phương sai, ta thay n bằng n/(1 − n/N). Bước này chưa được chứng minh cho ước lượng argmax.

## Tính mới (diễn đạt trung thực)

Định lý 5a là **áp dụng trực tiếp** Kim–Pollard và Banerjee–McKeague cho tiêu chí Youden theo nhãn giả. Đóng góp nằm ở ba chỗ:
1. Nhận ra Ṽ và Ã là các đại lượng của phân phối nhãn giả, nên ước lượng được không cần nhãn.
2. Chỉ ra luật n_eff và bất biến Menon là trường hợp riêng.
3. Biến kết quả thành quy tắc dừng cho nguồn mới.

Cách diễn đạt phù hợp là "mở rộng và kết hợp", không phải "định lý mới hoàn toàn".
