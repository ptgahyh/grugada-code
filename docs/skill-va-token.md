# Skill đã cài và cách tiết kiệm token

*Cập nhật 31/10/2026.*

## 1. Skill hoạt động thế nào (và tốn token ra sao)
- Khi mở phiên, Claude chỉ nạp **tên + mô tả ngắn** của mỗi skill (vài chục đến ~200 token mỗi skill).
- Nội dung đầy đủ (`SKILL.md`, 3–7 nghìn token với skill K-Dense) chỉ được nạp **khi Claude thật sự dùng skill đó**. File trong `references/` và `scripts/` chỉ được đọc khi cần.
- Vì vậy thêm vài skill chọn lọc gần như không tốn thêm. Nhưng **cài cả kho 140 skill** thì mọi phiên đều mang theo 140 mô tả, khoảng 10 nghìn token trở lên, và Claude dễ chọn nhầm skill.

## 2. Skill trong repo (`.claude/skills/`)
Claude tự nạp các skill này khi bạn mở repo bằng Claude Code.

### Skill riêng của dự án (ưu tiên cao nhất)
| Skill | Việc |
|---|---|
| `ecg-brugada-pipeline` | Quy tắc xử lý ECG trong `ecg_pipeline.py` |
| `small-sample-stats` | Quy tắc thống kê khi chỉ có 11 ca đột tử |
| `lit-review` | Cập nhật `docs/tong-quan-tai-lieu.md`, không bịa trích dẫn |
| `research-writing-vn` | Viết báo cáo/bài báo từ `results/` và `figures/` |

### Skill tổng quát từ K-Dense *Scientific Agent Skills* (giấy phép MIT)
Nguồn: https://github.com/K-Dense-AI/scientific-agent-skills (trước đây là `claude-scientific-skills`), nhánh `main`, tải ngày 31/10/2026.

| Skill | Phiên bản | Dùng cho dự án này |
|---|---|---|
| `neurokit2` | 1.4 | Đối chiếu dò QRS/điểm J với NeuroKit2 (`ecg_delineate`) để kiểm chứng thuật toán của mình |
| `statistical-analysis` | 2.0 | Chọn kiểm định, kiểm tra giả định, cỡ hiệu ứng, báo cáo kiểu APA |
| `scientific-visualization` | 1.4 | Hình đạt chuẩn tạp chí: màu thân thiện người mù màu, khoảng tin cậy, xuất file |
| `scientific-writing` | 2.3 | Viết bản thảo có truy vết bằng chứng, checklist báo cáo (STROBE, TRIPOD…) |
| `literature-review` | 1.11 | Tổng quan hệ thống/tường thuật, PRISMA, kiểm tra DOI |
| `paper-lookup` | 2.4 | Tra 18 nguồn: PubMed, PMC, Europe PMC, OpenAlex, Crossref, Semantic Scholar… |
| `scientific-critical-thinking` | 1.5 | Đánh giá chất lượng bằng chứng, thiên lệch, GRADE |

**Đã chỉnh so với bản gốc:**
- Bỏ mục "Citing Scientific Agent Skills" ở cuối mọi skill. Mục này yêu cầu Claude tự thêm bài báo của K-Dense vào danh mục tài liệu tham khảo trong bài của bạn.
- Bỏ phần vẽ sơ đồ bằng AI (`generate_schematic.py`) trong `literature-review` và `scientific-critical-thinking`. Phần này gửi nội dung lên dịch vụ trả phí OpenRouter và tìm API key trong file `.env`.
- Rút gọn mô tả của `paper-lookup` (từ ~200 xuống ~80 token).
- Viết lại mô tả 4 skill dự án bằng tiếng Anh và nói rõ phạm vi, để không trùng với skill tổng quát.

Đã quét các script đi kèm tìm lệnh gọi mạng, chạy lệnh hệ thống và đọc biến môi trường: chỉ thấy gọi API học thuật công khai (OpenAlex, Crossref, arXiv, bioRxiv, EBI) và Pandoc khi xuất PDF.

### Đã xem nhưng không cài
| Repo | Lý do |
|---|---|
| `anthropics/skills` (chính thức) | Bạn đã có sẵn qua plugin `anthropic-skills` (docx, pdf, pptx, xlsx, skill-creator) |
| `obra/superpowers` | Thiên về lập trình (TDD, debug); ít giá trị cho nghiên cứu |
| Các danh sách `awesome-claude-skills` (travisvn, ComposioHQ…) | Chỉ là danh sách liên kết, nhiều skill chưa được kiểm định |
| Phần còn lại của K-Dense (~130 skill: hoá học, gen, lượng tử…) | Không liên quan đề tài, chỉ làm phình mô tả. Chính K-Dense cũng cảnh báo nhiều skill cộng đồng chưa được rà soát kỹ |

## 2b. Tự thêm skill khi cần: `skill-scout`
Bạn không cần tự đi tìm skill nữa. Khi bạn hỏi việc mới mà chưa có skill phù hợp, `skill-scout` sẽ:
1. Tra danh mục ~330 skill đã kiểm duyệt (`skill-scout/references/catalog.tsv`) bằng `grep`, không đọc cả file, để đỡ tốn token:
   - 163 skill khoa học/dữ liệu của K-Dense (MIT)
   - 17 skill của Anthropic (Word, Excel, PowerPoint, PDF, thiết kế, tạo skill…)
   - 135 skill từ 15 plugin chính thức của Anthropic cho công việc văn phòng: bán hàng, tài chính, pháp lý, marketing, chăm sóc khách hàng, quản lý sản phẩm, nhân sự, vận hành, kỹ thuật, thiết kế, dữ liệu, năng suất cá nhân… (Apache-2.0; bỏ các plugin của bên thứ ba)
   - 15 skill `obra/superpowers` cho lập trình có kỷ luật: gỡ lỗi có hệ thống, TDD, lập kế hoạch, review code (MIT)
2. Chọn tối đa 2 skill phù hợp, tải về bằng `scripts/fetch_skill.py`. Script tự bỏ đoạn quảng cáo trích dẫn, kiểm tra định dạng và quét nội dung rủi ro.
3. Nếu sạch: **Claude Code** tự cài vào `.claude/skills/` (hoặc `~/.claude/skills/`). **Chat/Cowork** tạo file ZIP để bạn tải lên, vì Claude không có quyền tải lên tài khoản của bạn. Dù ở đâu, Claude cũng đọc skill và dùng ngay cho câu hỏi đang làm.
4. Nếu có rủi ro (gửi dữ liệu cho dịch vụ AI trả phí, đọc file `.env`/mật khẩu, tải-và-chạy lệnh…): **dừng lại và hỏi bạn** trước khi cài.

Muốn gọi thẳng thì nói "tìm skill cho …" hoặc gõ `/skill-scout`.

Với nhóm công việc văn phòng (bán hàng, tài chính, pháp lý…), nếu cần nhiều skill cùng mảng thì nên **bật cả plugin** thay vì tải lẻ từng skill: trong Cowork/claude.ai vào thư mục plugin (claude.com/plugins hoặc thanh Customize); trong Claude Code chạy `claude plugin marketplace add anthropics/knowledge-work-plugins` rồi `claude plugin install <tên-plugin>@knowledge-work-plugins`.

## 3. Dùng trong chat claude.ai và Cowork
Skill trong repo **không** tự có ở chat hay Cowork; phải tải lên tài khoản:

1. Nén từng thư mục skill thành một file ZIP, thư mục nằm ở gốc ZIP (ví dụ `neurokit2.zip` chứa `neurokit2/SKILL.md`). Tên thư mục phải trùng `name:` trong `SKILL.md`.
2. Trên claude.ai: **Customize → Skills → "+" → Create skill → Upload a skill**, chọn file ZIP.
3. Bật công tắc của skill vừa tải lên.
4. Cần bật **Code execution** (thực thi mã) trong cài đặt; skill không chạy nếu tắt.

Tải lên một lần là dùng được ở chat và Cowork. Claude Code đăng nhập cùng tài khoản (bản v2.1.273 trở lên) cũng tự đồng bộ các skill này về.

Nên tải: **`skill-scout`** (để chat/Cowork tự tìm skill mới) + 7 skill K-Dense + `small-sample-stats`, `lit-review`, `research-writing-vn`. Skill `ecg-brugada-pipeline` chủ yếu hướng dẫn sửa `ecg_pipeline.py`, nên chỉ hữu ích khi có repo; vẫn tải được nếu muốn Claude nhớ các định nghĩa đo ECG.

## 4. Dùng cho mọi dự án trong Claude Code trên máy bạn
Chép vào thư mục skill của người dùng (áp dụng cho mọi repo):

```bash
mkdir -p ~/.claude/skills && cp -r .claude/skills/{neurokit2,statistical-analysis,scientific-visualization,scientific-writing,literature-review,paper-lookup,scientific-critical-thinking} ~/.claude/skills/
```

Windows PowerShell:

```powershell
foreach ($s in "neurokit2","statistical-analysis","scientific-visualization","scientific-writing","literature-review","paper-lookup","scientific-critical-thinking") { Copy-Item -Recurse ".claude\skills\$s" "$env:USERPROFILE\.claude\skills\" }
```

Bốn skill dự án (`ecg-brugada-pipeline`…) nên để lại trong repo vì chỉ đúng cho đề tài này.

## 5. Thói quen tiết kiệm token (quan trọng hơn chọn skill)
Phần lớn token bị tiêu vào **lịch sử hội thoại dài, file lớn bị đọc nguyên, và kết quả lệnh dài**, không phải vào skill.

1. **Mỗi việc một phiên.** Xong việc thì gõ `/clear`. Mỗi tin nhắn mới phải gửi kèm toàn bộ lịch sử phía trước, nên phiên càng dài, mỗi câu càng đắt.
2. **Xem token đang đi đâu**: `/context` hiện phần nào chiếm bao nhiêu (CLAUDE.md, skill, hội thoại).
3. **CLAUDE.md ngắn và bằng tiếng Anh.** File này nạp ở *mọi* phiên. Cùng nội dung, tiếng Việt có dấu thường tốn nhiều token hơn đáng kể so với tiếng Anh. Vì vậy CLAUDE.md của dự án đã được viết lại bằng tiếng Anh, giữ nguyên quy tắc "luôn trả lời bằng tiếng Việt". Claude vẫn trả lời bạn bằng tiếng Việt.
4. **Nói rõ file và việc cần làm.** Ví dụ "sửa hàm `delineate` trong `ecg_pipeline.py` để…" thay vì "sửa code cho tốt hơn". Claude đỡ phải đi tìm.
5. **Đừng dán dữ liệu lớn vào chat.** Đưa đường dẫn file; Claude sẽ đọc phần cần thiết.
6. **Chọn model theo việc**: model mạnh cho thiết kế phương pháp và review; model nhẹ hơn cho sửa lặt vặt, định dạng.
7. **Tra cứu nhiều nguồn**: một lượt tìm kiếm rộng có thể tốn nhiều; hãy nêu câu hỏi cụ thể (tác giả, năm, khái niệm) để Claude tìm trúng ngay.
