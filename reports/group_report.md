# Báo cáo nhóm — Lab Day 08: Full RAG Pipeline

**Tên nhóm:** C401-E2
**Thành viên:**
| Tên | Vai trò | Email |
|-----|---------|-------|
| Mạc Phạm Thiên Lọng | Tech Lead | longmac321@gmail.com |
| Cao Chí Hải | Retrieval Owner | caochihai1710@gmail.com |
| Nguyễn Doãn Hiếu| Eval Owner | |
| Bùi Hữu Huấn | Documentation Owner | anhhuanvg02@gmail.com  |

**Ngày nộp:** 2026-04-13  
**Repo:** https://github.com/Logm12/Lab8_C401_E2 

---

## 1. Pipeline nhóm đã xây dựng

Hệ thống trợ lý được xây dựng trên kiến trúc RAG hiện đại, tối ưu cho việc tra cứu các tài liệu chính sách và quy trình có độ chính xác cao.

**Chunking decision:**
Nhóm quyết định sử dụng chiến lược Paragraph-based chunking với kích thước 400 ký tự và 80 ký tự overlap. Lựa chọn này dựa trên đặc điểm của bộ dữ liệu (SOP và policy)  trình bày thông tin theo các điều khoản hoặc các bước thực hiện trong mỗi đoạn văn. Việc giữ kích thước chunk 400 tokens giúp AI không bị loãng thông tin khi xử lý đồng thời nhiều tài liệu.

**Embedding model:**
Hệ thống sử dụng model **text-embedding-3-small** của OpenAI với 1536 dimensions. Đây là lựa chọn tối ưu về chi phí nhưng vẫn đảm bảo khả năng biểu diễn ngữ nghĩa vượt trội so với các model mã nguồn mở khác.

**Retrieval variant (Sprint 3):**
Nhóm đã triển khai **Hybrid retrieval (Dense + Sparse)**. Kết quả thực tế cho thấy Dense search (Vector) rất mạnh trong các câu hỏi ngữ nghĩa, nhưng lại thường xuyên bỏ lỡ các mã lỗi kỹ thuật (như "ERR-403") hoặc các tên tài liệu cũ (như "Approval Matrix"). Việc kết hợp thêm BM25 giúp hệ thống đạt độ chính xác cao trong các truy vấn có từ khóa đặc thù.

---

## 2. Quyết định kỹ thuật quan trọng nhất

**Quyết định:** Chuyển đổi từ "Abstain tuyệt đối" sang "Helpful refusal" (PA 2).

**Bối cảnh vấn đề:**
Trong sprint 2, hệ thống được yêu cầu chỉ trả lời "Không đủ dữ liệu" nếu không tìm thấy bằng chứng trong tài liệu. Tuy nhiên, khi chạy thử với các câu hỏi thực tế của người dùng doanh nghiệp, câu trả lời này gây ra trải nghiệm tiêu cực vì người dùng không biết phải làm gì tiếp theo.

**Các phương án đã cân nhắc:**

| Phương án | Ưu điểm | Nhược điểm |
|-----------|---------|-----------|
| Abstain tuyệt đối | An toàn 100%, không bao giờ hallucinate. | Không hỗ trợ được người dùng khi tài liệu thiếu. |
| Helpful refusal | Chuyên nghiệp, hướng dẫn người dùng liên hệ bộ phận liên quan (IT, HR). | Nếu không kiểm soát chặt, AI có thể tự bịa ra thông tin liên hệ. |

**Phương án đã chọn và lý do:**
Nhóm chọn **Helpful refusal**. Chúng tôi điều chỉnh Prompt để AI xác nhận việc không tìm thấy thông tin trong tài liệu, sau đó sử dụng thông tin Metadata (Department) hoặc các thông tin liên hệ trong Context (nếu có) để gợi ý người dùng. Điều này giúp tăng điểm hữu ích mà vẫn giữ được tính trung thực.

**Bằng chứng từ scorecard/tuning-log:**
Trong đợt chạy `grading_questions`, tại câu **gq07** (Hỏi về mức phạt SLA không có trong tài liệu), hệ thống không chỉ trả lời thiếu thông tin mà còn gợi ý liên hệ bộ phận IT. Kết quả này giúp điểm **Relevance** và **Completeness** được cải thiện rõ rệt so với bản baseline ban đầu.

---

## 3. Kết quả grading questions

Hệ thống đã hoàn thành 10 câu hỏi thử thách từ BTC với cấu hình Hybrid retrieval.

**Ước tính điểm raw:** 82 / 98  
(Điểm số này dựa trên việc hệ thống xử lý tốt các câu hỏi khó về tổng hợp đa tài liệu và từ chối trả lời đúng lúc).

**Câu tốt nhất:** ID: **gq01** — Lý do: Hệ thống đã bóc tách chính xác sự thay đổi giữa phiên bản cũ (6 giờ) và phiên bản mới (4 giờ) nhờ metadata `effective_date` và logic xử lý tiêu đề chính xác.

**Câu fail:** ID: **gq05** — Root cause: indexing/retrieval. Câu hỏi về quyền Admin Access của contractor nằm ở hai section cách xa nhau. Với `top_k_select=4`, các đoạn văn bản chứa chi tiết về quy trình phê duyệt Level 4 đã đẩy đoạn văn bản nói về phạm vi áp dụng (Section 1) xuống dưới, khiến AI kết luận thiếu căn cứ.

**Câu gq07 (abstain):** Pipeline xử lý tốt. AI nhận định tài liệu không có quy định về mức phạt và hướng dẫn liên hệ bộ phận IT chuyên trách.

---

## 4. A/B Comparison — Baseline vs Variant

Hệ thống được so sánh giữa chế độ tìm kiếm chỉ dùng vector (Baseline) và tìm kiếm kết hợp (Variant - Hybrid).

**Biến đã thay đổi (chỉ 1 biến):** `retrieval_mode (dense -> hybrid)`.

| Metric | Baseline | Variant | Delta |
|--------|---------|---------|-------|
| Faithfulness | 3.40 | 3.80 | +0.40 |
| Answer relevance | 4.70 | 4.70 | 0.00 |
| Context recall | 5.00 | 5.00 | 0.00 |
| Completeness | 5.00 | 4.60 | -0.40 |

**Kết luận:**
Variant (Hybrid) cho kết quả tốt hơn về Faithfulness. Việc BM25 tham gia vào quá trình tìm kiếm giúp thu hẹp phạm vi context về đúng những đoạn có chứa keyword, giúp AI giảm bớt việc suy diễn lan man. Dù Completeness giảm nhẹ do AI thận trọng hơn trong các câu trả lời phức tạp, nhưng đây là trade-off cần thiết để đảm bảo an toàn thông tin trong doanh nghiệp.

---

## 5. Phân công và đánh giá nhóm

**Phân công thực tế:**

| Thành viên | Phần đã làm | Sprint |
|------------|-------------|--------|
| Mạc Phạm Thiên Lọng | Thiết kế kiến trúc tổng thể, sửa lỗi xử lý header trong index.py, quản lý chung. | 1, 2, 4 |
| Cao Chí Hải | Triển khai Hybrid retrieval (Dense + BM25), tối ưu thuật toán RRF fusion. | 3 |
| Nguyễn Doãn Hiếu | Xây dựng bộ khung Evaluation, triển khai LLM-as-Judge để tự động hóa chấm điểm. | 4 |
| Bùi Hữu Huấn Huấn | Tiền xử lý dữ liệu docs, bóc tách metadata, hoàn thiện tài liệu architecture.md. | 1, 4 |

**Điều nhóm làm tốt:**
- Phối hợp nhịp nhàng giữa khâu Preprocessing và Retrieval. Việc sửa lỗi mất dữ liệu header ngay từ sprint 1 đã cải thiện kết quả khi kiểm thử nhiều câu hỏi khó ở Sprint 4.
- Ứng dụng thành công kỹ thuật LLM-as-Judge giúp tiết kiệm thời gian chấm điểm thủ công 10 câu hỏi với 2 cấu hình khác nhau.

**Điều nhóm làm chưa tốt:**
- Quản lý context window còn hạn chế. Ở các câu hỏi cần tổng hợp thông tin từ hơn 4-5 đoạn văn bản cách xa nhau, hệ thống vẫn có xu hướng bỏ sót chi tiết do giới hạn `top_k_select`.

---

## 6. Nếu có thêm 1 ngày, nhóm sẽ làm gì?

Nhóm sẽ triển khai thêm bước **Reranking** bằng mô hình Cross-encoder chuyên dụng. Theo quan sát từ `scorecard_variant.md`, một số câu hỏi fail là do BM25 mang về các đoạn văn bản chứa đúng từ khóa nhưng sai ngữ cảnh (Noise). Một bộ Reranker sẽ giúp lọc bỏ các nhiễu này, đảm bảo 4 đoạn văn bản đưa vào LLM đều là tốt nhất, từ đó đẩy điểm Faithfulness lên mức trên 4.5.
