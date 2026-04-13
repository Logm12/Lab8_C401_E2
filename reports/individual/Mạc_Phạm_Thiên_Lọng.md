# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Mạc Phạm Thiên Long
**Vai trò trong nhóm:** Tech Lead / Retrieval Owner / Eval Owner / Documentation Owner  
**Ngày nộp:** 13/04/2026  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong Lab này, với vai trò Tech Lead, tôi đã trực tiếp tham gia vào toàn bộ vòng đời phát triển của hệ thống RAG. 
Ở Sprint 1 & 2, tôi chịu trách nhiệm thiết kế chiến lược chunking dựa trên đoạn văn (`\n\n`) để đảm bảo không cắt ngang các điều khoản quan trọng. Tôi cũng là người thực hiện logic kết nối với OpenAI để lấy Embedding và xây dựng hàm `rag_answer` cơ sở, đặc biệt là việc thiết kế Prompt tiếng Việt nghiêm ngặt để giải quyết bài toán "Ảo giác" (Abstain).
Ở Sprint 3, tôi đã trực tiếp cài đặt thuật toán Hybrid Retrieval (kết hợp Dense/Vector và Sparse/BM25) sử dụng phương pháp Reciprocal Rank Fusion (RRF). Đây là quyết định quan trọng nhất giúp hệ thống xử lý được các query về mã lỗi và tên tài liệu cũ.
Ở Sprint 4, tôi xây dựng hệ thống đánh giá tự động "LLM-as-Judge", giúp nhóm tiết kiệm hàng giờ đồng hồ chấm điểm thủ công 10 câu hỏi test qua nhiều cấu hình khác nhau.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Concept mà tôi tâm đắc nhất chính là **Hybrid Retrieval và cơ chế RRF**. Trước đây, tôi cứ nghĩ Vector Search là "chìa khóa vạn năng", nhưng thực tế cho thấy Dense search đôi khi quá "mơ mộng" — nó tìm theo ý nghĩa nhưng lại bỏ qua những keyword quan trọng nhất (như mã lỗi ERR-403). 

Việc hiểu về RRF giúp tôi thấy được cách chúng ta có thể kết hợp cái "trí óc" của Vector với cái "tỉ mỉ" của Keyword search. Ngoài ra, tôi cũng hiểu sâu hơn về **Grounded Evaluation**. Việc chỉ có câu trả lời là chưa đủ, chúng ta cần các bộ chỉ số (Faithfulness, Recall) để định lượng được sự thay đổi của Code có thực sự mang lại giá trị hay không. Việc đánh giá dựa trên dữ liệu giúp chúng ta tránh được các quyết định cảm tính khi tinh chỉnh (tuning) tham số.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Tôi thực sự ngạc nhiên trước sự nhạy cảm của LLM đối với Prompt. Ở Sprint 2, chỉ cần quên không dặn AI "Không được tự ý suy diễn", nó sẵn sàng bịa ra một quy trình hoàn tiền cho khách hàng VIP dù trong tài liệu không hề có. Điều này cho tôi thấy RAG không chỉ là bài toán lập trình, mà còn là bài toán về **kiểm soát hành vi (guardrailing)** thông qua kỹ thuật Prompt Engineering.

Khó khăn lớn nhất tôi gặp phải là lỗi `max_tokens` của OpenAI API khi họ chuyển sang model mới. Việc hệ thống báo lỗi mà không rõ nguyên nhân ban đầu buộc tôi phải sử dụng kỹ năng **Systematic Debugging**, đọc kỹ tài liệu API mới nhất để phát hiện ra tham số đã đổi thành `max_completion_tokens`. Đây là một bài học đắt giá về việc luôn phải cập nhật với sự thay đổi của các bên cung cấp dịch vụ thứ ba.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** "Approval Matrix để cấp quyền hệ thống là tài liệu nào?"

**Phân tích:** 
- **Baseline (Dense):** Trả về kết quả không chính xác. Lý do là vì trong tài liệu mới, cụm từ "Approval Matrix" chỉ xuất hiện một lần duy nhất trong phần ghi chú về lịch sử tên tài liệu cũ. Embedding vector của query "Approval Matrix" bị kéo về phía các tài liệu HR liên quan đến phê duyệt nghỉ phép thay vì đúng tài liệu IT Security do sự tương đồng về ngữ nghĩa chung của từ "Approval".
- **Variant (Hybrid):** Kết quả cải thiện rõ rệt (đạt điểm 5/5 về Context Recall). Nhờ có BM25, từ khóa "Approval Matrix" được tìm thấy chính xác trong đoạn ghi chú của file `access-control-sop.md`. RRF đã đẩy chunk này lên vị trí đầu tiên dù điểm Dense không cao. LLM sau đó đọc được đúng dòng "Tài liệu này trước đây có tên Approval Matrix..." và trả lời cực kỳ chính xác cho người dùng. Đây là minh chứng rõ nhất cho sức mạnh của Hybrid Retrieval trong môi trường doanh nghiệp thực tế.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm 1 giờ, tôi sẽ thực hiện **Reranking với Cross-Encoder**. Hiện tại, BM25 đôi khi mang về những đoạn văn có chứa từ khóa nhưng nội dung thực sự không liên quan. Một mô hình Cross-Encoder (như `ms-marco-MiniLM-L-6-v2`) sẽ đóng vai trò lọc lại lần cuối, chấm điểm mức độ liên quan thực sự giữa câu hỏi và 10 đoạn văn được tìm thấy. Điều này sẽ giúp loại bỏ nhiễu và tăng tính chính xác (Faithfulness) tuyệt đối cho câu trả lời của AI.
