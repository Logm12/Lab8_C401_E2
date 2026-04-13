# Báo Cáo Cá Nhân - Lab Day 08: RAG Pipeline

**Họ và tên:** Cao Chí Hải  
**Vai trò trong nhóm:** Sprint 1-3 Developer (Indexing, Retrieval, Grounded Answer)  
**Ngày nộp:** 13/04/2026  
**Độ dài yêu cầu:** 500-800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab này, tôi phụ trách trực tiếp hai file `index.py` và `rag_answer.py`, tức là phần cốt lõi của pipeline RAG từ khâu xây chỉ mục đến khâu truy xuất và sinh câu trả lời. Ở `index.py`, tôi triển khai bước tiền xử lý tài liệu bằng cách tách metadata từ phần đầu file như `source`, `department`, `effective_date`, `access`, sau đó làm sạch nội dung trước khi chunk. Tôi cũng xây dựng cơ chế chia tài liệu theo section heading `=== ... ===`, rồi tiếp tục tách theo paragraph nếu section quá dài để giữ ngữ nghĩa tự nhiên. Sau đó, tôi tích hợp embedding bằng OpenAI và lưu chunk vào ChromaDB với metadata đầy đủ. Ở `rag_answer.py`, tôi triển khai dense retrieval, sparse retrieval bằng BM25, hybrid retrieval bằng Reciprocal Rank Fusion, hàm tạo prompt grounded, hàm gọi LLM, và pipeline `rag_answer()`.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Điều tôi hiểu rõ hơn sau lab này là một hệ thống RAG tốt không chỉ phụ thuộc vào model mạnh, mà phụ thuộc rất lớn vào chất lượng indexing và retrieval. Trước khi làm bài này, tôi thường nghĩ embedding là phần quan trọng nhất. Tuy nhiên khi tự viết `preprocess_document()` và `chunk_document()`, tôi nhận ra nếu metadata sai hoặc chunk bị cắt không hợp lý thì vector search phía sau cũng khó trả về đúng bằng chứng. Tôi cũng hiểu rõ hơn sự khác nhau giữa dense retrieval và sparse retrieval. Dense retrieval mạnh ở việc nắm nghĩa tổng quát của câu hỏi, còn sparse retrieval hữu ích khi câu hỏi chứa từ khóa đặc biệt hoặc mã lỗi. Khi kết hợp hai cách bằng hybrid retrieval, hệ thống cân bằng tốt hơn giữa hiểu ngữ nghĩa và bám sát từ khóa.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)
Khó khăn lớn nhất tôi gặp là làm sao để chunking vừa đủ ngắn cho embedding, nhưng vẫn giữ nguyên ranh giới tự nhiên của tài liệu. Nếu cắt quá cứng theo độ dài ký tự thì rất dễ làm mất ý nghĩa của một điều khoản; nhưng nếu giữ nguyên section quá dài thì retrieval lại kém chính xác. Vì vậy tôi chọn hướng chia theo heading trước, rồi mới chia tiếp theo paragraph và thêm overlap ở mức hợp lý. Tôi cũng khá ngạc nhiên khi phần prompt trong `build_grounded_prompt()` ảnh hưởng mạnh đến chất lượng câu trả lời. Chỉ cần không ràng buộc rõ việc “chỉ dùng thông tin trong context” và “không được tự bịa”, LLM có thể suy diễn thêm những chi tiết không có trong tài liệu.

---

## 4. Phân tích một phần kỹ thuật tôi trực tiếp làm (150-200 từ)

Phần tôi thấy quan trọng nhất trong đóng góp của mình là thiết kế retrieval trong `rag_answer.py`, đặc biệt là `retrieve_hybrid()`. Với dense retrieval, hệ thống lấy embedding của câu hỏi, truy vấn ChromaDB và xếp hạng các chunk theo độ tương đồng ngữ nghĩa. Cách này hoạt động tốt với những câu hỏi diễn đạt tự nhiên như “SLA xử lý ticket P1 là bao lâu?”. Tuy nhiên, khi query chứa từ khóa đặc thù hoặc tên cũ của tài liệu, dense retrieval có thể bỏ sót bằng chứng vì embedding ưu tiên nghĩa tổng quát hơn là khớp chính xác cụm từ. Đó là lý do tôi bổ sung `retrieve_sparse()` bằng BM25 để khai thác keyword matching. Sau đó, trong `retrieve_hybrid()`, tôi dùng Reciprocal Rank Fusion để hợp nhất thứ hạng từ hai nguồn. Kết quả là hệ thống vừa trả lời tốt các câu hỏi ngôn ngữ tự nhiên, vừa xử lý tốt hơn những truy vấn thiên về keyword như mã lỗi hoặc tên tài liệu.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi muốn hoàn thiện phần `rerank()` thay vì mới dừng ở mức chọn top-k đầu tiên. Tôi sẽ dùng cross-encoder để chấm lại mức độ liên quan của các candidate chunk sau khi search rộng, từ đó giảm nhiễu trước khi đưa context vào prompt. Ngoài ra, tôi cũng muốn bổ sung test cho chunking, metadata, và khả năng abstain của `rag_answer()`.