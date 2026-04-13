# Architecture diagram

Kiến trúc hệ thống RAG trợ lý nội bộ được xây dựng trên mô hình Retriever-Generator chuẩn doanh nghiệp, tập trung vào tính trung thực và khả năng trích dẫn nguồn dữ liệu.

<p align="center">
  <img src="Architechture Diagram.png" width="400">
</p>

## Mô tả hệ thống

Hệ thống được thiết kế để giải quyết nhu cầu tra cứu thông tin chính xác trong các tài liệu chính sách (Policy), quy trình vận hành (SOP) và câu hỏi thường gặp (FAQ) của doanh nghiệp. Điểm cốt lõi của giải pháp là việc kết hợp giữa sức mạnh tìm kiếm theo ngữ nghĩa và khả năng bắt từ khóa chính xác để tránh sai sót trong các thuật ngữ chuyên ngành.

---

## Thành phần indexing

### Tài liệu được index
| File | Nguồn | Department | Số chunk |
|------|-------|-----------|---------|
| `policy_refund_v4.txt` | policy/refund-v4.pdf | CS | 7 |
| `sla_p1_2026.txt` | support/sla-p1-2026.pdf | IT | 6 |
| `access_control_sop.txt` | it/access-control-sop.md | IT Security | 8 |
| `it_helpdesk_faq.txt` | support/helpdesk-faq.md | IT | 6 |
| `hr_leave_policy.txt` | hr/leave-policy-2026.pdf | HR | 7 |

### Quyết định chunking
| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| Chunk size | 400 ký tự | Cân bằng giữa việc giữ đủ ngữ cảnh và tránh làm loãng thông tin khi đưa vào prompt. |
| Overlap | 80 ký tự | Đảm bảo tính liên kết giữa các đoạn văn, tránh việc thông tin quan trọng bị cắt đôi. |
| Chunking strategy | Paragraph-based | Dựa trên ranh giới tự nhiên của đoạn văn để giữ trọn vẹn ý nghĩa của một điều khoản. |
| Metadata fields | source, section, effective_date, department | Phục vụ cho việc lọc dữ liệu, trích dẫn nguồn và kiểm tra tính cập nhật. |

### Embedding model
- **Model**: OpenAI text-embedding-3-small (1536 dims).
- **Vector store**: ChromaDB (Persistent client).
- **Similarity metric**: Cosine similarity.

---

## Thành phần Retrieval

### Baseline (Sprint 2)
- **Chiến lược**: Dense retrieval (Tìm kiếm vector).
- **Top-k search**: 10.
- **Top-k select**: 4.

### Variant (Sprint 3)
| Tham số | Giá trị | Thay đổi so với baseline |
|---------|---------|------------------------|
| Strategy | Hybrid (Dense + Sparse) | Kết hợp thế mạnh của Vector similarity và BM25 keyword matching. |
| Top-k search | 10 | Giữ nguyên chiều sâu tìm kiếm. |
| Top-k select | 4 | Tăng số lượng ngữ cảnh cung cấp cho mô hình ngôn ngữ. |
| Fusion algorithm | RRF (Reciprocal rank fusion) | Sử dụng thuật toán hòa trộn xếp hạng để lấy ra các đoạn văn bản có điểm số cao nhất từ cả hai phương pháp. |

**Lý do chọn variant này:**
Trong môi trường doanh nghiệp, người dùng thường sử dụng cả câu hỏi tự nhiên lẫn các mã lỗi hoặc thuật ngữ chuyên môn không đổi (ví dụ: "SLA P1", "ERR-403"). Hybrid retrieval giúp bắt trúng các từ khóa này trong khi vẫn hiểu được ý nghĩa của câu hỏi.

---

## Thành phần Generation

### Prompting strategy
- **Grounded answering**: Ép mô hình chỉ trả lời dựa trên context được cung cấp.
- **Mandatory citations**: Yêu cầu đánh dấu số thứ tự nguồn [1], [2] ngay sau thông tin được trích dẫn.
- **Helpful refusal**: Thay vì chỉ nói "Không đủ dữ liệu", mô hình sẽ gợi ý các bộ phận liên quan (IT Helpdesk, HR...) dựa trên tài liệu có sẵn.

### LLM Configuration
| Tham số | Giá trị |
|---------|---------|
| Model | OpenAI gpt-4o-mini / 5.4-mini |
| Temperature | 0 (Đảm bảo tính ổn định và khách quan cho kết quả đánh giá) |
| Max tokens | 512 |

---

## Thành phần Evaluation

Hệ thống được đánh giá định lượng qua 4 chỉ số chính:
1. **Faithfulness**: Độ trung thực của câu trả lời so với ngữ cảnh (Hallucination check).
2. **Answer relevance**: Độ liên quan của câu trả lời đối với câu hỏi gốc.
3. **Context recall**: Khả năng của retriever tìm thấy đúng tài liệu chứa đáp án.
4. **Completeness**: Độ đầy đủ của thông tin so với đáp án mẫu (Gold answer).
