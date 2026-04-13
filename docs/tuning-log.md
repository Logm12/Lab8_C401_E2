# Tuning log — RAG pipeline

Bản ghi nhật ký này ghi lại quá trình tinh chỉnh các tham số hệ thống nhằm tối ưu hóa hiệu quả truy xuất và trả lời câu hỏi. Mọi thí nghiệm đều tuân thủ nguyên tắc A/B rule (chỉ thay đổi một biến số duy nhất tại mỗi thời điểm).

---

## Baseline (Sprint 2)

**Ngày ghi nhận:** 2026-04-13  
**Mục tiêu:** Thiết lập mốc hiệu năng cơ bản sử dụng tìm kiếm vector (Dense retrieval).

**Cấu hình hệ thống:**
```python
retrieval_mode = "dense"
chunk_size = 400
chunk_overlap = 80
top_k_search = 10
top_k_select = 4
use_rerank = False
llm_model = "gpt-4o-mini"
```

**Kết quả đo lường (Trung bình):**
| Chỉ số | Điểm số |
|--------|---------|
| Faithfulness | 3.40 / 5 |
| Answer relevance | 4.70 / 5 |
| Context recall | 5.00 / 5 |
| Completeness | 5.00 / 5 |

**Phân tích lỗi:**
- Hệ thống gặp khó khăn với các câu hỏi yêu cầu khớp từ khóa chính xác (Keyword matching) như mã lỗi hoặc tên cũ của tài liệu.
- Điểm Faithfulness thấp do mô hình đôi khi bị "hallucination" khi không tìm thấy bằng chứng cụ thể trong context nhưng vẫn cố gắng trả lời dựa trên kiến thức sẵn có của mô hình ngôn ngữ lớn.

---

## Variant 1 (Sprint 3)

**Ngày ghi nhận:** 2026-04-13  
**Biến số thay đổi:** `retrieval_mode` từ `"dense"` chuyển sang `"hybrid"`  
**Lý do lựa chọn:** 
Các tài liệu chính sách của doanh nghiệp thường chứa nhiều thuật ngữ kỹ thuật, mã lỗi (ERR-403) và các tên gọi cũ (Approval Matrix). Tìm kiếm Hybrid kết hợp giữa Semantic search (Dense) và Keyword search (BM25) sẽ giúp cải thiện khả năng thu thập chính xác các đoạn văn bản có chứa từ khóa chuyên mục.

**Cấu hình hệ thống:**
```python
retrieval_mode = "hybrid"
# Các tham số khác được giữ nguyên theo Baseline
```

**Kết quả so sánh (So với baseline):**
| Chỉ số | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 3.40 | **3.80** | **+0.40** |
| Answer relevance | 4.70 | 4.70 | 0.00 |
| Context recall | 5.00 | 5.00 | 0.00 |
| Completeness | 5.00 | 4.60 | -0.40 |

**Nhận xét chuyên môn:**
1. **Sự cải thiện của Faithfulness**: Việc sử dụng Hybrid giúp hệ thống lấy được các đoạn văn bản chứa chính xác thuật ngữ kỹ thuật. Khi có context chính xác, AI ít bị rơi vào tình trạng suy diễn sai lệch, dẫn đến điểm trung thực tăng lên.
2. **Sự sụt giảm nhẹ của Completeness**: Trong phiên bản Hybrid kết hợp với Prompt "Helpful Refusal", hệ thống trở nên thận trọng hơn. Khi thông tin không thực sự chắc chắn, AI chọn cách từ chối trả lời hoặc hướng dẫn liên hệ bộ phận hỗ trợ thay vì đưa ra câu trả lời đầy đủ nhưng có nguy cơ sai sót. Đây là một sự đánh đổi tích cực trong môi trường doanh nghiệp (ưu tiên an toàn hơn đầy đủ).
3. **Context recall**: Cả hai phương án đều tìm thấy nguồn tài liệu rất tốt (điểm 5.0 tuyệt đối), cho thấy bộ dữ liệu hiện tại đủ nhỏ để retriever hoạt động hiệu quả.

---

## Tổng kết kinh nghiệm

1. **Lỗi hệ thống phổ biến nhất**:
Lỗi "Nuốt dữ liệu" trong khâu tiền xử lý (Preprocessing). Trước khi điều chỉnh, hệ thống đã bỏ qua các dòng tiêu đề quan trọng và các ghi chú phiên bản. Sau khi sửa lỗi này, khả năng trả lời các câu hỏi về lịch sử thay đổi (Version history) và tên gọi cũ (Alias) đã được cải thiện đáng kể.

2. **Yếu tố tác động lớn nhất**:
Sự kết hợp giữa `Hybrid retrieval` và chiến lược `Prompting PA 2` (Từ chối hữu ích). Điều này giúp hệ thống không chỉ tìm thấy dữ liệu tốt hơn mà còn giao tiếp chuyên nghiệp hơn khi đối mặt với các câu hỏi nằm ngoài phạm vi tài liệu.

3. **Hướng phát triển tiếp theo**:
Để nâng cao hơn nữa chỉ số Faithfulness, việc triển khai **Reranking** (Sử dụng Cross-Encoder) là cần thiết. Bước này sẽ giúp sắp xếp lại thứ tự của top 10 chunks một cách thông minh hơn trước khi chọn lọc ra top 4 đưa vào mô hình ngôn ngữ, loại bỏ các đoạn văn bản gây nhiễu cho AI.
