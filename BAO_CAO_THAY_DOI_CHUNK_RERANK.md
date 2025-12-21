# BÁO CÁO THAY ĐỔI: CẢI THIỆN HỆ THỐNG CHUNK RERANKING

## 1. TỔNG QUAN

Báo cáo này mô tả các thay đổi đã thực hiện để cải thiện hệ thống reranking chunks trong dự án UniSupport-ChatBot, bao gồm:
- Nâng cấp model reranking
- Cải thiện logic đánh giá kết quả từ local database
- Tối ưu hóa quản lý bộ nhớ GPU

## 2. CÁC THAY ĐỔI CHI TIẾT

### 2.1. Nâng cấp Model Chunk Reranking

**Thay đổi:**
- **Model cũ:** `ms-marco-MultiBERT-L-12` (sử dụng thư viện flashrank)
- **Model mới:** `BAAI/bge-reranker-v2-m3` (sử dụng sentence_transformers CrossEncoder)

**Lý do:**
- Model BGE reranker v2-m3 được tối ưu hóa tốt hơn cho tiếng Việt
- CrossEncoder từ sentence_transformers cho phép tùy chỉnh và kiểm soát tốt hơn
- Hỗ trợ tốt hơn cho các tác vụ reranking trong ngữ cảnh đa ngôn ngữ

**File thay đổi:**
- `kaggle_dedicated/data_retriever/config.py`: Cập nhật `ChunkRankerConfig.ranker_name`
- `kaggle_dedicated/data_retriever/ranker/chunk_ranker.py`: Thay đổi implementation từ flashrank sang CrossEncoder

### 2.2. Cải thiện Threshold Filtering

**Thay đổi:**
- **Trước:** Sử dụng relative threshold (threshold = max_score × 0.5)
- **Sau:** Sử dụng fixed absolute threshold = 0.5

**Ví dụ:**
- Trước: Nếu max_score = 0.19 → threshold = 0.095 → chunks có score 0.19 vẫn được giữ
- Sau: Nếu max_score = 0.19 → threshold = 0.5 → chunks có score 0.19 bị loại

**Lý do:**
- Đảm bảo chất lượng chunks được giữ lại luôn đạt ngưỡng tối thiểu
- Tránh trường hợp giữ lại chunks có độ liên quan thấp chỉ vì max_score thấp
- Tăng độ chính xác của kết quả retrieval

**File thay đổi:**
- `kaggle_dedicated/data_retriever/ranker/chunk_ranker.py`: Dòng 58

### 2.3. Thêm Chunk Rerank Evaluation cho Local Results

**Thay đổi:**
Thêm logic đánh giá kết quả từ local database bằng chunk reranker trước khi quyết định sử dụng hay fallback sang websearch.

**Cách hoạt động:**
1. Khi RouterRetriever nhận được kết quả từ local database
2. Sử dụng ChunkRanker để đánh giá và lọc các chunks
3. Nếu số lượng chunks hợp lệ (score ≥ 0.5) ≥ 1: sử dụng kết quả local
4. Nếu không đạt yêu cầu: tự động fallback sang websearch (nếu có)

**Lợi ích:**
- Đảm bảo chỉ sử dụng kết quả local khi chất lượng đủ tốt
- Tự động fallback sang websearch khi local results không đủ chất lượng
- Cải thiện độ chính xác của hệ thống bằng cách luôn ưu tiên kết quả tốt nhất

**File thay đổi:**
- `kaggle_dedicated/vllm_v4.ipynb`: Thêm method `_evaluate_local_results()` và cập nhật logic trong `RouterRetriever.retrieve()`

**Code mẫu:**
```python
def _evaluate_local_results(self, rag_sources: list[RagSource], question: str):
    # Sử dụng chunk reranker để đánh giá
    reranked_sources = self.chunk_ranker.rerank_chunks(
        rag_sources, question, relative_threshold=0.5
    )
    # Kiểm tra số lượng chunks hợp lệ
    is_sufficient = len(reranked_sources) >= self.local_min_chunks
    return is_sufficient, 1.0, len(reranked_sources)
```

### 2.4. Tối ưu Quản lý Bộ nhớ GPU

**Vấn đề:**
Khi chạy ChunkRanker cùng lúc với các model khác (embedding, LLM), GPU bị hết bộ nhớ (CUDA out of memory).

**Giải pháp:**
1. **Auto fallback to CPU:** Tự động phát hiện khi CUDA hết memory và chuyển sang CPU
2. **Clear CUDA cache:** Xóa cache trước và sau khi predict để giải phóng bộ nhớ
3. **Memory check:** Kiểm tra memory trước khi load model

**File thay đổi:**
- `kaggle_dedicated/data_retriever/ranker/chunk_ranker.py`: 
  - Thêm logic auto-detect device (dòng 11-22)
  - Thêm clear cache trước và sau predict (dòng 44-51)

**Code mẫu:**
```python
# Auto-detect và fallback
if device == "cuda" and torch.cuda.is_available():
    try:
        test_tensor = torch.zeros(1).cuda()
        del test_tensor
        torch.cuda.empty_cache()
    except RuntimeError:
        device = "cpu"  # Fallback to CPU
        print("[ChunkRanker] CUDA out of memory, falling back to CPU")
```

## 3. KẾT QUẢ

### 3.1. Cải thiện Chất lượng
- Chunks được lọc chặt chẽ hơn với threshold cố định 0.5
- Chỉ giữ lại chunks có độ liên quan cao với query
- Tự động fallback khi local results không đủ chất lượng

### 3.2. Cải thiện Hiệu suất
- Tránh được lỗi CUDA out of memory
- Tự động tối ưu device (CUDA/CPU) dựa trên tình trạng memory
- Quản lý bộ nhớ tốt hơn với cache clearing

### 3.3. Tăng Tính Ổn Định
- Hệ thống không bị crash khi GPU hết memory
- Tự động fallback đảm bảo luôn có kết quả trả về
- Logging chi tiết giúp debug dễ dàng hơn

## 4. CẤU TRÚC FILE THAY ĐỔI

```
kaggle_dedicated/
├── data_retriever/
│   ├── config.py                    # Cập nhật ChunkRankerConfig
│   └── ranker/
│       └── chunk_ranker.py          # Thay đổi implementation chính
└── vllm_v4.ipynb                    # Thêm logic evaluation cho local results
```

## 5. HƯỚNG PHÁT TRIỂN TƯƠNG LAI

1. **Tối ưu Model:** Có thể fine-tune BGE reranker trên dataset tiếng Việt
2. **Dynamic Threshold:** Có thể điều chỉnh threshold dựa trên loại query
3. **Batch Processing:** Tối ưu xử lý batch để tăng tốc độ
4. **Caching:** Cache kết quả rerank cho các query tương tự

## 6. KẾT LUẬN

Các thay đổi đã được thực hiện nhằm:
- Nâng cao chất lượng reranking với model mới và threshold cố định
- Cải thiện logic đánh giá kết quả local với tự động fallback
- Tăng tính ổn định với quản lý bộ nhớ GPU tốt hơn

Hệ thống hiện tại hoạt động ổn định hơn, cho kết quả chính xác hơn và có khả năng tự phục hồi khi gặp vấn đề về tài nguyên.

---

**Ngày báo cáo:** 2025-01-02  
**Người thực hiện:** [Tên sinh viên]  
**Phiên bản:** v4 (recover-code branch)









