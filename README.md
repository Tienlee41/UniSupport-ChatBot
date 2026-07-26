# UniAdmission-ChatBot

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-green.svg)
![Qwen](https://img.shields.io/badge/Qwen3--4B-LLM-orange.svg)
![vLLM](https://img.shields.io/badge/vLLM-inference-red.svg)

UniAdmission-ChatBot là hệ thống chatbot hỗ trợ tư vấn tuyển sinh đại học tại Việt Nam. Hệ thống kết hợp mô hình ngôn ngữ lớn đã tinh chỉnh, cơ sở dữ liệu vector, tìm kiếm Web và cơ chế định tuyến truy vấn nhằm cung cấp câu trả lời có căn cứ, phù hợp với các câu hỏi về trường, ngành, điểm chuẩn, học phí và thông tin tuyển sinh.


---

## 1. Tổng quan

Hệ thống được thiết kế để hỗ trợ học sinh, phụ huynh và sinh viên tra cứu thông tin tuyển sinh bằng tiếng Việt tự nhiên.

Thay vì chỉ dựa vào tri thức nội tại của mô hình ngôn ngữ, hệ thống sử dụng kiến trúc truy xuất tăng cường sinh, kết hợp:

* **Qwen3-4B đã tinh chỉnh** cho miền tư vấn tuyển sinh.
* **Vector Database Search** cho dữ liệu đã chuẩn hóa.
* **Web Search** cho thông tin mới hoặc chưa có trong cơ sở dữ liệu nội bộ.
* **Smart Search Routing** để lựa chọn nguồn dữ liệu phù hợp với từng câu hỏi.
* **vLLM** để phục vụ suy luận và sinh phản hồi dạng streaming.

---

## 2. Tính năng chính

* Trả lời câu hỏi tuyển sinh bằng tiếng Việt tự nhiên.
* Truy xuất thông tin từ cơ sở dữ liệu của hơn 200 trường đại học.
* Hỗ trợ các nhóm thông tin: thông tin chung, điểm chuẩn, học phí, tuyển sinh.
* Tự động định tuyến câu hỏi sang Local Search, Web Search hoặc Hybrid Search.
* Sinh câu trả lời dạng streaming.
* Hiển thị nguồn tham khảo để người dùng kiểm chứng.
* Lưu lịch sử hội thoại và quản lý phiên làm việc.
* Triển khai mô hình Qwen3-4B thông qua vLLM.

---

## 3. Kiến trúc hệ thống

Hệ thống gồm các thành phần chính:

```text
User
  ↓
Web Chat UI
  ↓
FastAPI Backend
  ↓
Smart Query Router
  ├── Local Search: FAISS Vector Database
  ├── Web Search: Search API + Crawl + Rerank
  └── Hybrid Search: kết hợp Local và Web
  ↓
RAG Context Builder
  ↓
Qwen3-4B + LoRA Adapter on vLLM
  ↓
Streaming Answer + Sources
```

### Frontend

* Giao diện hội thoại Web.
* Hỗ trợ đăng nhập, đăng ký và quản lý phiên chat.
* Hiển thị phản hồi dạng streaming và nguồn tham khảo.

### Backend

* Xây dựng bằng FastAPI.
* Quản lý xác thực, phiên hội thoại và lịch sử tin nhắn.
* Điều phối truy vấn đến các pipeline tìm kiếm và mô hình.

### AI Pipeline

* Qwen3-4B được tinh chỉnh bằng LoRA.
* FAISS Vector Database dùng embedding `intfloat/multilingual-e5-small`.
* Web Search dùng Brave Search API hoặc Google Custom Search.
* vLLM phục vụ inference tốc độ cao.

---

## 4. Dữ liệu

Dữ liệu cục bộ được tổ chức theo từng trường đại học và từng nhóm thông tin.

```text
vectordb/
├── university_1/
│   ├── thong_tin_chung
│   ├── hoc_phi
│   ├── diem_chuan
│   └── tuyen_sinh
├── university_2/
│   ├── thong_tin_chung
│   ├── hoc_phi
│   ├── diem_chuan
│   └── tuyen_sinh
└── ...
```

Mỗi tài liệu được gắn metadata như:

* `school_id`: định danh trường.
* `section`: nhóm thông tin.
* `source`: nguồn dữ liệu.
* `content`: nội dung văn bản.

Cách tổ chức này giúp hệ thống truy xuất nhanh theo cả metadata và ngữ nghĩa.

---

## 5. Cài đặt

### Yêu cầu

* Python 3.9+
* pip hoặc conda
* GPU nếu chạy mô hình cục bộ
* Các API key cần thiết cho Web Search và mô hình

### Clone repository

```bash
git clone https://github.com/photienanh/UniAdmission-ChatBot.git
cd UniAdmission-ChatBot
```

### Tạo môi trường ảo

Windows:

```bash
python -m venv venv
venv\Scripts\activate.bat
```

Linux/MacOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

## 6. Cấu hình môi trường

Tạo file `server.env` trong thư mục `app/`:

```env
NGROK_TOKEN=your_ngrok_token
JWT_SECRET_KEY=your_jwt_secret_key
```

Tạo file `worker.env` trong thư mục `app/`:

```env
HUGGING_FACE_TOKEN=your_hugging_face_token
GEMINI_API_KEY=your_gemini_api_key
BRAVE_SEARCH_API_KEY=your_brave_api_key
OPENAI_API_KEY=your_openai_api_key
GOOGLE_SEARCH_API_KEY=your_google_api_key
GOOGLE_SEARCH_CX=your_google_search_cx
SCRAPINGBEE_API_KEY=your_scrapingbee_api_key
NGROK_TOKEN=your_ngrok_token
NGROK_TOKEN_1=your_ngrok_token_1
```

---

## 7. Chạy ứng dụng

### Development

```bash
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Ứng dụng chạy tại:

```text
http://localhost:8000
```

---

## 8. Triển khai mô hình

### Chạy API model trên local

```bash
cd kaggle_dedicated
python api_v3.py
```

Hoặc sử dụng notebook:

```text
kaggle_dedicated/api_v3.ipynb
```

### Triển khai trên Kaggle

Dự án hỗ trợ triển khai Qwen3-4B với vLLM trên Kaggle Notebook.

Các bước chính:

1. Upload notebook `kaggle_dedicated/vllm_single_v3.ipynb`.
2. Cấu hình biến `DOMAIN` trỏ đến server backend.
3. Chạy notebook để:

   * tải mô hình và LoRA adapter;
   * khởi động vLLM inference server;
   * tạo ngrok tunnel;
   * kết nối với FastAPI backend.

---

## 9. Cập nhật Vector Database

Quy trình cập nhật dữ liệu gồm crawl, chuẩn hóa, tạo mapping và xây dựng lại FAISS index.

```bash
cd vector_database/crawl

python selenium_crawler.py
python re_format.py
python create_mapping.py
```

Sau đó rebuild vector database:

```bash
cd ..
python create_vector_db.py
```

---

## 10. Cấu trúc thư mục

```text
UniSupport-ChatBot/
├── app/
│   ├── frontend/              # Giao diện web, template, static assets
│   ├── package/               # Model/adapter đóng gói cho backend
│   ├── main.py                # FastAPI application
│   ├── server.env             # Cấu hình backend
│   └── worker.env             # Cấu hình worker/model/web search
├── kaggle_dedicated/
│   ├── data_retriever/        # Pipeline RAG: search, crawl, chunk, rerank
│   ├── instruction/           # Prompt/router/decomposer/reader instructions
│   ├── lora/                  # LoRA adapters dùng khi triển khai
│   ├── old_versions/          # Notebook và phiên bản cũ
│   ├── api_v3.py              # Worker API phục vụ suy luận
│   └── vllm_v4.ipynb          # Notebook triển khai vLLM/Kaggle
├── vector_database/
│   ├── crawl/                 # Crawler, chuẩn hóa dữ liệu trường đại học
│   └── create_vector_db.py    # Xây dựng FAISS vector database
├── finetune/                  # Dữ liệu và tài liệu phục vụ huấn luyện LoRA/RAFT
├── eval/                      # Notebook/script đánh giá thử nghiệm
├── validation_data/           # Dữ liệu kiểm thử/đối chiếu
├── logtest/                   # Log chạy thử pipeline
├── files/                     # Tệp phụ trợ hoặc dữ liệu đầu vào
├── test_qe/                   # Tệp kiểm thử query expansion/retrieval
├── README.md                  # Tài liệu hướng dẫn dự án
├── requirements.txt           # Danh sách thư viện Python
└── ngrok.py                   # Script hỗ trợ tạo tunnel
```

---

## 11. Ghi chú

Dự án phục vụ mục tiêu nghiên cứu và thử nghiệm trong bài toán tư vấn tuyển sinh. Thông tin do hệ thống sinh ra cần được kiểm chứng lại với nguồn chính thức của các trường đại học hoặc cơ quan quản lý giáo dục khi sử dụng cho các quyết định quan trọng.
