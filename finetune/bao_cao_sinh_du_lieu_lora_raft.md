# Báo cáo cách thức sinh dữ liệu LoRA + RAFT

## 1. Mục tiêu

Bộ dữ liệu được xây dựng để huấn luyện và đánh giá mô hình tư vấn tuyển sinh theo hai hướng:

- **LoRA/SFT reader**: mô hình đọc tài liệu tham khảo trong prompt và sinh câu trả lời có căn cứ.
- **RAFT/RAG training**: mô hình học cách trả lời từ oracle documents, đồng thời chống nhiễu từ distractor documents.

Trọng tâm của bộ dữ liệu là các lỗi hệ thống từng gặp trong pipeline RAG/multi-hop:

- nhầm **điểm sàn** với **điểm chuẩn**;
- trả lời khi tài liệu không đủ dữ liệu;
- xếp hạng top trường nhưng thiếu điểm chuẩn/học phí;
- lọc sai điều kiện điểm, học phí, khu vực, ngành;
- bị nhiễu bởi tài liệu không đúng ngữ cảnh như du học, cao học, tin tuyển dụng.

## 2. Nguồn đầu vào

Hai tệp Markdown được dùng làm khung ý định câu hỏi:

| Tệp | Vai trò |
|---|---|
| `danh_sach_cau_hoi_finetune.md` | Danh sách template câu hỏi tuyển sinh, học phí, học vụ, ngành đào tạo, so sánh, tư vấn |
| `thong_tin_cong_khai_kiem_dinh.md` | Template câu hỏi về thông tin công khai, cơ sở vật chất, đội ngũ, kiểm định, báo cáo chất lượng |

Script đọc các dòng dạng câu hỏi trong hai tệp này, sau đó thiết kế các generator tương ứng thay vì chỉ thay thế placeholder một cách cơ học. Cách này giúp sinh được cả câu hỏi đơn và các câu hỏi giống sub-question trong multi-hop.

## 3. Script sinh dữ liệu

Script chính:

```powershell
python finetune\build_lora_raft_dataset.py
```

Các file đầu ra:

| File | Mục đích |
|---|---|
| `finetune/lora_raft_dataset_5000.jsonl` | File đầy đủ, chứa cả QA, documents, messages, metadata |
| `finetune/lora_messages_5000.jsonl` | File dùng cho SFT/LoRA dạng chat messages |
| `finetune/raft_qa_documents_5000.jsonl` | File dùng cho RAFT/RAG, có oracle và distractor document ids |
| `finetune/lora_raft_dataset_5000_summary.json` | Thống kê tổng quan |

Script chạy deterministic theo seed mặc định `20260521`, nên có thể sinh lại cùng một bộ dữ liệu.

## 4. Kho tri thức synthetic

Script định nghĩa một kho tri thức tổng hợp gồm:

- danh sách trường đại học;
- tên viết tắt;
- loại hình trường;
- cơ quan chủ quản;
- địa chỉ;
- website;
- số lượng sinh viên, giảng viên;
- nhóm ngành đào tạo hợp lý theo từng trường;
- danh sách ngành, mã ngành, nhóm ngành, số tín chỉ, thời gian đào tạo, vị trí việc làm.

Dữ liệu này **không được xem là dữ liệu thật**. Nó được dùng để tạo tài liệu giả lập có cấu trúc, nhằm huấn luyện hành vi đọc nguồn, lọc điều kiện và từ chối khi thiếu bằng chứng.

## 5. Nhóm generator dữ liệu

Bộ sinh dữ liệu gồm 8 nhóm chính:

| Nhóm | Số mẫu | Nội dung |
|---|---:|---|
| `tuyen_sinh_diem` | 750 | Điểm chuẩn, điểm sàn, điểm trúng tuyển, tăng giảm theo năm |
| `tuyen_sinh_chi_tieu_phuong_thuc` | 450 | Chỉ tiêu, phương thức xét tuyển, tổ hợp xét tuyển, mã ngành |
| `hoc_phi_tai_chinh` | 600 | Học phí ngành/trường/chương trình, so sánh học phí |
| `hoc_vu_sinh_vien` | 550 | Đăng ký học phần, học lại, học phí kỳ, nợ học phí |
| `cong_khai_kiem_dinh` | 800 | Thông tin pháp lý, cơ sở vật chất, đội ngũ, kiểm định |
| `nganh_dao_tao_nghe_nghiep` | 650 | Chương trình đào tạo, số tín chỉ, chuẩn đầu ra, việc làm |
| `so_sanh_tu_van_xep_hang` | 950 | So sánh, ranking top 5, lọc theo điểm/học phí/khu vực |
| `khong_du_du_lieu_guardrail` | 250 | Mẫu không đủ dữ liệu, buộc model không suy đoán |

Tổng số mẫu: **5000**.

## 6. Cách sinh documents

Mỗi mẫu đều có một hoặc nhiều tài liệu tham khảo. Có hai loại tài liệu:

### 6.1. Oracle documents

Oracle documents là tài liệu chứa đủ hoặc gần đủ bằng chứng để trả lời. Ví dụ:

- tài liệu điểm chuẩn tuyển sinh;
- tài liệu học phí;
- tài liệu chương trình đào tạo;
- tài liệu công khai kiểm định;
- bảng tổng hợp điểm chuẩn và học phí dùng cho ranking;
- thông báo học vụ.

Trong file RAFT, các tài liệu này được đánh dấu:

```json
"oracle_document_ids": ["D5120_R"]
```

và trong từng document:

```json
"is_relevant": true
```

### 6.2. Distractor documents

Distractor documents là tài liệu nhiễu, được thiết kế để kiểm tra khả năng bỏ qua nguồn không phù hợp. Ví dụ:

- bản tin du học Mỹ;
- tài liệu chỉ có điểm sàn nhưng câu hỏi hỏi điểm chuẩn;
- tin tuyển dụng;
- thông tin cao học;
- hoạt động đoàn/học vụ không liên quan.

Trong file RAFT:

```json
"distractor_document_ids": ["D5120_N1"]
```

và trong từng document:

```json
"is_relevant": false
```

Sau lần chỉnh cuối, **mỗi mẫu đều có ít nhất một oracle document và ít nhất một distractor document**.

## 7. Cách sinh câu trả lời

Câu trả lời được sinh bằng rule/template có kiểm soát, không gọi API bên ngoài. Mục tiêu là đảm bảo:

- câu trả lời bám đúng tài liệu;
- có bảng khi dữ liệu có cấu trúc;
- nêu rõ nguồn bằng `doc_id`;
- không dùng điểm sàn thay cho điểm chuẩn;
- không bịa học phí nếu tài liệu không có học phí;
- với câu hỏi lọc/ranking, chỉ giữ các trường thỏa toàn bộ điều kiện.

Ví dụ nhóm ranking:

- câu hỏi yêu cầu top 5 điểm chuẩn ngành Công nghệ thông tin;
- oracle document chứa bảng gồm trường, điểm chuẩn, học phí;
- answer sắp xếp giảm dần theo điểm chuẩn;
- lưu ý rõ: không dùng điểm sàn để xếp hạng.

Ví dụ nhóm không đủ dữ liệu:

- câu hỏi hỏi học phí;
- document chỉ có điểm sàn;
- answer trả lời “không đủ dữ liệu”, nêu rõ tài liệu không có bảng học phí.

## 8. Schema dữ liệu

### 8.1. File đầy đủ

Mỗi dòng trong `lora_raft_dataset_5000.jsonl` có dạng:

```json
{
  "id": "lora_raft_00001",
  "split": "train",
  "category": "tuyen_sinh_diem",
  "task_type": "qa_numeric",
  "question": "...",
  "answer": "...",
  "documents": [
    {
      "doc_id": "D1_A",
      "title": "...",
      "source": "synthetic://...",
      "content": "...",
      "is_relevant": true
    }
  ],
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "school": "UET",
    "major": "Công nghệ thông tin",
    "year": 2025,
    "synthetic": true
  }
}
```

### 8.2. File LoRA

`lora_messages_5000.jsonl` chỉ giữ phần cần cho SFT:

```json
{
  "id": "...",
  "split": "train",
  "messages": [...],
  "metadata": {...}
}
```

User message đã chứa toàn bộ tài liệu tham khảo, nên model học trực tiếp cách đọc context rồi trả lời.

### 8.3. File RAFT

`raft_qa_documents_5000.jsonl` có dạng:

```json
{
  "id": "...",
  "split": "train",
  "category": "...",
  "task_type": "...",
  "question": "...",
  "answer": "...",
  "documents": [...],
  "oracle_document_ids": [...],
  "distractor_document_ids": [...],
  "metadata": {...}
}
```

File này phù hợp để huấn luyện/evaluate pipeline retrieve-then-read.

## 9. Phân bổ dữ liệu

Split:

| Split | Số mẫu |
|---|---:|
| train | 4500 |
| validation | 250 |
| test | 250 |

Phân bổ theo task type:

| Task type | Số mẫu |
|---|---:|
| `qa_fact` | 1900 |
| `qa_numeric` | 1350 |
| `procedural` | 550 |
| `filter_recommendation` | 475 |
| `comparison` | 355 |
| `no_answer` | 250 |
| `ranking` | 120 |

## 10. Kiểm tra chất lượng đã thực hiện

Các kiểm tra đã chạy:

- tổng số dòng mỗi file JSONL là `5000`;
- không có câu hỏi trùng;
- không còn placeholder `[Tên trường]`, `[Năm]`;
- mỗi mẫu có ít nhất một document liên quan;
- mỗi mẫu RAFT có oracle document;
- mỗi mẫu RAFT có distractor document;
- split đúng `4500/250/250`;
- nhóm ranking có các ca trọng tâm như Công nghệ thông tin, Kỹ thuật phần mềm, Khoa học máy tính, Trí tuệ nhân tạo, An toàn thông tin.

Một số thống kê audit:

```text
total = 5000
train = 4500
validation = 250
test = 250
unique_questions = 5000
no_oracle = 0
no_distractor = 0
unique_first3 = 174
unique_first1 = 30
```

## 11. Điểm mạnh

Bộ dữ liệu hiện có các ưu điểm:

- phủ nhiều nhóm câu hỏi tuyển sinh, học vụ, công khai, kiểm định;
- có cả câu hỏi đơn và câu hỏi dạng multi-hop;
- có tài liệu oracle/distractor rõ ràng;
- có mẫu guardrail chống hallucination;
- nhấn mạnh các lỗi thực tế của hệ thống RAG như nhầm điểm sàn/điểm chuẩn;
- có thể dùng ngay cho LoRA và RAFT.

## 12. Giới hạn

Bộ dữ liệu chưa nên xem là hoàn hảo vì:

- dữ liệu là synthetic, không phải dữ liệu thật từ website trường;
- văn phong vẫn dựa nhiều trên template;
- chưa có nhiều lỗi gõ, câu không dấu, câu hỏi rất ngắn, câu hỏi mơ hồ;
- chưa phủ tuyệt đối mọi template trong hai file nguồn;
- số mẫu ranking hiện là 120, đủ để học hành vi nhưng có thể tăng nếu muốn tối ưu riêng lỗi top-N;
- câu hỏi multi-hop phức tạp chưa có đầy đủ DAG/sub-question trace như pipeline thật.

## 13. Đề xuất cải tiến tiếp theo

Nếu muốn dùng bộ này để cải thiện rõ rệt hệ thống production, nên bổ sung thêm:

1. **Paraphrase/noise layer**: tạo thêm biến thể câu hỏi không dấu, viết tắt, sai chính tả nhẹ, câu hỏi cụt.
2. **Real-data grounding**: trộn thêm dữ liệu thật từ các thông báo điểm chuẩn/học phí chính thức.
3. **Multi-hop trace format**: thêm trường `sub_questions`, `dependencies`, `expected_intermediate_answers`.
4. **Hard negative documents**: tạo distractor giống oracle hơn, ví dụ tài liệu điểm sàn cùng trường/cùng ngành nhưng sai metric.
5. **Coverage validator**: với ranking top 5, kiểm tra đủ 5 item có `score`, `tuition`, `major`, `year`, `method`.
6. **Tăng nhóm ranking/filter**: đặc biệt cho Công nghệ thông tin, Kỹ thuật phần mềm, An toàn thông tin, Khoa học máy tính.

## 14. Kết luận

Bộ dữ liệu hiện tại đáp ứng tốt mục tiêu huấn luyện ban đầu cho LoRA + RAFT: có 5000 mẫu, có tài liệu oracle/distractor, có nhiều nhóm câu hỏi đơn và câu hỏi dạng multi-hop, có guardrail chống trả lời khi thiếu dữ liệu.

Tuy nhiên, vì đây là synthetic dataset, bước tiếp theo nên là trộn thêm dữ liệu thật và sinh thêm paraphrase/noise để mô hình bền hơn với câu hỏi tự nhiên của người dùng.

---

# Phụ lục: Lý thuyết chi tiết về cách tạo dữ liệu

## A. Bản chất của dữ liệu huấn luyện cho hệ tư vấn tuyển sinh

Một chatbot tư vấn tuyển sinh không chỉ cần trả lời đúng một câu hỏi đơn lẻ. Nó phải thực hiện nhiều năng lực cùng lúc:

- hiểu ý định câu hỏi;
- nhận diện trường, ngành, năm, phương thức xét tuyển, chương trình đào tạo;
- tìm đúng tài liệu;
- phân biệt các khái niệm gần nhau như điểm chuẩn, điểm sàn, điểm trúng tuyển;
- tổng hợp dữ liệu nhiều nguồn;
- từ chối trả lời khi thiếu bằng chứng;
- trình bày câu trả lời dễ đọc, có bảng và nguồn.

Vì vậy dữ liệu huấn luyện không nên chỉ là cặp `question -> answer`. Với bài toán RAG/RAFT, mỗi mẫu nên có cấu trúc:

```text
question
documents
oracle document labels
distractor document labels
answer grounded on documents
metadata
```

Cách tổ chức này giúp mô hình học hành vi bám nguồn thay vì ghi nhớ hoặc suy đoán.

## B. Khác biệt giữa dữ liệu LoRA/SFT và dữ liệu RAFT

## B.1. Dữ liệu LoRA/SFT

LoRA/SFT thường huấn luyện mô hình theo định dạng hội thoại:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Mục tiêu là dạy model:

- phong cách trả lời;
- format trả lời;
- cách dùng bảng;
- cách nêu nguồn;
- cách nói rõ khi thiếu dữ liệu;
- cách trả lời ngắn/dài theo intent.

Nếu user message có chứa tài liệu tham khảo, model còn học được cách đọc context và trích xuất dữ liệu.

## B.2. Dữ liệu RAFT

RAFT là hướng huấn luyện model trong bối cảnh có retrieval. Một mẫu RAFT thường có:

```json
{
  "question": "...",
  "documents": [...],
  "oracle_document_ids": [...],
  "distractor_document_ids": [...],
  "answer": "..."
}
```

Mục tiêu là dạy model:

- đọc nhiều tài liệu;
- chọn tài liệu đúng;
- bỏ qua tài liệu nhiễu;
- trả lời dựa trên oracle documents;
- không bị distractor kéo sang câu trả lời sai.

Với hệ thống tuyển sinh, RAFT đặc biệt quan trọng vì web search thường trả về nguồn gần đúng nhưng sai metric. Ví dụ câu hỏi cần điểm chuẩn nhưng nguồn lại là điểm sàn.

## C. Nguyên tắc thiết kế một mẫu dữ liệu tốt

Một mẫu tốt cần thỏa các điều kiện sau:

1. **Intent rõ ràng**: câu hỏi phải xác định được người dùng đang hỏi điểm chuẩn, học phí, chỉ tiêu, học vụ, kiểm định hay tư vấn.
2. **Entity rõ ràng**: nếu có trường/ngành/năm/phương thức thì phải xuất hiện đủ trong câu hỏi hoặc tài liệu.
3. **Document đủ bằng chứng**: answer phải lấy được từ document.
4. **Không suy đoán**: nếu document thiếu trường thông tin, answer phải nói thiếu.
5. **Có nhiễu hợp lý**: distractor phải đủ giống để model học cách phân biệt.
6. **Có metadata**: lưu `category`, `task_type`, `school`, `major`, `year` để audit và chia tập.
7. **Không trộn khái niệm**: điểm sàn không được dùng thay cho điểm chuẩn; học phí không được suy ra từ điểm chuẩn.
8. **Có kiểm tra sau sinh**: kiểm tra duplicate, placeholder, số lượng oracle/distractor, split.

## D. Xây taxonomy câu hỏi

Taxonomy là hệ phân loại câu hỏi. Đây là bước quan trọng nhất trước khi sinh dữ liệu.

Với chatbot tuyển sinh, taxonomy nên có tối thiểu các nhóm:

| Nhóm | Ví dụ |
|---|---|
| Điểm chuẩn/điểm sàn | Điểm chuẩn ngành CNTT UET 2025 là bao nhiêu? |
| Chỉ tiêu | Chỉ tiêu ngành Kỹ thuật phần mềm PTIT 2025? |
| Phương thức xét tuyển | UET có xét tuyển bằng ĐGNL không? |
| Tổ hợp xét tuyển | Ngành CNTT xét tổ hợp nào? |
| Học phí | Học phí ngành CNTT HUST 2025? |
| Học vụ | Đăng ký học lại học kỳ hè như thế nào? |
| Thông tin công khai | Trường có bao nhiêu giảng viên cơ hữu? |
| Kiểm định | Ngành này đã được kiểm định chưa? |
| Chương trình đào tạo | Ngành này học bao nhiêu tín chỉ? |
| Cơ hội việc làm | Ra trường làm vị trí gì? |
| So sánh | So sánh UET và PTIT ngành CNTT |
| Ranking | Top 5 trường điểm chuẩn CNTT cao nhất |
| Tư vấn lọc điều kiện | 26.5 điểm, học phí dưới 50 triệu nên chọn trường nào? |
| Không đủ dữ liệu | Tài liệu không có học phí nhưng câu hỏi hỏi học phí |

Taxonomy tốt giúp tránh dataset lệch quá nhiều vào một nhóm câu hỏi.

## E. Thiết kế câu hỏi đơn

Câu hỏi đơn là câu hỏi có thể trả lời từ một document hoặc một bảng dữ liệu.

Ví dụ:

```text
Điểm chuẩn ngành Công nghệ thông tin trường UET năm 2025 là bao nhiêu?
```

Các biến cần thay:

- trường;
- ngành;
- năm;
- phương thức;
- chương trình;
- tổ hợp;
- loại dữ liệu cần hỏi.

Câu hỏi đơn nên có nhiều dạng diễn đạt:

```text
Điểm chuẩn ngành CNTT UET 2025?
Ngành CNTT của UET năm 2025 lấy bao nhiêu điểm?
Điểm trúng tuyển CNTT UET theo THPT là bao nhiêu?
Năm 2025 UET công bố điểm chuẩn CNTT thế nào?
```

Nếu chỉ dùng một văn phong, model sẽ học thuộc cấu trúc câu thay vì học intent.

## F. Thiết kế câu hỏi giống sub-question trong multi-hop

Trong pipeline multi-hop, câu hỏi lớn thường bị tách thành các câu hỏi nhỏ.

Ví dụ câu hỏi lớn:

```text
Xếp hạng 5 trường có điểm chuẩn ngành Công nghệ thông tin cao nhất năm 2025 kèm học phí.
```

Các sub-question có thể là:

```text
Danh sách các trường có điểm chuẩn ngành Công nghệ thông tin năm 2025?
Điểm chuẩn ngành Công nghệ thông tin của từng trường là bao nhiêu?
Học phí ngành Công nghệ thông tin của từng trường là bao nhiêu?
Xếp hạng các trường theo điểm chuẩn giảm dần.
```

Dataset nên chứa cả:

- câu hỏi lớn cuối cùng;
- câu hỏi đơn tương ứng với từng sub-question;
- câu hỏi reasoning để tổng hợp kết quả.

Lý do: nếu chỉ huấn luyện câu hỏi lớn, model không học tốt các bước trung gian. Nếu chỉ huấn luyện câu hỏi nhỏ, model không học tổng hợp.

## G. Thiết kế oracle documents

Oracle document là tài liệu chứa bằng chứng đúng.

Một oracle document tốt cần:

- chứa đúng trường/ngành/năm được hỏi;
- chứa đúng metric được hỏi;
- có dữ liệu ở dạng có thể trích xuất;
- không chứa quá nhiều thông tin thừa;
- có `doc_id` để answer dẫn nguồn.

Ví dụ oracle cho điểm chuẩn:

```markdown
# Điểm chuẩn tuyển sinh đại học chính quy 2025 - UET

| Mã ngành | Ngành | Tổ hợp | Điểm sàn | Điểm chuẩn |
|---|---|---|---:|---:|
| 7480201 | Công nghệ thông tin | A00, A01 | 23.0 | 27.8 |
```

Answer đúng phải lấy `27.8` từ cột `Điểm chuẩn`, không lấy `23.0` từ cột `Điểm sàn`.

## H. Thiết kế distractor documents

Distractor document là tài liệu nhiễu. Đây là phần rất quan trọng với RAFT.

Distractor không nên quá dễ. Nếu distractor hoàn toàn không liên quan, model không học được nhiều. Distractor tốt nên giống oracle ở một vài mặt nhưng sai ở điểm cốt lõi.

Các loại distractor nên có:

| Loại nhiễu | Ví dụ |
|---|---|
| Sai metric | Hỏi điểm chuẩn nhưng document là điểm sàn |
| Sai năm | Hỏi 2025 nhưng document là 2024 |
| Sai ngành | Hỏi CNTT nhưng document là Kinh tế |
| Sai bậc đào tạo | Hỏi đại học nhưng document là cao học |
| Sai quốc gia/ngữ cảnh | Hỏi tuyển sinh Việt Nam nhưng document du học Mỹ |
| Sai trường | Hỏi UET nhưng document PTIT |
| Gần đúng nhưng thiếu dữ liệu | Có điểm chuẩn nhưng không có học phí |

Ví dụ:

```markdown
# Ngưỡng đảm bảo chất lượng đầu vào 2025 - UET
Tài liệu này chỉ công bố điểm sàn/điểm nhận hồ sơ, chưa công bố điểm chuẩn.
```

Nếu câu hỏi hỏi điểm chuẩn, document này phải bị bỏ qua hoặc chỉ dùng để nói rằng nó không đủ dữ liệu.

## I. Thiết kế câu trả lời grounded

Answer nên được tạo theo nguyên tắc:

```text
Answer = dữ liệu trong oracle document + diễn giải ngắn + nguồn + lưu ý
```

Không nên tạo answer bằng kiến thức ngoài document.

Với câu hỏi số liệu, answer nên có bảng:

```markdown
| Trường | Ngành | Điểm chuẩn | Học phí |
|---|---|---:|---:|
| UET | Công nghệ thông tin | 27.8 | 35 triệu/năm |
```

Với câu hỏi thủ tục, answer nên có bước:

```markdown
1. Đăng nhập cổng sinh viên.
2. Chọn học phần.
3. Xác nhận đăng ký.
4. Nộp học phí trước hạn.
```

Với câu hỏi không đủ dữ liệu, answer phải nói rõ:

```text
Tài liệu được cung cấp không có học phí ngành này, nên chưa đủ căn cứ để trả lời.
```

## J. Mẫu không đủ dữ liệu

Mẫu no-answer là bắt buộc trong dataset RAG.

Nếu không có mẫu này, model sẽ học rằng câu hỏi nào cũng phải trả lời, dẫn đến hallucination.

Các tình huống no-answer nên sinh:

- document chỉ có điểm sàn, câu hỏi hỏi điểm chuẩn;
- document có điểm chuẩn nhưng không có học phí;
- document có học phí chung nhưng không có học phí ngành;
- document có thông tin tuyển sinh nhưng không có kiểm định;
- document có thông tin trường nhưng không có khảo sát việc làm;
- document chỉ có một trường nhưng câu hỏi yêu cầu top 5.

Answer trong nhóm này phải nhất quán:

```text
Không đủ dữ liệu để trả lời.
Tài liệu hiện có không chứa ...
Cần bổ sung ...
```

## K. Sinh dữ liệu ranking và filter

Ranking/filter là nhóm dễ lỗi nhất trong chatbot tuyển sinh.

## K.1. Ranking

Câu hỏi:

```text
Xếp hạng 5 trường có điểm chuẩn ngành Công nghệ thông tin cao nhất năm 2025 kèm học phí.
```

Điều kiện để answer hợp lệ:

- có ít nhất 5 trường;
- mỗi trường có điểm chuẩn;
- mỗi trường có học phí;
- cùng ngành;
- cùng năm;
- cùng phương thức hoặc ghi rõ phương thức;
- sắp xếp theo điểm chuẩn giảm dần;
- không lấy điểm sàn thay điểm chuẩn.

Nếu thiếu một trong các điều kiện trên, nên trả lời thiếu dữ liệu hoặc chỉ trả lời partial có cảnh báo.

## K.2. Filter recommendation

Câu hỏi:

```text
Với 26.5 điểm A00, muốn học CNTT ở Hà Nội, học phí dưới 50 triệu thì nên chọn trường nào?
```

Đây là bài toán lọc nhiều điều kiện:

```text
major = Công nghệ thông tin
city = Hà Nội
score <= 26.5
tuition < 50 triệu
block includes A00
```

Model phải học rằng tất cả điều kiện đều bắt buộc. Không được đưa trường vượt học phí hoặc sai ngành vào danh sách.

## L. Đa dạng hóa văn phong câu hỏi

Đa dạng văn phong giúp model không phụ thuộc vào template.

Các hướng đa dạng hóa:

1. Thay đổi từ khóa:
   - điểm chuẩn;
   - điểm trúng tuyển;
   - lấy bao nhiêu điểm;
   - mức điểm vào ngành;
   - ngưỡng trúng tuyển.

2. Thay đổi thứ tự:
   - `Điểm chuẩn ngành CNTT UET 2025?`
   - `UET năm 2025 ngành CNTT lấy bao nhiêu điểm?`

3. Thay đổi mức trang trọng:
   - `Cho em hỏi...`
   - `Mình muốn biết...`
   - `Tư vấn giúp tôi...`

4. Thêm viết tắt:
   - CNTT;
   - KTPM;
   - KHMT;
   - ATTT;
   - ĐGNL;
   - THPT.

5. Thêm lỗi tự nhiên:
   - không dấu;
   - sai chính tả nhẹ;
   - thiếu dấu câu;
   - câu hỏi cụt.

Dataset hiện tại đã có nhiều template khác nhau, nhưng nếu muốn sát production hơn thì nên bổ sung thêm lớp paraphrase/noise.

## M. Kiểm soát tính hợp lý của synthetic data

Khi sinh synthetic data, cần tránh các tổ hợp phi lý.

Ví dụ không nên sinh:

```text
Ngành Y khoa tại Đại học Ngoại thương
Ngành Marketing tại UET
Ngành Kỹ thuật điều khiển tại Đại học Kinh tế Quốc dân
```

Vì vậy script cần có mapping:

```text
school -> nhóm ngành có thể đào tạo
```

Sau đó chỉ sinh ngành nằm trong nhóm hợp lý.

Đây là lý do script có hàm tương đương:

```text
offered_majors(school)
schools_for_major(major)
```

Mục tiêu không phải mô phỏng chính xác 100% thực tế, mà là tránh các mẫu sai logic làm model học lệch.

## N. Chia train/validation/test

Dataset được chia:

```text
train: 4500
validation: 250
test: 250
```

Nguyên tắc:

- train dùng để học;
- validation dùng để chọn checkpoint/hyperparameter;
- test dùng để đánh giá cuối cùng;
- không để câu hỏi trùng giữa các split;
- nên giữ phân bố category tương đối ổn định.

Với dataset nhỏ 5000 mẫu, split 90/5/5 là hợp lý.

## O. Kiểm tra chất lượng dữ liệu

Sau khi sinh dữ liệu, cần chạy các kiểm tra:

1. Đủ số dòng:

```text
len(dataset) == 5000
```

2. Không trùng câu hỏi:

```text
len(set(question)) == len(dataset)
```

3. Không còn placeholder:

```text
[Tên trường], [Tên ngành], [Năm] không xuất hiện
```

4. Có oracle:

```text
len(oracle_document_ids) >= 1
```

5. Có distractor:

```text
len(distractor_document_ids) >= 1
```

6. Answer có nguồn:

```text
answer chứa doc_id tương ứng
```

7. Với ranking:

```text
có đủ item
sắp xếp đúng
không dùng điểm sàn
không thiếu học phí nếu câu hỏi yêu cầu học phí
```

8. Với filter:

```text
mọi item đều thỏa tất cả điều kiện
```

9. Với no-answer:

```text
answer không được bịa số liệu
```

## P. Rủi ro khi dùng synthetic data

Synthetic data có lợi vì dễ kiểm soát, nhưng có rủi ro:

- model học văn phong quá đều;
- số liệu không phản ánh thực tế;
- câu hỏi thiếu sự hỗn loạn tự nhiên của người dùng;
- document có cấu trúc quá sạch;
- distractor chưa đủ khó;
- model có thể học pattern thay vì kỹ năng đọc.

Do đó synthetic data nên là bước nền. Khi đưa vào production, nên trộn thêm:

- dữ liệu thật;
- log người dùng đã ẩn danh;
- câu trả lời chuyên gia;
- tài liệu crawl thực tế;
- hard negatives từ web search thật.

## Q. Quy trình tạo dữ liệu khuyến nghị

Một quy trình chuẩn nên gồm 8 bước:

1. **Xây taxonomy**
   - Liệt kê nhóm câu hỏi.
   - Xác định intent và slot cần điền.

2. **Chuẩn hóa entity**
   - Danh sách trường.
   - Danh sách ngành.
   - Alias, viết tắt, mã ngành.

3. **Sinh document**
   - Document điểm chuẩn.
   - Document học phí.
   - Document học vụ.
   - Document kiểm định.
   - Document chương trình đào tạo.

4. **Sinh câu hỏi**
   - Câu hỏi đơn.
   - Câu hỏi so sánh.
   - Câu hỏi ranking.
   - Câu hỏi filter.
   - Câu hỏi no-answer.

5. **Gán oracle/distractor**
   - Oracle chứa bằng chứng.
   - Distractor giống nhưng sai hoặc thiếu dữ liệu.

6. **Sinh answer**
   - Rule-based để đảm bảo đúng.
   - Có bảng khi cần.
   - Có nguồn.
   - Có lưu ý.

7. **Validate**
   - Kiểm tra schema.
   - Kiểm tra duplicate.
   - Kiểm tra oracle/distractor.
   - Kiểm tra placeholder.

8. **Xuất nhiều format**
   - Full dataset.
   - LoRA messages.
   - RAFT QA documents.
   - Summary.

## R. Kết luận lý thuyết

Tạo dữ liệu cho chatbot tuyển sinh không chỉ là tạo nhiều câu hỏi. Điều quan trọng hơn là tạo được **tình huống đọc hiểu có kiểm soát**:

- câu hỏi có intent rõ;
- tài liệu có oracle và distractor;
- answer bám nguồn;
- có mẫu thiếu dữ liệu;
- có mẫu reasoning nhiều điều kiện;
- có kiểm tra chất lượng sau sinh.

Với hướng này, model không chỉ học trả lời, mà còn học cách **không trả lời sai** khi bằng chứng không đủ. Đây là yêu cầu cốt lõi của hệ thống tư vấn tuyển sinh dùng RAG/RAFT.
