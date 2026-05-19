"""Instruction / prompt cho Question Decomposer.

Mục tiêu: tách câu hỏi phức tạp của user thành một DAG các sub-question nguyên tử,
phục vụ Multi-Hop Orchestrator.
"""

DECOMPOSER_INSTRUCTION = (
    "Bạn là chuyên gia phân rã câu hỏi cho hệ thống tư vấn tuyển sinh đại học. "
    "Nhiệm vụ: tách câu hỏi phức tạp thành các câu hỏi con NGUYÊN TỬ, chỉ khi CẦN THIẾT. "
    "Nếu câu hỏi đã đơn giản (1 ý duy nhất), trả về đúng 1 sub-question."
)

DECOMPOSER_PREFIX = """# MỤC TIÊU
Phân tích câu hỏi người dùng và trả ra DAG các sub-question để pipeline retrieval xử lý tuần tự hoặc song song.

# QUY TẮC
1. **Nguyên tử hóa**: mỗi sub-question chỉ hỏi 1 sự kiện / 1 con số / 1 danh sách nhỏ.
2. **Phụ thuộc (depends_on)**: liệt kê id các sub-q mà sub-q này CẦN kết quả trước mới trả lời được (multi-hop).
3. **Resolver**: chọn 1 trong 4:
   - `local_db`: thông tin có sẵn trong DB trường ĐH (điểm chuẩn, học phí, tuyển sinh, thông tin chung).
   - `web`: thông tin cần tìm ngoài (giảng viên, bảng xếp hạng, thủ khoa, CTĐT chi tiết...).
   - `hybrid`: vừa local vừa web (ít dùng, chỉ khi câu hỏi đa miền rõ rệt).
   - `reasoning`: sub-q CHỈ cần tính toán/suy luận từ kết quả sub-q khác (không retrieve).
4. **evidence_type**: `factual` / `numeric` / `list` / `comparison` / `computation`.
5. **LIMIT**: tối đa 5 sub-question. Nếu câu hỏi đơn giản → đúng 1 sub-question với depends_on=[].
6. **KHÔNG được tạo chu trình phụ thuộc.**

7. **Bat buoc final operation**:
   - Neu cau hoi yeu cau so sanh/loc/xep hang/top N/goi y theo dieu kien, PHAI co sub-question cuoi `resolver="reasoning"` phu thuoc vao tat ca sub-q du lieu lien quan.
   - Sub-q final phai neu ro phep toan: sort theo diem, loc diem > X, loc hoc phi < Y, giao cac danh sach, so sanh A/B.
8. **Coverage cho list/ranking/filter**:
   - Neu can danh sach nhieu truong hoac top N, uu tien `resolver="hybrid"` cho buoc thu thap du lieu de local DB va web bo sung nhau.
   - Khong chon vai truong theo phan doan; tach buoc lay candidates va buoc reasoning sort/filter.

# ĐỊNH DẠNG
Trả về CHỈ 1 JSON object (không thêm chữ nào khác):
```json
{
  "sub_questions": [
    {
      "id": 1,
      "text": "<câu hỏi con - viết đầy đủ rõ nghĩa>",
      "depends_on": [],
      "resolver": "local_db|web|hybrid|reasoning",
      "evidence_type": "factual|numeric|list|comparison|computation"
    }
  ]
}
```

# VÍ DỤ

## Ví dụ 1 — câu đơn giản
Câu hỏi: "Điểm chuẩn ngành CNTT UET năm 2024?"
```json
{
  "sub_questions": [
    {"id": 1, "text": "điểm chuẩn ngành CNTT UET năm 2024", "depends_on": [], "resolver": "local_db", "evidence_type": "numeric"}
  ]
}
```

## Ví dụ 2 — so sánh (song song + reasoning)
Câu hỏi: "So sánh học phí ngành CNTT UET và HUST năm 2025, chênh bao nhiêu phần trăm?"
```json
{
  "sub_questions": [
    {"id": 1, "text": "học phí ngành CNTT UET năm 2025", "depends_on": [], "resolver": "local_db", "evidence_type": "numeric"},
    {"id": 2, "text": "học phí ngành CNTT HUST năm 2025", "depends_on": [], "resolver": "local_db", "evidence_type": "numeric"},
    {"id": 3, "text": "tính phần trăm chênh lệch học phí CNTT giữa UET và HUST", "depends_on": [1, 2], "resolver": "reasoning", "evidence_type": "computation"}
  ]
}
```

## Ví dụ 3 — multi-hop bắc cầu thực thể
Câu hỏi: "Thủ khoa ngành Trí tuệ nhân tạo UET năm 2024 học trường THPT nào?"
```json
{
  "sub_questions": [
    {"id": 1, "text": "thủ khoa ngành Trí tuệ nhân tạo UET năm 2024 là ai", "depends_on": [], "resolver": "web", "evidence_type": "factual"},
    {"id": 2, "text": "trường THPT của thủ khoa AI UET 2024", "depends_on": [1], "resolver": "web", "evidence_type": "factual"}
  ]
}
```

## Ví dụ 4 — có điều kiện
Câu hỏi: "Với 26,5 điểm khối A00 thì có thể học ngành AI nào ở Hà Nội năm 2024?"
```json
{
  "sub_questions": [
    {"id": 1, "text": "danh sách trường đào tạo ngành AI ở Hà Nội năm 2024", "depends_on": [], "resolver": "web", "evidence_type": "list"},
    {"id": 2, "text": "điểm chuẩn ngành AI khối A00 năm 2024 của các trường ở Hà Nội", "depends_on": [1], "resolver": "local_db", "evidence_type": "numeric"},
    {"id": 3, "text": "lọc các ngành có điểm chuẩn ≤ 26.5", "depends_on": [2], "resolver": "reasoning", "evidence_type": "computation"}
  ]
}
```

## Ví dụ 5 — thủ tục nhiều mảnh
Câu hỏi: "Đăng ký học lại kỳ hè UET cần thủ tục, hạn và phí như thế nào?"
```json
{
  "sub_questions": [
    {"id": 1, "text": "thủ tục đăng ký học lại kỳ hè UET", "depends_on": [], "resolver": "web", "evidence_type": "factual"},
    {"id": 2, "text": "hạn đăng ký học lại kỳ hè UET", "depends_on": [], "resolver": "web", "evidence_type": "factual"},
    {"id": 3, "text": "phí học lại kỳ hè UET", "depends_on": [], "resolver": "web", "evidence_type": "numeric"}
  ]
}
```

# LƯU Ý CUỐI
- Nếu KHÔNG chắc có phải multi-hop → ưu tiên 1 sub-question (giữ nguyên câu gốc) để fallback pipeline cũ.
- Các sub-q song song (depends_on=[]) NÊN được tạo khi có thể, vì pipeline chạy song song rất nhanh.
- KHÔNG THÊM BẤT KỲ chữ nào ngoài JSON."""

DECOMPOSER_TEMPLATE = """Câu hỏi: {question}"""

FACT_EXTRACTOR_INSTRUCTION = (
    "Bạn là trợ lý trích xuất sự kiện ngắn gọn từ tài liệu. "
    "Đọc các đoạn dưới đây và trả lời CÂU HỎI bằng 1 câu ngắn (<= 25 từ). "
    "Nếu tài liệu KHÔNG chứa câu trả lời, trả về chuỗi rỗng."
)

FACT_EXTRACTOR_TEMPLATE = """Câu hỏi: {question}

Các đoạn liên quan:
{context}

Trả về JSON:
```json
{{"answer": "<câu trả lời ngắn gọn hoặc rỗng>", "confidence": <float 0.0-1.0>}}
```
Chỉ JSON, không giải thích."""


# ──────────────────────────────────────────────────────────────────
# LIST / NUMERIC / COMPARISON fact extractor
# Khác FACT_EXTRACTOR_INSTRUCTION: cho phép output list nhiều dòng,
# dùng khi sub-q evidence_type ∈ {list, numeric, comparison}.
# ──────────────────────────────────────────────────────────────────

LIST_FACT_EXTRACTOR_INSTRUCTION = (
    "Bạn là trợ lý trích xuất bảng/danh sách từ tài liệu cho hệ thống tư vấn tuyển sinh. "
    "Mục tiêu: liệt kê ĐẦY ĐỦ các entry liên quan đến CÂU HỎI từ tài liệu, mỗi entry kèm giá trị số/text gốc. "
    "TUYỆT ĐỐI KHÔNG bịa số liệu, KHÔNG bỏ sót entry có trong tài liệu. "
    "Nếu tài liệu KHÔNG chứa entry nào → trả mảng rỗng."
)

LIST_FACT_EXTRACTOR_TEMPLATE = """Câu hỏi: {question}

Các đoạn tài liệu:
{context}

# YÊU CẦU
- Đọc kỹ TẤT CẢ đoạn trên.
- Trích các entry phù hợp với câu hỏi (không bỏ sót).
- Mỗi entry dạng `<tên / chủ thể>: <giá trị gốc trong tài liệu>`.
- Nếu câu hỏi yêu cầu lọc theo NGÀNH cụ thể (CNTT, AI, ...), chỉ lấy entry thực sự thuộc ngành đó.

Trả về JSON:
```json
{{"answer": "<entry1>; <entry2>; <entry3>; ...", "items": [{{"name": "...", "value": "..."}}, ...], "confidence": <float 0.0-1.0>}}
```
- `answer`: chuỗi ghép các entry, phân cách bằng ` ; ` để hop sau dùng làm bridge.
- `items`: mảng từng entry rõ ràng (có thể rỗng).
- `confidence`: 0.95 nếu chắc chắn, 0.5 nếu không chắc, 0.0 nếu không tìm thấy.
Chỉ JSON, không giải thích."""


def select_fact_extractor_prompt(evidence_type: str) -> tuple[str, str]:
    """Chọn cặp (instruction, template) cho fact extractor theo evidence_type.

    - factual: 1 câu ≤25 từ → dùng FACT_EXTRACTOR_*
    - numeric/list/comparison/computation: trả list/items đầy đủ → dùng LIST_FACT_EXTRACTOR_*
    """
    et = (evidence_type or "factual").lower()
    if et in {"list", "numeric", "comparison", "computation"}:
        return LIST_FACT_EXTRACTOR_INSTRUCTION, LIST_FACT_EXTRACTOR_TEMPLATE
    return FACT_EXTRACTOR_INSTRUCTION, FACT_EXTRACTOR_TEMPLATE


# ──────────────────────────────────────────────────────────────────
# REASONER prompt
# Dùng cho sub-question có resolver=reasoning. LLM nhận:
#   - sub-q text (mô tả phép tính / lọc / so sánh cần làm)
#   - bằng chứng (fact + evidence chunks) từ các sub-q phụ thuộc
# Output: kết quả tính toán/lọc đầy đủ, không pass-through.
# ──────────────────────────────────────────────────────────────────

REASONER_INSTRUCTION = (
    "Bạn là chuyên gia suy luận / so sánh / lọc dữ liệu cho hệ thống tư vấn tuyển sinh. "
    "Nhiệm vụ: thực hiện ĐÚNG phép tính/so sánh/lọc được yêu cầu trên các bằng chứng đã thu thập. "
    "TUYỆT ĐỐI tuân thủ: (1) không bịa entry, (2) chỉ dùng số liệu có trong bằng chứng, "
    "(3) khi câu hỏi gốc yêu cầu lọc theo ngành/category cụ thể, BỎ QUA mọi entry không thuộc ngành/category đó."
)

REASONER_TEMPLATE = """# CÂU HỎI GỐC CỦA NGƯỜI DÙNG
{original_question}

# YÊU CẦU CỦA BƯỚC SUY LUẬN HIỆN TẠI
{sub_question}

# BẰNG CHỨNG TỪ CÁC SUB-QUESTION PHỤ THUỘC
{evidence}

# QUY TẮC
1. Đọc kỹ "Câu hỏi gốc" để hiểu category/điều kiện thực sự (ví dụ: chỉ ngành CNTT, chỉ học phí < 50 triệu).
2. Áp dụng phép tính/lọc được yêu cầu trong "Yêu cầu của bước suy luận" lên bằng chứng.
3. Nếu bằng chứng có entry thuộc ngành/category KHÁC với câu hỏi gốc → loại bỏ.
4. Nếu bằng chứng KHÔNG đủ → trả answer rỗng và confidence thấp.
5. Trình bày kết quả theo dạng có thể đọc được (list/bảng đơn giản).

Trả về JSON:
```json
{{"answer": "<kết quả chi tiết, có thể là list nhiều dòng phân cách bằng ' ; '>", "items": [{{"name": "...", "value": "..."}}, ...], "confidence": <float 0.0-1.0>, "explanation": "<1-2 câu giải thích cách lọc/tính>"}}
```
Chỉ JSON, không giải thích thêm bên ngoài."""


# Runtime prompt hardening. Kept as append-only text so existing templates stay compatible.
DECOMPOSER_PREFIX += """

# BAT BUOC CHO CAU HOI SO SANH / LOC / XEP HANG
- So sanh/loc/xep hang/top N/goi y theo dieu kien phai co sub-question cuoi `resolver="reasoning"`.
- Buoc reasoning cuoi phai depends_on TAT CA sub-question du lieu can dung.
- Neu can top N hoac danh sach nhieu truong, dung `resolver="hybrid"` cho buoc thu thap candidates, khong tu doan san danh sach truong.
"""

LIST_FACT_EXTRACTOR_INSTRUCTION += (
    " Khi doc bang, phai chon dung DONG theo ten nganh/truong va dung COT theo nam duoc hoi. "
    "Khong lay gia tri o dong dau tien cua bang neu dong do khong phai nganh duoc hoi."
)

REASONER_INSTRUCTION += (
    " Neu output khong phai JSON hop le, cau tra loi se bi loai bo. "
    "Khong lap lai mot gia tri nhieu lan. Khong dung gia tri diem chuan thay cho hoc phi."
)

REASONER_TEMPLATE += """

# KIEM TRA BAT BUOC TRUOC KHI TRA JSON
- Moi item trong answer phai co du bang chung tu evidence. Neu thieu hoc phi/diem cua mot truong, ghi ro `unknown`, khong suy doan.
- Voi sort/ranking: sap xep bang so thuc giam/tang dung yeu cau, khong tron thang diem khac nhau neu khong ghi ro.
- Voi filter: chi giu item thoa TAT CA dieu kien.
- Output phai la JSON object hop le, khong boc trong markdown fence.
"""
