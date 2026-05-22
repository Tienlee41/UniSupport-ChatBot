# Bộ dữ liệu LoRA + RAFT 5000 mẫu

Bộ dữ liệu này được sinh từ các nhóm câu hỏi trong:

- `danh_sach_cau_hoi_finetune.md`
- `thong_tin_cong_khai_kiem_dinh.md`

Dữ liệu là synthetic nhưng được thiết kế theo kiểu grounded QA: mỗi câu hỏi đều có tài liệu tham khảo đi kèm, câu trả lời phải bám theo tài liệu và có cả các mẫu guardrail khi tài liệu thiếu thông tin.

## File đầu ra

| File | Mục đích |
|---|---|
| `lora_raft_dataset_5000.jsonl` | File đầy đủ: question, answer, documents, messages, metadata |
| `lora_messages_5000.jsonl` | Dùng trực tiếp cho SFT/LoRA dạng chat messages |
| `raft_qa_documents_5000.jsonl` | Dùng cho RAFT/RAG: question, answer, documents, oracle/distractor doc ids |
| `lora_raft_dataset_5000_summary.json` | Thống kê split, category, task_type và schema |
| `build_lora_raft_dataset.py` | Script sinh lại dữ liệu, deterministic theo seed |

## Schema chính

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
    {"role": "user", "content": "Thông tin tham khảo... Câu hỏi: ..."},
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

## Phân bổ

- Tổng: 5000 mẫu
- Split: train 4500, validation 250, test 250
- Nhóm chính:
  - tuyển sinh/điểm chuẩn/điểm sàn
  - chỉ tiêu/phương thức/tổ hợp
  - học phí/tài chính
  - học vụ sinh viên
  - thông tin công khai/kiểm định
  - ngành đào tạo/nghề nghiệp
  - so sánh/tư vấn/xếp hạng/lọc điều kiện
  - không đủ dữ liệu/guardrail

## Sinh lại dữ liệu

```powershell
python finetune\build_lora_raft_dataset.py
```

Có thể đổi thư mục output:

```powershell
python finetune\build_lora_raft_dataset.py --out-dir finetune --seed 20260521
```

## Lưu ý huấn luyện

- Với LoRA reader, nên dùng `lora_messages_5000.jsonl` để model học cách đọc tài liệu trong user prompt rồi trả lời.
- Với RAFT, dùng `raft_qa_documents_5000.jsonl`; `oracle_document_ids` là tài liệu đủ thông tin, `distractor_document_ids` là tài liệu nhiễu. Mỗi mẫu có ít nhất một oracle document và ít nhất một distractor document.
- Không dùng dữ liệu này như cơ sở tri thức thật về điểm chuẩn/học phí. Mục tiêu chính là dạy hành vi bám nguồn, lọc điều kiện, phân biệt điểm sàn với điểm chuẩn và biết từ chối khi thiếu bằng chứng.
