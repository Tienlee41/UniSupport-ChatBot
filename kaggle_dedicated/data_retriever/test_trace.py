"""Structured test logging for the retrieval and reader pipeline.

The regular logs are useful for live debugging, but test runs need stable
input/output blocks for every pipeline step. These helpers keep that format
consistent without changing retrieval behavior.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any


_FALSE_VALUES = {"0", "false", "no", "off"}
_STEP_LOCK = threading.Lock()
_STEP_COUNTER = 0
_OPEN_STEPS: dict[str, list[int]] = {}


STEP_NOTES: dict[str, str] = {
    "SOURCE_ROUTER": "Chọn nguồn truy xuất: local DB, web search, hybrid hoặc không truy xuất.",
    "LOCAL_DB": "Tìm tài liệu trong cơ sở dữ liệu nội bộ theo school_id và section.",
    "QUERY": "Chuẩn hóa câu hỏi hoặc sinh truy vấn tìm kiếm từ câu hỏi/sub-query.",
    "RETRIEVAL_CONFIG": "Ghi lại cấu hình retrieval đang được dùng cho truy vấn hiện tại.",
    "WEB_SEARCH": "Gửi truy vấn lên search engine và nhận danh sách URL ứng viên.",
    "WEB_SEARCH_LLM_RERANK": "Xếp hạng lại các kết quả web ở mức trang; nếu tắt thì ghi rõ lý do bỏ qua.",
    "CRAWL_EXTRACT": "Chọn URL cần tải, crawl trang và lấy HTML/snippet đầu vào cho trích xuất.",
    "SNIPPET_CHECK": "Đánh giá snippet có đủ dùng không để tránh crawl không cần thiết.",
    "CONTENT_EXTRACT": "Chuyển HTML/PDF/image đã tải thành web source dạng text.",
    "QUALITY_GATE": "Kiểm tra độ liên quan, độ tin cậy và an toàn nguồn trước khi đưa vào RAG.",
    "CHUNKING": "Cắt nội dung nguồn thành các chunk ứng viên cho RAG.",
    "RERANK_CHUNK": "Chấm/rerank các chunk theo truy vấn và giữ chunk phù hợp nhất.",
    "DEDUP": "Gộp, khử trùng lặp và xử lý danh sách chunk cuối cho reader.",
    "MULTIHOP_DECOMPOSE": "Phân rã câu hỏi phức tạp thành DAG các sub-query.",
    "MULTIHOP_PLAN": "Ghi kế hoạch multi-hop gồm sub-query, thứ tự phụ thuộc và resolver.",
    "MULTIHOP_EXECUTE": "Chạy các sub-query đã sẵn sàng theo thứ tự topo; các sub-query độc lập chạy song song.",
    "MULTIHOP_AGGREGATE": "Gộp evidence từ toàn bộ sub-query và khử trùng lặp cuối.",
    "SUBQUERY": "Vòng đời của một sub-query: nhận input, rewrite, retrieve/reason và xuất fact/evidence.",
    "SUBQUERY_RETRIEVE": "Kết quả retrieval của sub-query sau khi chạy như single-hop.",
    "SUBQUERY_FALLBACK": "Fallback nguồn truy xuất, ví dụ local DB rỗng thì thử web search.",
    "REASONING": "Suy luận trên fact/evidence từ các sub-query phụ thuộc.",
    "FINAL_AGGREGATE": "Tập web sources và RAG chunks cuối cùng trước khi kiểm tra đủ bằng chứng.",
    "SUFFICIENCY_GATE": "Kiểm tra ngữ cảnh RAG có đủ để reader trả lời hay không.",
    "RAG_CONTEXT": "Format các RAG chunks thành context đưa vào reader.",
    "FINAL_READER": "Reader sinh câu trả lời cuối cùng từ câu hỏi và ngữ cảnh RAG.",
}


def trace_enabled(params: dict | None = None) -> bool:
    if params and bool(params.get("quality_log", False)):
        return True
    return str(os.getenv("BOT_TEST_TRACE", "0")).strip().lower() not in _FALSE_VALUES


def trace_scope(params: dict | None = None, scope: str | None = None) -> str:
    if scope:
        return scope
    if params:
        subq_id = params.get("_trace_subq_id")
        if subq_id is not None:
            return f"SQ#{subq_id}"
        return str(params.get("_trace_scope", "SINGLE_HOP"))
    return "SINGLE_HOP"


def _json_default(value: Any) -> str:
    return str(value)


def _next_step_number(label: str, step: str, direction: str) -> int:
    global _STEP_COUNTER
    direction = (direction or "").upper()
    key = f"{label}|{step}"
    with _STEP_LOCK:
        if direction == "INPUT":
            _STEP_COUNTER += 1
            _OPEN_STEPS.setdefault(key, []).append(_STEP_COUNTER)
            return _STEP_COUNTER
        if direction == "OUTPUT" and _OPEN_STEPS.get(key):
            return _OPEN_STEPS[key].pop(0)
        _STEP_COUNTER += 1
        return _STEP_COUNTER


def log_step(
    step: str,
    direction: str,
    payload: Any,
    params: dict | None = None,
    *,
    scope: str | None = None,
) -> None:
    """Print one stable test-trace block.

    Args:
        step: Pipeline step name, e.g. WEB_SEARCH, QUALITY_GATE.
        direction: INPUT or OUTPUT.
        payload: JSON-serializable payload.
        params: Runtime params; quality_log=True enables output.
        scope: Optional scope label. Defaults to SQ#n when _trace_subq_id exists.
    """
    if not trace_enabled(params):
        return
    label = trace_scope(params, scope)
    step_number = _next_step_number(label, step, direction)
    note = STEP_NOTES.get(step, "Ghi lại input/output của bước pipeline.")
    record = {
        "step_number": step_number,
        "scope": label,
        "step": step,
        "direction": (direction or "").upper(),
        "note": note,
        "data": payload,
    }
    print("\n" + "=" * 80)
    print(f"[TEST_TRACE][BƯỚC {step_number:03d}][{label}][{step}][{(direction or '').upper()}]")
    print("=" * 80)
    print(json.dumps(record, ensure_ascii=False, indent=2, default=_json_default))


def preview(text: Any, limit: int = 240) -> str:
    value = "" if text is None else str(text)
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def compact_search_result(item: dict, rank: int | None = None) -> dict:
    payload = {
        "rank": rank,
        "query": item.get("query", ""),
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "score": item.get("score"),
        "snippet": preview(item.get("description", ""), 280),
    }
    return {k: v for k, v in payload.items() if v is not None}


def compact_html_result(item: dict, rank: int | None = None) -> dict:
    payload = compact_search_result(item, rank)
    html = item.get("html", "") or ""
    payload.update({
        "html_chars": len(html),
        "html_preview": preview(html, 220),
    })
    return payload


def compact_web_source(item: dict, rank: int | None = None) -> dict:
    text = item.get("text", "") or ""
    files = item.get("files") or []
    payload = {
        "rank": rank,
        "query": item.get("query", ""),
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "score": item.get("score"),
        "text_chars": len(text),
        "text_preview": preview(text, 260),
        "file_count": len(files),
    }
    return {k: v for k, v in payload.items() if v is not None}


def compact_rag_source(item: dict, rank: int | None = None) -> dict:
    text = item.get("text", "") or ""
    payload = {
        "rank": rank,
        "query": item.get("query", ""),
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "chunk_index": item.get("chunk_index"),
        "score": item.get("score"),
        "text_chars": len(text),
        "text_preview": preview(text, 280),
    }
    return {k: v for k, v in payload.items() if v is not None}


def compact_sources(items: list[dict], kind: str, limit: int = 20) -> list[dict]:
    compactor = {
        "search": compact_search_result,
        "html": compact_html_result,
        "web": compact_web_source,
        "rag": compact_rag_source,
    }[kind]
    return [compactor(item, idx) for idx, item in enumerate(items[:limit], 1)]
