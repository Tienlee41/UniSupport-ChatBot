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
    "REQUEST": "Nhan cau hoi nguoi dung va cau hinh runtime dau vao.",
    "SOURCE_ROUTER": "Chon nguon truy xuat: local DB, web search, hybrid hoac khong truy xuat.",
    "LOCAL_DB": "Tim tai lieu trong co so du lieu noi bo theo school_id va section.",
    "QUERY": "Sinh hoac chuan hoa truy van tim kiem tu cau hoi/sub-query.",
    "RETRIEVAL_CONFIG": "Ghi lai cau hinh retrieval dang duoc dung cho truy van hien tai.",
    "WEB_SEARCH": "Gui truy van len search engine va nhan danh sach URL ung vien.",
    "WEB_SEARCH_LLM_RERANK": "Xep hang lai ket qua web o muc trang; neu tat thi ghi ro ly do bo qua.",
    "QUALITY_GATE": "Loc URL truoc crawl bang relevance, trust, safety va duplicate check.",
    "SNIPPET_CHECK": "Danh gia snippet co du dung khong de tranh crawl khong can thiet.",
    "CRAWL_EXTRACT": "Chon URL can tai va crawl/snippet thanh HTML dau vao.",
    "CONTENT_EXTRACT": "Chuyen HTML/PDF/image da tai thanh web source dang text.",
    "CHUNKING": "Cat noi dung nguon thanh cac chunk ung vien cho RAG.",
    "RERANK_CHUNK": "Cham/rerank cac chunk theo truy van va giu chunk phu hop nhat.",
    "DEDUP": "Gop, khu trung lap va xu ly danh sach chunk cuoi cho reader.",
    "CHUNK_GATE": "Loc web sources va chunks sau crawl/chunking truoc khi dua vao context RAG.",
    "MULTIHOP_DECOMPOSE": "Phan ra cau hoi phuc tap thanh DAG cac sub-query.",
    "MULTIHOP_PLAN": "Ghi ke hoach multi-hop gom sub-query, thu tu phu thuoc va resolver.",
    "MULTIHOP_EXECUTE": "Chay cac sub-query san sang theo thu tu topo; sub-query doc lap co the chay song song.",
    "MULTIHOP_AGGREGATE": "Gop evidence tu toan bo sub-query va khu trung lap cuoi.",
    "SUBQUERY": "Vong doi mot sub-query: input, rewrite, retrieve/reason va fact/evidence dau ra.",
    "SUBQUERY_RETRIEVE": "Ket qua retrieval cua sub-query sau khi chay nhu single-hop.",
    "SUBQUERY_FALLBACK": "Fallback nguon truy xuat, vi du local DB rong thi thu web search.",
    "REASONING": "Suy luan tren fact/evidence tu cac sub-query phu thuoc.",
    "FINAL_AGGREGATE": "Tap web sources va RAG chunks cuoi cung truoc khi kiem tra du bang chung.",
    "SUFFICIENCY_GATE": "Kiem tra ngu canh RAG co du de reader tra loi hay khong.",
    "RAG_CONTEXT": "Format cac RAG chunks thanh context dua vao reader.",
    "FINAL_READER": "Reader sinh cau tra loi cuoi cung tu cau hoi va ngu canh RAG.",
}


def reset_trace() -> None:
    global _STEP_COUNTER
    with _STEP_LOCK:
        _STEP_COUNTER = 0
        _OPEN_STEPS.clear()


def trace_enabled(params: dict | None = None) -> bool:
    if params and bool(params.get("quality_log", False)):
        return True
    if params and bool(params.get("test_trace", False)):
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
    """Print one stable test-trace block."""
    if not trace_enabled(params):
        return
    label = trace_scope(params, scope)
    step_number = _next_step_number(label, step, direction)
    note = STEP_NOTES.get(step, "Ghi lai input/output cua buoc pipeline.")
    record = {
        "step_number": step_number,
        "scope": label,
        "step": step,
        "direction": (direction or "").upper(),
        "note": note,
        "data": payload,
    }
    print("\n" + "=" * 80)
    print(f"[TEST_TRACE][BUOC {step_number:03d}][{label}][{step}][{(direction or '').upper()}]")
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
