"""Multi-Hop Orchestrator — bước đầu xử lý câu hỏi phức tạp.

Thành phần:
- `SubQuestion` / `DecomposerPlan`: data class mô tả DAG.
- `DecomposerProtocol`: interface cho LLM decompose.
- `FactExtractorProtocol` (tùy chọn): trích fact trung gian cho multi-hop.
- `MultiHopOrchestrator`: thực thi DAG với retrieval có sẵn.

Điểm quan trọng:
- **Opt-in**: mặc định disabled, bật qua `MultiHopConfig.enabled=True` hoặc
  `params["use_multi_hop"]=True`.
- **Fallback**: nếu decomposer lỗi hoặc trả ≤1 sub-q → trả về single-hop pipeline cũ.
- **Chi phí kiểm soát**: `max_hops=3`, `max_sub_questions=5`.
- **Không breaking**: orchestrator là 1 lớp bọc ngoài `RouterRetriever`, không sửa nội bộ.
"""
from __future__ import annotations

import asyncio
import copy
import json
import re
import traceback
import unicodedata
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, Protocol

from .schema import RagSource, WebSource
from .retriever.utils import CmdLogger

try:
    from server import GenerationParams
except Exception:  # khi dùng ngoài pipeline
    GenerationParams = dict  # type: ignore


# ──────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────

@dataclass
class SubQuestion:
    id: int
    text: str
    depends_on: list[int] = field(default_factory=list)
    resolver: str = "web"                 # local_db | web | hybrid | reasoning
    evidence_type: str = "factual"        # factual | numeric | list | comparison | computation


@dataclass
class DecomposerPlan:
    sub_questions: list[SubQuestion]

    @property
    def is_trivial(self) -> bool:
        return len(self.sub_questions) <= 1


@dataclass
class SubQuestionResult:
    sub_id: int
    rewritten_text: str
    web_sources: list[WebSource] = field(default_factory=list)
    rag_sources: list[RagSource] = field(default_factory=list)
    fact: str = ""
    confidence: float = 0.0


@dataclass
class MultiHopTrace:
    """Trace toàn bộ quá trình multi-hop để log / debug / hiện trên UI."""
    plan: DecomposerPlan
    hop_count: int
    per_sub_q: dict[int, SubQuestionResult]


@dataclass
class MultiHopConfig:
    enabled: bool = False
    auto_detect_complexity: bool = True
    complexity_threshold: int = 2
    max_hops: int = 3
    max_sub_questions: int = 5
    enable_fact_extraction: bool = True
    fact_confidence_threshold: float = 0.3
    # Giới hạn số query web per sub-q (sub-q đã atomic nên mặc định 1).
    subq_max_query: int = 1
    # Fallback: khi resolver=local_db trả empty, tự thử web.
    fallback_web_when_local_empty: bool = True
    # Thêm synthetic RagSource cho reasoning SQ để reader thấy fact trung gian.
    materialize_reasoning_evidence: bool = True

    # ── P0: LLM-based reasoning ──
    # Khi True, resolver=reasoning sẽ gọi ReasonerProtocol (LLM) để tính/lọc/so sánh
    # trên evidence từ deps thay vì pass-through.
    enable_llm_reasoning: bool = True
    # Số chunks tối đa lấy từ MỖI dep để feed cho reasoner.
    reasoning_max_evidence_chunks_per_dep: int = 6
    # Cắt chunk text khi feed reasoner để kiểm soát prompt length.
    reasoning_max_chars_per_chunk: int = 1200
    reasoning_confidence_threshold: float = 0.65
    allow_reasoning_pass_through_fallback: bool = False

    # ── P2: skip bridge entity khi dep là list/comparison ──
    # Tránh "(tức 18 trường: A; B; C…)" làm noise cho keyword extractor.
    skip_bridge_entity_for_types: tuple = ("list", "comparison", "computation")

    # ── Major-keyword filter: lọc chunks không khớp ngành/category ──
    enable_major_keyword_filter: bool = True
    # Khi sau filter còn 0 chunks → giữ lại bản gốc (tránh wipe).
    major_filter_keep_minimum: int = 1


# ──────────────────────────────────────────────────────────────────
# Protocols
# ──────────────────────────────────────────────────────────────────

class DecomposerProtocol(Protocol):
    async def decompose(self, question: str, params: "GenerationParams") -> DecomposerPlan: ...


class FactExtractorProtocol(Protocol):
    async def extract(
        self,
        sub_q_text: str,
        rag_sources: list[RagSource],
        params: "GenerationParams",
        evidence_type: str = "factual",
    ) -> tuple[str, float]: ...


class ReasonerProtocol(Protocol):
    """LLM thực hiện tính toán / lọc / so sánh trên bằng chứng từ deps.

    Trả (answer_text, confidence). answer_text có thể là list nhiều dòng.
    """
    async def reason(
        self,
        original_question: str,
        sub_q_text: str,
        evidence_blocks: list[str],
        params: "GenerationParams",
    ) -> tuple[str, float]: ...


class BaseRetrieverProtocol(Protocol):
    """Retriever tái sử dụng (RouterRetriever hoặc tương đương)."""
    async def retrieve(
        self, question: str, params: "GenerationParams"
    ) -> tuple[list[WebSource], list[RagSource]]: ...


# ──────────────────────────────────────────────────────────────────
# Utility: parse LLM output thành DecomposerPlan
# ──────────────────────────────────────────────────────────────────

_JSON_BLOCK_PATTERN = re.compile(r"```json\s*(.*?)```", re.DOTALL)


def _extract_first_json(text: str) -> Optional[dict]:
    """Parse JSON an toàn từ output LLM."""
    m = _JSON_BLOCK_PATTERN.search(text)
    candidate = m.group(1).strip() if m else text.strip()
    try:
        return json.loads(candidate)
    except Exception:
        # Thử tìm JSON object "ôm" cả text (LLM hay bọc thêm giải thích)
        brace_start = candidate.find("{")
        brace_end = candidate.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(candidate[brace_start : brace_end + 1])
            except Exception:
                return None
        return None


def parse_plan_json(text: str, max_sub: int = 5) -> Optional[DecomposerPlan]:
    data = _extract_first_json(text)
    if not data or "sub_questions" not in data:
        return None
    items = data["sub_questions"]
    if not isinstance(items, list) or not items:
        return None
    sub_qs: list[SubQuestion] = []
    for it in items[:max_sub]:
        try:
            sub_qs.append(
                SubQuestion(
                    id=int(it["id"]),
                    text=str(it["text"]).strip(),
                    depends_on=[int(x) for x in it.get("depends_on", []) if isinstance(x, (int, float))],
                    resolver=str(it.get("resolver", "web")).lower(),
                    evidence_type=str(it.get("evidence_type", "factual")).lower(),
                )
            )
        except Exception:
            continue
    if not sub_qs:
        return None
    # Kiểm tra chu trình đơn giản (DFS)
    id_set = {sq.id for sq in sub_qs}
    graph = {sq.id: [d for d in sq.depends_on if d in id_set] for sq in sub_qs}
    if _has_cycle(graph):
        return None
    return DecomposerPlan(sub_questions=sub_qs)


def _has_cycle(graph: dict[int, list[int]]) -> bool:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}

    def dfs(node: int) -> bool:
        color[node] = GRAY
        for nb in graph.get(node, []):
            if color.get(nb, WHITE) == GRAY:
                return True
            if color.get(nb, WHITE) == WHITE and dfs(nb):
                return True
        color[node] = BLACK
        return False

    return any(dfs(n) for n in graph if color[n] == WHITE)


# ──────────────────────────────────────────────────────────────────
# Query rewriting với resolved facts
# ──────────────────────────────────────────────────────────────────

def rewrite_with_resolved(
    sq: SubQuestion,
    resolved: dict[int, str],
    dep_evidence_types: Optional[dict[int, str]] = None,
    skip_types: tuple = ("list", "comparison", "computation"),
) -> str:
    """Chèn fact đã resolve vào sub-q text dưới dạng câu tiếng Việt tự nhiên.

    Quy tắc P2: nếu dep có evidence_type ∈ skip_types (list/comparison/computation)
    thì BỎ QUA — vì fact của dep đó là một danh sách/bảng dài, chèn vào sẽ
    nhiễu cho keyword extractor và router LLM. Reasoner sẽ tự đọc evidence_blocks
    đầy đủ ở bước reasoning.
    """
    if not sq.depends_on:
        return sq.text
    facts: list[str] = []
    for d in sq.depends_on:
        if d not in resolved or not resolved[d]:
            continue
        if dep_evidence_types is not None:
            et = (dep_evidence_types.get(d) or "factual").lower()
            if et in skip_types:
                continue
        f = resolved[d].strip().rstrip(".")
        if f:
            facts.append(f)
    if not facts:
        return sq.text
    if len(facts) == 1:
        clause = f"(tức {facts[0]})"
    else:
        clause = f"(biết rằng: {'; '.join(facts)})"
    return f"{sq.text} {clause}"


# ──────────────────────────────────────────────────────────────────
# Major-keyword filter
# ──────────────────────────────────────────────────────────────────

# Ánh xạ viết tắt phổ biến → variants xuất hiện trong tài liệu.
# Khi sub-q đề cập tới các viết tắt này, chỉ giữ chunks chứa ≥1 variant.
_MAJOR_ABBREVIATIONS: dict[str, list[str]] = {
    "cntt": ["công nghệ thông tin", "cntt", "information technology"],
    "ai": ["trí tuệ nhân tạo", " ai ", "artificial intelligence"],
    "ttntao": ["trí tuệ nhân tạo"],
    "khmt": ["khoa học máy tính", "computer science"],
    "ktpm": ["kỹ thuật phần mềm", "software engineering"],
    "attt": ["an toàn thông tin", "cyber security", "an ninh mạng"],
    "httt": ["hệ thống thông tin"],
    "dttt": ["điện tử viễn thông", "điện tử - viễn thông", "điện tử truyền thông"],
}


_PROGRAM_PATTERNS = [
    re.compile(r"ng[àa]nh\s+([A-Za-zÀ-ỹĐđ\s]+?)(?=\s+(?:c[ủu]a|n[ăa]m|tr[ưu][ơờ]ng|t[ạa]i|với|\d|$|,|;|\.))", re.IGNORECASE),
]


def _extract_major_keywords(sub_q_text: str) -> list[str]:
    """Trích keyword ngành/category từ sub-q text.

    Returns: list các string (lowercase, đã trim) — chunks phải chứa ≥1
    string trong list này thì mới được giữ.
    """
    text = (sub_q_text or "").lower()
    out: list[str] = []

    # 1) Viết tắt
    # Thêm word-boundary để tránh match "AI" trong "WAI", "TAI", ...
    for abbr, variants in _MAJOR_ABBREVIATIONS.items():
        # boundary check: dùng regex \bABBR\b (lowercase)
        if re.search(rf"\b{re.escape(abbr)}\b", text):
            out.extend(v.strip().lower() for v in variants)

    # 2) Pattern "ngành <X>"
    for pat in _PROGRAM_PATTERNS:
        for m in pat.finditer(text):
            kw = (m.group(1) or "").strip()
            if 2 <= len(kw) <= 80:
                out.append(kw)

    # Dedup giữ thứ tự
    seen: set[str] = set()
    dedup: list[str] = []
    for kw in out:
        if kw and kw not in seen:
            seen.add(kw)
            dedup.append(kw)
    return dedup


def filter_chunks_by_major(
    chunks: list[RagSource],
    keywords: list[str],
    keep_minimum: int = 1,
) -> list[RagSource]:
    """Giữ chunks chứa ≥1 keyword (case-insensitive substring match).

    Nếu sau filter còn ít hơn `keep_minimum` chunks → giữ nguyên gốc để tránh
    xóa sạch khi keyword không khớp do format tài liệu khác.
    """
    if not keywords or not chunks:
        return chunks
    norm_kws = [k.lower() for k in keywords if k]
    filtered = []
    for c in chunks:
        text = (c.get("text", "") or "").lower()
        title = (c.get("title", "") or "").lower()
        haystack = title + " || " + text
        if any(kw in haystack for kw in norm_kws):
            filtered.append(c)
    if len(filtered) < keep_minimum:
        return chunks  # fallback giữ nguyên
    return filtered


def _strip_accents_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d").replace("Đ", "D")


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_accents_for_match(text).lower()).strip()


def _term_in_text(term: str, text: str) -> bool:
    term_norm = _normalize_for_match(term)
    if not term_norm:
        return False
    if re.fullmatch(r"[a-z0-9\s.+/#&-]+", term_norm):
        return re.search(rf"(?<!\w){re.escape(term_norm)}(?!\w)", text) is not None
    return term_norm in text


def _query_focus_terms(question: str) -> list[str]:
    text = _normalize_for_match(question)
    terms: list[str] = []
    signal_terms = [
        "tuyen sinh", "de an tuyen sinh", "thong tin tuyen sinh",
        "diem chuan", "diem trung tuyen", "diem xet tuyen", "diem san",
        "diem nhan ho so", "nguong dau vao", "nguong dam bao chat luong",
        "hoc phi", "le phi", "hoc bong", "mien giam", "tin dung sinh vien",
        "chi tieu", "ma nganh", "ten nganh", "to hop mon", "khoi thi",
        "phuong thuc xet tuyen", "xet tuyen", "xet tuyen som", "xet tuyen ket hop",
        "tuyen thang", "uu tien xet tuyen", "nguyen vong", "ho so",
        "thoi gian dang ky", "han dang ky", "lich tuyen sinh", "nhap hoc",
        "xac nhan nhap hoc", "cong bo ket qua",
        "hoc ba", "thi thpt", "tot nghiep thpt", "danh gia nang luc",
        "danh gia tu duy", "dgnl", "dgtd", "tsa", "sat", "ielts", "toefl",
        "chuong trinh dao tao", "chuan dau ra", "thoi gian dao tao",
        "bang cap", "cu nhan", "ky su", "chat luong cao", "tien tien",
        "lien ket quoc te", "song bang",
        "ky tuc xa", "noi tru", "ngoai tru", "co so dao tao", "dia diem hoc",
        "campus", "co so vat chat",
        "co hoi viec lam", "viec lam", "muc luong", "thuc tap",
        "doanh nghiep", "dau ra",
    ]
    terms.extend(term for term in signal_terms if _term_in_text(term, text))

    phrase_patterns = [
        r"\b(?:nganh|nhom nganh|linh vuc|chuyen nganh|chuong trinh|chuong trinh dao tao|khoa)\s+([a-z0-9][a-z0-9\s/&+.\-]{2,80})",
        r"\b(?:truong|dai hoc|hoc vien)\s+([a-z0-9][a-z0-9\s/&+.\-]{2,80})",
    ]
    stop_tail = re.compile(
        r"\b(?:nam|tai|o|cua|cac|nhung|co|khong|la|bao nhieu|cao|thap|nhat|kem|voi|va|so sanh|xep hang|top|theo)\b.*$"
    )
    for pattern in phrase_patterns:
        for match in re.finditer(pattern, text):
            phrase = stop_tail.sub("", match.group(1)).strip(" ,.;:-")
            if len(phrase) >= 3:
                terms.append(phrase)

    for abbr in re.findall(r"\b[A-Z0-9][A-Z0-9+/&.-]{1,8}\b", question or ""):
        terms.append(_normalize_for_match(abbr))

    domain_aliases = {
        "cong nghe thong tin": ["cntt", "it", "information technology"],
        "khoa hoc may tinh": ["khmt", "computer science"],
        "ky thuat may tinh": ["computer engineering"],
        "tri tue nhan tao": ["ai", "artificial intelligence"],
        "khoa hoc du lieu": ["khdl", "data science"],
        "an toan thong tin": ["attt", "information security"],
        "an ninh mang": ["cyber security", "cybersecurity"],
        "ky thuat phan mem": ["software engineering", "ktpm"],
        "he thong thong tin": ["httt", "information systems"],
        "mang may tinh": ["computer network", "networking"],
        "thuong mai dien tu": ["tmdt", "ecommerce", "e-commerce"],
        "dien tu vien thong": ["dtvt", "electronics and telecommunications"],
        "ky thuat dien": ["dien dien tu", "electrical engineering"],
        "dieu khien va tu dong hoa": ["automation", "tu dong hoa"],
        "co dien tu": ["mechatronics"],
        "ky thuat co khi": ["co khi", "mechanical engineering"],
        "ky thuat o to": ["cong nghe ky thuat o to", "automotive engineering"],
        "xay dung": ["ky thuat xay dung", "civil engineering"],
        "kien truc": ["architecture"],
        "ky thuat hoa hoc": ["hoa hoc", "chemical engineering"],
        "cong nghe thuc pham": ["food technology"],
        "ky thuat moi truong": ["moi truong", "environmental engineering"],
        "logistics": ["logistics va quan ly chuoi cung ung", "supply chain"],
        "quan tri kinh doanh": ["qtkd", "business administration"],
        "marketing": ["digital marketing"],
        "kinh doanh quoc te": ["international business"],
        "tai chinh ngan hang": ["finance banking", "ngan hang"],
        "ke toan": ["accounting"],
        "kiem toan": ["auditing"],
        "ngon ngu anh": ["english language"],
        "ngon ngu trung": ["tieng trung", "chinese language"],
        "ngon ngu nhat": ["tieng nhat", "japanese language"],
        "luat": ["law"],
        "luat kinh te": ["economic law"],
        "quan he quoc te": ["international relations"],
        "bao chi": ["journalism"],
        "truyen thong da phuong tien": ["multimedia communication"],
        "tam ly hoc": ["psychology"],
        "su pham": ["giao duc", "teacher education"],
        "y khoa": ["medicine", "bac si da khoa"],
        "duoc hoc": ["pharmacy"],
        "dieu duong": ["nursing"],
        "rang ham mat": ["dentistry"],
        "y hoc co truyen": ["traditional medicine"],
        "xet nghiem y hoc": ["medical laboratory"],
        "thu y": ["veterinary"],
        "nong nghiep": ["agriculture"],
        "thuy san": ["aquaculture"],
        "thiet ke do hoa": ["graphic design"],
        "my thuat": ["fine arts"],
    }
    for canonical, aliases in domain_aliases.items():
        if _term_in_text(canonical, text) or any(_term_in_text(alias, text) for alias in aliases):
            terms.extend([canonical, *aliases])
    terms.extend(re.findall(r"\b(?:19|20)\d{2}\b", question or ""))
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        norm = _normalize_for_match(term)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _select_relevant_text(question: str, text: str, max_chars: int) -> str:
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return text or ""
    focus_terms = _query_focus_terms(question)
    q_tokens = set(re.findall(r"\w+", _normalize_for_match(question), flags=re.UNICODE))
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return text[:max_chars]
    selected_idx: set[int] = set()
    scored: list[tuple[int, int]] = []
    for idx, line in enumerate(lines):
        norm = _normalize_for_match(line)
        score = 0
        if "|" in line and any(h in norm for h in [
            "ma nganh", "ten nganh", "2024", "2025", "hoc phi", "diem chuan",
            "diem trung tuyen", "diem san", "chi tieu", "to hop", "khoi thi",
            "phuong thuc", "xet tuyen", "hoc bong", "ma xet tuyen",
        ]):
            score += 4
            selected_idx.add(idx)
            if idx + 1 < len(lines):
                selected_idx.add(idx + 1)
        score += sum(5 for term in focus_terms if term and term in norm)
        line_tokens = set(re.findall(r"\w+", norm, flags=re.UNICODE))
        score += min(5, len(q_tokens.intersection(line_tokens)))
        if score > 0:
            scored.append((score, idx))
    for _, idx in sorted(scored, key=lambda item: (-item[0], item[1])):
        for j in range(max(0, idx - 1), min(len(lines), idx + 2)):
            selected_idx.add(j)
        candidate = "\n".join(lines[j] for j in sorted(selected_idx))
        if len(candidate) >= max_chars:
            break
    selected = "\n".join(lines[j] for j in sorted(selected_idx))
    if not selected.strip():
        selected = text[:max_chars]
    return selected[:max_chars]


# ──────────────────────────────────────────────────────────────────
# MultiHopOrchestrator
# ──────────────────────────────────────────────────────────────────

def _multi_hop_complexity_score(question: str) -> tuple[int, list[str]]:
    """Fast heuristic gate: simple one-fact questions should stay single-hop."""
    text = _normalize_for_match(question)
    reasons: list[str] = []
    score = 0

    strong_patterns = [
        (r"\btop\s*\d+\b|\b\d+\s+truong\b", "top/list_n"),
        (r"\b(xep hang|ranking|sap xep)\b", "ranking"),
        (r"\b(so sanh|khac nhau|giong nhau|hon|kem hon|versus| vs )\b", "comparison"),
        (r"\b(cao nhat|thap nhat|tot nhat|phu hop nhat|nen chon|goi y|de xuat)\b", "selection"),
        (r"\b(loc|tim cac|nhung truong|cac truong|danh sach|truong nao)\b", "list/filter"),
        (r"\b(vua .+ vua|dong thoi|kem theo|kem)\b", "compound_requirement"),
        (r"\b(trong danh sach tren|cac truong tren|nhom tren)\b", "dependent_reference"),
    ]
    for pattern, reason in strong_patterns:
        if re.search(pattern, text):
            score += 2
            reasons.append(reason)

    data_slots = [
        "diem chuan", "diem trung tuyen", "diem xet tuyen", "diem san",
        "hoc phi", "le phi", "hoc bong", "chi tieu", "ma nganh", "ten nganh",
        "to hop mon", "khoi thi", "phuong thuc xet tuyen", "xet tuyen",
        "hoc ba", "thi thpt", "dgnl", "dgtd", "tuyen thang",
        "ky tuc xa", "dia diem hoc", "co so dao tao", "co hoi viec lam",
        "viec lam", "muc luong", "chuong trinh dao tao", "chuan dau ra",
    ]
    matched_slots = [slot for slot in data_slots if _term_in_text(slot, text)]
    if len(matched_slots) >= 2:
        score += 1
        reasons.append("multiple_data_slots:" + ",".join(matched_slots[:3]))

    school_list_markers = len(re.findall(r"\b(dai hoc|hoc vien|truong)\b", text))
    if school_list_markers >= 2 or re.search(r"\b(giua|voi)\s+.+\s+(va|,)\s+", text):
        score += 1
        reasons.append("multiple_entities")

    if score == 1 and len(matched_slots) <= 1:
        score = 0
        reasons.append("single_fact_reset")

    return score, reasons


class MultiHopOrchestrator:
    def __init__(
        self,
        base_retriever: BaseRetrieverProtocol,
        decomposer: DecomposerProtocol,
        fact_extractor: Optional[FactExtractorProtocol] = None,
        reasoner: Optional[ReasonerProtocol] = None,
        config: Optional[MultiHopConfig] = None,
    ) -> None:
        self.base_retriever = base_retriever
        self.decomposer = decomposer
        self.fact_extractor = fact_extractor
        self.reasoner = reasoner
        self.config = config or MultiHopConfig()
        self.logger = CmdLogger("MultiHop")
        self._original_question: str = ""
        self._dep_evidence_types: dict[int, str] = {}
        self._per_sq_view: dict[int, SubQuestionResult] = {}

    @staticmethod
    def _source_mode(params: "GenerationParams") -> str:
        raw = str(params.get("source_mode", "") or "").lower()  # type: ignore[attr-defined]
        if raw in {"auto", "local", "web", "hybrid"}:
            return raw

        if bool(params.get("auto_source", True)):  # type: ignore[attr-defined]
            return "auto"

        use_local = bool(params.get("use_localdb", False))  # type: ignore[attr-defined]
        use_web = bool(params.get("use_websearch", False))  # type: ignore[attr-defined]
        if use_local and use_web:
            return "hybrid"
        if use_local:
            return "local"
        if use_web:
            return "web"
        return "auto"

    async def retrieve(
        self, question: str, params: "GenerationParams"
    ) -> tuple[list[WebSource], list[RagSource], Optional[MultiHopTrace]]:
        """Trả (web_sources, rag_sources, trace). Trace=None khi fallback single-hop."""
        # Kiểm tra bật / tắt (runtime override)
        enabled = params.get("use_multi_hop", self.config.enabled)  # type: ignore
        if not enabled:
            web, rag = await self.base_retriever.retrieve(question, params)
            return web, rag, None

        auto_gate = bool(params.get("auto_multi_hop", self.config.auto_detect_complexity))  # type: ignore
        force_multi_hop = bool(params.get("force_multi_hop", False))  # type: ignore
        if auto_gate and not force_multi_hop:
            score, reasons = _multi_hop_complexity_score(question)
            threshold = int(params.get("multi_hop_complexity_threshold", self.config.complexity_threshold))  # type: ignore
            if score < threshold:
                self.logger.log(
                    f"[AutoGate] simple question -> single-hop "
                    f"(score={score}, threshold={threshold}, reasons={reasons})"
                )
                web, rag = await self.base_retriever.retrieve(question, params)
                return web, rag, None
            self.logger.log(
                f"[AutoGate] complex question -> multi-hop "
                f"(score={score}, threshold={threshold}, reasons={reasons})"
            )

        # 1) Decompose
        try:
            plan = await self.decomposer.decompose(question, params)
        except Exception as e:
            self.logger.log(f"[Decompose] fail → fallback single-hop ({e})")
            traceback.print_exc()
            web, rag = await self.base_retriever.retrieve(question, params)
            return web, rag, None

        if plan is None or plan.is_trivial:
            self.logger.log("[Decompose] trivial plan → fallback single-hop")
            web, rag = await self.base_retriever.retrieve(question, params)
            return web, rag, None

        self.logger.log(f"[Decompose] {len(plan.sub_questions)} sub-questions")
        for sq in plan.sub_questions:
            self.logger.log(
                f"  SQ#{sq.id} [{sq.resolver}/{sq.evidence_type}] deps={sq.depends_on} | {sq.text}"
            )

        # Cache state cho reasoning step
        self._original_question = question
        self._dep_evidence_types = {sq.id: sq.evidence_type for sq in plan.sub_questions}

        # 2) Execute hops theo topological order
        resolved: dict[int, str] = {}
        per_sq: dict[int, SubQuestionResult] = {}
        hop_count = 0
        # Cập nhật view để _execute_subq truy cập rag_sources của deps khi reasoning.
        self._per_sq_view = per_sq

        for hop in range(self.config.max_hops):
            ready = [
                sq for sq in plan.sub_questions
                if sq.id not in per_sq
                and all(d in resolved for d in sq.depends_on)
            ]
            if not ready:
                break
            hop_count += 1
            self.logger.log(f"[Hop {hop+1}] running {len(ready)} sub-q in parallel")

            tasks = [
                asyncio.create_task(self._execute_subq(sq, resolved, params))
                for sq in ready
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for sq, res in zip(ready, results):
                if isinstance(res, Exception):
                    self.logger.log(f"  SQ#{sq.id} FAILED: {res}")
                    per_sq[sq.id] = SubQuestionResult(
                        sub_id=sq.id, rewritten_text=sq.text, fact="", confidence=0.0
                    )
                    resolved[sq.id] = ""
                else:
                    per_sq[sq.id] = res
                    resolved[sq.id] = res.fact

        # Sub-q chưa chạy (do deps thất bại) — vẫn tạo empty entry
        for sq in plan.sub_questions:
            if sq.id not in per_sq:
                per_sq[sq.id] = SubQuestionResult(sub_id=sq.id, rewritten_text=sq.text)

        # 3) Aggregate evidence, dedup theo (url, chunk_index)
        all_web, all_rag = self._aggregate(per_sq)
        trace = MultiHopTrace(plan=plan, hop_count=hop_count, per_sub_q=per_sq)
        self.logger.log(
            f"[Aggregate] web={len(all_web)} rag={len(all_rag)} hops={hop_count}"
        )
        return all_web, all_rag, trace

    # ──────────────────────────────────────────────────────────────
    async def _execute_subq(
        self,
        sq: SubQuestion,
        resolved: dict[int, str],
        params: "GenerationParams",
    ) -> SubQuestionResult:
        # P2: bridge entity-aware rewrite (skip dep nếu evidence_type là list/comparison/computation)
        rewritten = rewrite_with_resolved(
            sq,
            resolved,
            dep_evidence_types=self._dep_evidence_types,
            skip_types=tuple(self.config.skip_bridge_entity_for_types),
        )

        # ── Reasoning sub-q ──
        if rewritten != sq.text:
            self.logger.log(f"  SQ#{sq.id} rewritten: {rewritten}")

        if sq.resolver == "reasoning":
            return await self._execute_reasoning_subq(sq, resolved, rewritten, params)

        # ── Retrieval sub-q ──
        subq_params = self._scoped_params(sq, params)
        web, rag = await self.base_retriever.retrieve(rewritten, subq_params)
        self.logger.log(
            f"  SQ#{sq.id} retrieve-result: web={len(web)} rag={len(rag)} "
            f"localdb={bool(subq_params.get('use_localdb'))} websearch={bool(subq_params.get('use_websearch'))}"
        )
        for idx, src in enumerate(rag[:3], 1):
            self.logger.log(
                f"    RAG#{idx}: title={(src.get('title') or '')[:80]} "
                f"url={src.get('url', '')} chunk_index={src.get('chunk_index', '')} "
                f"text_chars={len(src.get('text', '') or '')}"
            )

        # Resilient fallback: local_db rỗng → thử web nếu source_mode=auto.
        # Khi user ép source_mode=local/web/hybrid thì tôn trọng setting đó.
        source_mode = self._source_mode(params)
        if (
            source_mode == "auto"
            and sq.resolver == "local_db"
            and not rag
            and self.config.fallback_web_when_local_empty
        ):
            self.logger.log(
                f"  SQ#{sq.id} local_db empty → fallback web"
            )
            fb_params = self._scoped_params(
                SubQuestion(id=sq.id, text=sq.text, depends_on=sq.depends_on, resolver="web"),
                params,
            )
            web, rag = await self.base_retriever.retrieve(rewritten, fb_params)
            self.logger.log(
                f"  SQ#{sq.id} fallback-result: web={len(web)} rag={len(rag)} "
                f"localdb={bool(fb_params.get('use_localdb'))} websearch={bool(fb_params.get('use_websearch'))}"
            )

        # ── Major-keyword filter ──
        # Loại các chunks không đề cập đến ngành/category được hỏi (ví dụ
        # khi router LLM trả về tất cả school_id, có nhiều trường không có CNTT).
        if self.config.enable_major_keyword_filter and rag:
            keywords = _extract_major_keywords(sq.text)
            if keywords:
                before = len(rag)
                rag = filter_chunks_by_major(
                    rag, keywords, keep_minimum=self.config.major_filter_keep_minimum
                )
                after = len(rag)
                if after != before:
                    self.logger.log(
                        f"  SQ#{sq.id} major-filter: {before}→{after} chunks (kw={keywords[:3]})"
                    )

        # Fact extraction (optional) — chỉ chạy khi sub-q có follower trong DAG.
        fact = ""
        confidence = 0.0
        if self.config.enable_fact_extraction and self.fact_extractor is not None and rag:
            try:
                fact, confidence = await self._call_fact_extractor(
                    rewritten, rag, params, sq.evidence_type
                )
                if confidence < self.config.fact_confidence_threshold:
                    fact = ""  # không đủ tin cậy để dùng ở hop sau
            except Exception as e:
                self.logger.log(f"  SQ#{sq.id} fact extraction failed: {e}")

        return SubQuestionResult(
            sub_id=sq.id,
            rewritten_text=rewritten,
            web_sources=web,
            rag_sources=rag,
            fact=fact,
            confidence=confidence,
        )

    async def _call_fact_extractor(
        self,
        sub_q_text: str,
        rag_sources: list[RagSource],
        params: "GenerationParams",
        evidence_type: str,
    ) -> tuple[str, float]:
        """Gọi fact_extractor có truyền evidence_type. Backward-compat: nếu
        adapter cũ không nhận evidence_type, fallback signature 3-arg.
        """
        try:
            return await self.fact_extractor.extract(  # type: ignore[union-attr]
                sub_q_text, rag_sources, params, evidence_type=evidence_type
            )
        except TypeError:
            # Adapter cũ không nhận kwargs evidence_type → gọi 3-arg.
            return await self.fact_extractor.extract(  # type: ignore[union-attr]
                sub_q_text, rag_sources, params
            )

    # ──────────────────────────────────────────────────────────────
    async def _execute_reasoning_subq(
        self,
        sq: SubQuestion,
        resolved: dict[int, str],
        rewritten: str,
        params: "GenerationParams",
    ) -> SubQuestionResult:
        """Reasoning step: gọi LLM (nếu có) để tính/lọc/so sánh trên evidence
        thật từ deps. Fallback pass-through khi không có reasoner.
        """
        # 1) Build evidence blocks từ deps: gồm fact + các chunks rag thật của dep
        evidence_blocks: list[str] = []
        dep_facts: list[tuple[int, str]] = []
        max_chunks = max(1, int(self.config.reasoning_max_evidence_chunks_per_dep))
        max_chars = max(200, int(self.config.reasoning_max_chars_per_chunk))

        for d in sq.depends_on:
            dep_res = self._per_sq_view.get(d) if hasattr(self, "_per_sq_view") else None
            dep_fact = resolved.get(d, "") or ""
            if dep_fact:
                dep_facts.append((d, dep_fact))
            block_parts = [f"### Bằng chứng từ sub-question #{d}"]
            if dep_fact:
                block_parts.append(f"- Tóm tắt fact: {dep_fact}")
            if dep_res and dep_res.rag_sources:
                top = dep_res.rag_sources[:max_chunks]
                for i, c in enumerate(top, 1):
                    title = c.get("title", "") or ""
                    text = _select_relevant_text(
                        f"{self._original_question} {sq.text}",
                        c.get("text", "") or "",
                        max_chars,
                    )
                    if not text.strip():
                        continue
                    block_parts.append(f"- Chunk {i} [{title}]: {text}")
            block = "\n".join(block_parts)
            evidence_blocks.append(block)

        # 2) Gọi reasoner nếu có
        self.logger.log(
            f"  SQ#{sq.id} reasoning-evidence: deps={sq.depends_on} "
            f"blocks={len(evidence_blocks)} chars={sum(len(b) for b in evidence_blocks)} "
            f"dep_facts={len(dep_facts)}"
        )
        answer = ""
        confidence = 0.0
        explanation = ""
        if (
            self.config.enable_llm_reasoning
            and self.reasoner is not None
            and evidence_blocks
        ):
            try:
                answer, confidence = await self.reasoner.reason(
                    self._original_question or sq.text,
                    sq.text,
                    evidence_blocks,
                    params,
                )
                if answer:
                    self.logger.log(
                        f"  SQ#{sq.id} reasoner OK conf={confidence:.2f} "
                        f"answer_preview={answer[:120]}"
                    )
                if answer and confidence < self.config.reasoning_confidence_threshold:
                    self.logger.log(
                        f"  SQ#{sq.id} reasoner below threshold: conf={confidence:.2f} "
                        f"< {self.config.reasoning_confidence_threshold:.2f}; discard"
                    )
                    answer = ""
                    confidence = 0.0
            except Exception as e:
                self.logger.log(f"  SQ#{sq.id} reasoner failed: {e}")
                traceback.print_exc()

        # 3) Fallback nếu reasoner không có / fail
        if not answer and self.config.allow_reasoning_pass_through_fallback:
            answer = " / ".join(f for _, f in dep_facts)
            confidence = 1.0 if answer else 0.0
            explanation = "(fallback: pass-through từ fact của deps)"

        # 4) Materialize synthetic RagSource cho reader thấy kết quả reasoning
        elif not answer:
            self.logger.log(f"  SQ#{sq.id} reasoning produced no trusted answer; no synthetic evidence")

        synthetic_rag: list[RagSource] = []
        if self.config.materialize_reasoning_evidence and answer:
            lines = [f"[Kết quả bước suy luận: {sq.text}]", f"Đáp án: {answer}"]
            if explanation:
                lines.append(f"Lưu ý: {explanation}")
            if dep_facts:
                lines.append("Bằng chứng đầu vào:")
                for d, f in dep_facts:
                    lines.append(f"- Từ sub-question #{d}: {f[:300]}")
            synthetic_text = "\n".join(lines)
            synthetic_rag.append({  # type: ignore[typeddict-item]
                "query": sq.text,
                "url": f"multihop://reasoning/sq_{sq.id}",
                "title": f"Bước suy luận #{sq.id}",
                "text": synthetic_text,
                "chunk_index": 0,
            })

        return SubQuestionResult(
            sub_id=sq.id,
            rewritten_text=rewritten,
            rag_sources=synthetic_rag,
            fact=answer,
            confidence=confidence,
        )

    def _scoped_params(self, sq: SubQuestion, params: "GenerationParams") -> "GenerationParams":
        """Clone params (deep) và ép cấu hình phù hợp cho sub-q.

        Lý do deep-copy: nhiều sub-q chạy song song, một số module (WebRetriever,
        pipeline.retrieve) có ghi thêm field vào params → tránh race condition.

        Các điều chỉnh:
        - resolver-specific flags (use_localdb / use_websearch).
        - ép max_query = subq_max_query (sub-q đã atomic).
        - tắt quality_log để log sạch cho multi-hop.
        - KHÔNG ép use_multi_hop=True để tránh đệ quy lồng decomposer.
        """
        try:
            p = copy.deepcopy(dict(params))  # type: ignore
        except Exception:
            p = dict(params)  # type: ignore

        # Ngăn đệ quy: sub-q không được phép gọi lại Orchestrator.
        p["use_multi_hop"] = False

        source_mode = self._source_mode(p)
        p["source_mode"] = source_mode
        p["auto_source"] = source_mode == "auto"

        # source_mode=auto → resolver của decomposer quyết định nguồn cho sub-q.
        # source_mode=local/web/hybrid → setting của user ép nguồn cho mọi sub-q retrieval.
        if source_mode == "auto":
            if sq.resolver == "local_db":
                p["use_localdb"] = True
                p["use_websearch"] = False
            elif sq.resolver == "web":
                p["use_localdb"] = False
                p["use_websearch"] = True
            elif sq.resolver == "hybrid":
                p["use_localdb"] = True
                p["use_websearch"] = True
            else:
                p["use_localdb"] = True
                p["use_websearch"] = True
        elif source_mode == "local":
            p["use_localdb"] = True
            p["use_websearch"] = False
        elif source_mode == "web":
            p["use_localdb"] = False
            p["use_websearch"] = True
        elif source_mode == "hybrid":
            p["use_localdb"] = True
            p["use_websearch"] = True

        # Sub-q đã là atomic, không cần fan-out thêm.
        if (sq.evidence_type or "").lower() in {"list", "comparison", "computation"}:
            p["max_query"] = max(3, int(p.get("max_query", self.config.subq_max_query)))
        else:
            p["max_query"] = max(1, int(self.config.subq_max_query))
        # Đảm bảo web retrieval có tham số tối thiểu để chạy được.
        p.setdefault("k_pages", 3)
        p.setdefault("k_docs", 5)
        self.logger.log(
            f"  SQ#{sq.id} source-decision: source_mode={source_mode} "
            f"resolver={sq.resolver} -> localdb={bool(p.get('use_localdb'))} "
            f"websearch={bool(p.get('use_websearch'))} max_query={p.get('max_query')}"
        )
        return p  # type: ignore

    def _aggregate(
        self, per_sq: dict[int, SubQuestionResult]
    ) -> tuple[list[WebSource], list[RagSource]]:
        web_seen: set[str] = set()
        rag_seen: set[tuple[str, int]] = set()
        web_out: list[WebSource] = []
        rag_out: list[RagSource] = []
        for res in per_sq.values():
            for w in res.web_sources:
                url = w.get("url", "")
                if url and url not in web_seen:
                    web_seen.add(url)
                    web_out.append(w)
            for r in res.rag_sources:
                key = (r.get("url", ""), r.get("chunk_index", -1))
                if key not in rag_seen:
                    rag_seen.add(key)
                    rag_out.append(r)
        return web_out, rag_out
