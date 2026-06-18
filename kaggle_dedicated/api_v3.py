DOMAIN = "http://127.0.0.1:8000"
DEPLOY_URL = None#"https://uniadmission.me"


IS_LOCAL = DOMAIN == "http://127.0.0.1:8000"
BASE_PATH = "" if IS_LOCAL else "/kaggle/working"

ws_pipeline = None

import os
import platform
import requests
import io
import tarfile
import shutil


def _patch_platform_for_windows() -> None:
    """Avoid Windows WMI hangs in platform helpers."""
    if os.name != "nt":
        return
    arch = os.getenv("PROCESSOR_ARCHITECTURE", "").strip() or "AMD64"
    platform.machine = lambda: arch  # type: ignore[assignment]
    platform.system = lambda: "Windows"  # type: ignore[assignment]


_patch_platform_for_windows()

def unpack_folder(data: bytes, path: str):
    if os.path.exists(path): # Remove old code
        shutil.rmtree(path)
    with io.BytesIO(data) as tar_buffer:
        with tarfile.open(fileobj=tar_buffer, mode='r:gz') as tar:
            tar.extractall(path=path)
def unpack_file(data: bytes, path: str):
    os.makedirs(f"{BASE_PATH}files", exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'wb') as file:
        file.write(data)
def unpack_list(*names: str):
    # if DOMAIN == "http://127.0.0.1:8000": return
    for name in names:
        if "." in name:
            url = f"{DOMAIN}/package/{name}"
        else:
            url = f"{DOMAIN}/package/{name}"
        data = requests.get(url).content
        if "." in name:
            unpack_file(data, f"files/{name}")
        else:
            unpack_folder(data, name)
unpack_list(
    "worker.env", "school_name.json", "school_alias.json","local.pkl"
)

from dotenv import load_dotenv
load_dotenv(f"{BASE_PATH}files/worker.env")

NGROK_PORT = 8002
if DOMAIN != "http://127.0.0.1:8000":
    import subprocess
    subprocess.run(["ngrok", "config", "add-authtoken", os.getenv("NGROK_TOKEN_1", "")])
    subprocess.Popen(["ngrok", "http", str(NGROK_PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

# Replacement for cmd
    
from huggingface_hub import login
print("[Startup] Logging in Hugging Face...", flush=True)
login(token=os.getenv("HUGGING_FACE_TOKEN"))
print("[Startup] Hugging Face login completed.", flush=True)

# cmd = [
#     "hf", "auth", "login",
#     "--token", os.getenv("HUGGING_FACE_TOKEN")
# ]
# import subprocess
# subprocess.run(cmd)
# print("")

print("[Startup] Importing retrieval and server modules...", flush=True)
from data_retriever import *
from data_retriever.test_trace import compact_sources, log_step, preview as trace_preview, reset_trace
from server import *
print("[Startup] Core modules imported.", flush=True)
from school_mapper import SchoolMapper
from typing import Any, AsyncGenerator, NotRequired, Protocol
from typing import Callable, AsyncGenerator
from openai import AsyncOpenAI, OpenAI
from google import genai
from google.genai import types
import os
import pickle
import json
import asyncio
import enum
import traceback
import copy
import re
import unicodedata
from datetime import datetime

from typing import Protocol, AsyncGenerator, TypedDict


_DEBUG_FALSE_VALUES = {"0", "false", "no", "off"}


def _debug_trace_enabled() -> bool:
    return str(os.getenv("BOT_DEBUG_TRACE", "1")).strip().lower() not in _DEBUG_FALSE_VALUES


def _debug_full_prompt_enabled() -> bool:
    """Keep prompt internals opt-in so test logs stay readable."""
    return str(os.getenv("BOT_DEBUG_FULL_PROMPTS", "0")).strip().lower() not in _DEBUG_FALSE_VALUES


def _debug_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _debug_clip(text: Any, limit: int | None = None) -> str:
    value = "" if text is None else str(text)
    if limit is None:
        limit = _debug_int_env("BOT_DEBUG_TEXT_CHARS", 12000)
    if limit <= 0 or len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated {len(value) - limit} chars]"


def _debug_redact(value: Any) -> Any:
    secret_markers = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH")
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if any(marker in key_str.upper() for marker in secret_markers):
                out[key_str] = "[REDACTED]"
            else:
                out[key_str] = _debug_redact(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_debug_redact(item) for item in value]
    return value


def _debug_json(value: Any) -> str:
    try:
        return json.dumps(_debug_redact(value), ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(value)


def _debug_block(title: str, body: Any = "", *, limit: int | None = None) -> None:
    if not _debug_trace_enabled():
        return
    print("\n" + "=" * 80)
    print(f"[{title}]")
    print("=" * 80)
    if body is not None:
        print(_debug_clip(body, limit))


def _debug_source_list(label: str, sources: list[Any]) -> None:
    if not _debug_trace_enabled():
        return
    text_limit = _debug_int_env("BOT_DEBUG_SOURCE_TEXT_CHARS", 0)
    print("\n" + "=" * 80)
    print(f"[{label}] count={len(sources)} text_limit={text_limit} (0=full)")
    print("=" * 80)
    if not sources:
        print("(empty)")
        return
    for idx, source in enumerate(sources, 1):
        text = source.get("text", "") if isinstance(source, dict) else ""
        meta = {
            "index": idx,
            "title": source.get("title", "") if isinstance(source, dict) else "",
            "url": source.get("url", "") if isinstance(source, dict) else "",
            "query": source.get("query", "") if isinstance(source, dict) else "",
            "score": source.get("score", None) if isinstance(source, dict) else None,
            "chunk_index": source.get("chunk_index", None) if isinstance(source, dict) else None,
            "text_chars": len(text or ""),
        }
        print("-" * 80)
        print(_debug_json(meta))
        print("[SOURCE_TEXT]")
        print(_debug_clip(text, text_limit))


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


def _select_relevant_evidence_text(question: str, text: str, max_chars: int) -> str:
    """Keep table headers and rows matching the question instead of truncating the first chars."""
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return text or ""
    focus_terms = _query_focus_terms(question)
    q_tokens = set(re.findall(r"\w+", _normalize_for_match(question), flags=re.UNICODE))
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return text[:max_chars]

    header_idx: set[int] = set()
    scored: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        norm = _normalize_for_match(line)
        score = 0
        if "|" in line and any(h in norm for h in [
            "ma nganh", "ten nganh", "2024", "2025", "hoc phi", "diem chuan",
            "diem trung tuyen", "diem san", "chi tieu", "to hop", "khoi thi",
            "phuong thuc", "xet tuyen", "hoc bong", "ma xet tuyen",
        ]):
            score += 4
            header_idx.add(idx)
        score += sum(5 for term in focus_terms if term and term in norm)
        line_tokens = set(re.findall(r"\w+", norm, flags=re.UNICODE))
        score += min(5, len(q_tokens.intersection(line_tokens)))
        if score > 0:
            scored.append((score, idx, line))

    selected_idx: set[int] = set()
    for idx in sorted(header_idx):
        selected_idx.add(idx)
        if idx + 1 < len(lines):
            selected_idx.add(idx + 1)
    for _, idx, _ in sorted(scored, key=lambda item: (-item[0], item[1])):
        for j in range(max(0, idx - 1), min(len(lines), idx + 2)):
            selected_idx.add(j)
        candidate = "\n".join(lines[j] for j in sorted(selected_idx))
        if len(candidate) >= max_chars:
            break

    selected = "\n".join(lines[j] for j in sorted(selected_idx))
    if not selected.strip():
        selected = text[:max_chars]
    if len(selected) > max_chars:
        selected = selected[:max_chars]
    return selected


def _is_table_like_text(text: str) -> bool:
    return "[BANG]" in (text or "") or (text or "").count("|") >= 6


def _strict_evidence_required(question: str) -> bool:
    """True for queries where an ungrounded answer is worse than no answer."""
    q_norm = _normalize_for_match(question)
    structural_terms = [
        "danh sach", "liet ke", "so sanh", "xep hang", "top", "loc",
        "cao nhat", "thap nhat", "tren", "duoi", "lon hon", "nho hon",
        "kem", "giua", "nhung truong", "cac truong",
    ]
    metric_terms = [
        "diem chuan", "diem trung tuyen", "diem xet tuyen", "diem san",
        "hoc phi", "chi tieu", "ma nganh", "to hop",
    ]
    if any(_term_in_text(term, q_norm) for term in structural_terms):
        return True
    return sum(1 for term in metric_terms if _term_in_text(term, q_norm)) >= 2


def _negative_or_empty_answer(answer: str) -> bool:
    a_norm = _normalize_for_match(answer)
    if not a_norm:
        return True
    negative_terms = [
        "khong co thong tin", "khong tim thay", "chua tim thay",
        "khong du du lieu", "khong du thong tin", "toi khong tim thay",
        "chua duoc cong bo", "chua cong bo", "chua co thong tin",
        "chua duoc neu", "chua neu", "chua duoc de cap", "chua de cap",
        "khong neu", "khong duoc neu", "khong duoc de cap",
        "khong duoc cong bo", "not published",
        "unknown", "none", "n/a",
    ]
    return any(term in a_norm for term in negative_terms)


def _split_answer_entries(answer: str) -> list[str]:
    entries = re.split(r"\s*(?:;|\n|\r|\u2022)\s*", answer or "")
    return [entry.strip(" -\t") for entry in entries if len(entry.strip(" -\t")) >= 4]


def _calibrate_fact_confidence(
    answer: str,
    confidence: float,
    evidence_type: str,
    question: str,
    context: str = "",
) -> float:
    """Cap confidence when the extractor returned an unsupported-looking fact."""
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except Exception:
        confidence = 0.0
    if _negative_or_empty_answer(answer):
        return min(confidence, 0.15)

    q_norm = _normalize_for_match(question)
    et = (evidence_type or "factual").lower()
    metric_query = any(
        _term_in_text(term, q_norm)
        for term in [
            "diem chuan", "diem trung tuyen", "diem xet tuyen", "diem san",
            "hoc phi", "chi tieu",
        ]
    )
    if et in {"numeric", "comparison", "computation"} and metric_query and not re.search(r"\d", answer):
        return min(confidence, 0.25)

    context_norm = _normalize_for_match(context)
    if context_norm:
        asks_grad = any(term in q_norm for term in ["thac si", "cao hoc", "sau dai hoc", "tien si"])
        has_grad_context = any(term in context_norm for term in ["thac si", "cao hoc", "sau dai hoc", "nghien cuu sinh", "tien si"])
        has_undergrad_context = any(term in context_norm for term in ["dai hoc chinh quy", "bac dai hoc", "trinh do dai hoc", "sinh vien dai hoc"])
        if not asks_grad and has_grad_context and not has_undergrad_context:
            return min(confidence, 0.15)

        if et in {"numeric", "comparison", "computation"} and metric_query:
            answer_numbers = [_normalize_number_token(x) for x in re.findall(r"\d+(?:[,.]\d+)?", answer or "")]
            missing = [
                n for n in answer_numbers
                if n and n not in context_norm and n.replace(".", ",") not in context_norm
            ]
            if missing:
                return min(confidence, 0.25)

        if any(term in q_norm for term in ["diem chuan", "diem trung tuyen", "diem xet tuyen"]):
            explicit_method = any(term in q_norm for term in [
                "hoc ba", "ccqt", "ket hop", "dgnl", "danh gia nang luc",
                "dgtd", "danh gia tu duy", "tsa", "hsa", "aptitude",
            ])
            plain_a00_query = ("a00" in q_norm or "khoi a" in q_norm) and not explicit_method
            non_thpt_context = any(term in context_norm for term in [
                "hoc ba", "ccqt", "ket hop", "xet tuyen ket hop",
                "dgnl", "danh gia nang luc", "dgtd", "danh gia tu duy",
                "tsa", "hsa", "aptitude",
            ])
            thpt_score_signal = any(term in context_norm for term in [
                "diem thi thpt", "thi tot nghiep thpt", "tot nghiep thpt",
                "xet tuyen thpt", "phuong thuc thpt",
            ])
            if plain_a00_query and non_thpt_context and not thpt_score_signal:
                return min(confidence, 0.20)

    if _term_in_text("hoc phi", q_norm):
        a_norm = _normalize_for_match(answer)
        asks_waiver = any(term in q_norm for term in ["mien", "hoc bong", "ho tro"])
        zero_like = re.search(r"\b0(?:[,.]0+)?\b", a_norm) or "khong phai dong hoc phi" in a_norm
        if zero_like and not asks_waiver:
            return min(confidence, 0.25)
        if context_norm:
            waiver_hits = sum(1 for term in [
                "mien giam", "hoc bong", "ho tro chi phi", "chinh sach phat trien",
                "nguoi hoc tai nang", "tro cap", "cap bu hoc phi",
            ] if term in context_norm)
            schedule_signal = any(term in context_norm for term in [
                "muc thu hoc phi", "dinh muc hoc phi", "thong bao hoc phi",
                "quy dinh hoc phi", "don gia hoc phi", "hoc phi nam hoc",
                "dong/tin chi", "dong / tin chi",
            ])
            if waiver_hits and not asks_waiver and not schedule_signal:
                return min(confidence, 0.20)

    if et == "list" and _strict_evidence_required(question):
        entries = _split_answer_entries(answer)
        if len(entries) < 2 and any(_term_in_text(t, q_norm) for t in ["danh sach", "cac truong", "nhung truong", "top"]):
            return min(confidence, 0.25)
    return confidence


def _normalize_number_token(token: str) -> str:
    return (token or "").strip().replace(",", ".")


def _number_token_supported(token: str, evidence_norm: str) -> bool:
    token = (token or "").strip().strip(".,;:")
    if not token:
        return True

    direct_variants = {
        token,
        token.replace(",", "."),
        token.replace(".", ","),
        token.replace(",", ""),
        token.replace(".", ""),
    }
    for variant in direct_variants:
        if variant and variant in evidence_norm:
            return True

    compact = re.sub(r"[,.]", "", token)
    try:
        value = int(compact)
    except Exception:
        return False

    if value >= 1_000_000:
        million = value / 1_000_000
        million_variants = {
            f"{million:g} trieu",
            f"{million:g}trieu",
            f"{million:g} tr",
        }
        return any(variant in evidence_norm for variant in million_variants)
    return False


def _reasoner_answer_grounded(answer: str, evidence: str, question: str) -> tuple[bool, str]:
    """Guard against fabricated numeric filters/rankings from reasoning hops."""
    if _negative_or_empty_answer(answer):
        return True, "empty_or_negative"
    if not _strict_evidence_required(question):
        return True, "not_strict"

    evidence_norm = _normalize_for_match(f"{evidence}\n{question}")
    answer_numbers = re.findall(r"\d[\d.,]*", answer or "")
    missing_numbers = [
        number for number in answer_numbers
        if not _number_token_supported(number, evidence_norm)
    ]
    if missing_numbers:
        return False, f"numbers_not_in_evidence={missing_numbers[:5]}"

    entries = _split_answer_entries(answer)
    if not entries:
        return False, "no_entries"
    return True, "ok"


def _extract_admission_table_evidence(question: str, text: str, max_chars: int = 12000) -> str:
    if not text or max_chars <= 0 or not _is_table_like_text(text):
        return ""
    q_norm = _normalize_for_match(question)
    score_query = any(
        _term_in_text(term, q_norm)
        for term in ["diem chuan", "diem trung tuyen", "diem xet tuyen", "diem san"]
    )
    tuition_query = any(
        _term_in_text(term, q_norm)
        for term in [
            "hoc phi", "muc thu", "tin chi", "trieu",
            "dong/nam", "dong / nam", "dong/thang", "dong / thang",
        ]
    )
    broad_list_query = any(
        _term_in_text(term, q_norm)
        for term in ["danh sach", "so sanh", "xep hang", "top", "loc", "cac truong", "nhung truong"]
    )
    if not (score_query or tuition_query or broad_list_query):
        return ""

    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""

    header_terms = [
        "diem chuan", "diem trung tuyen", "diem xet tuyen", "diem san",
        "ma xet tuyen", "ma nganh", "ten nganh", "nganh dao tao", "to hop",
        "hoc phi", "muc thu", "tin chi", "don gia", "dong/nam", "dong / nam",
        "dong/thang", "dong / thang", "trieu", "nam hoc", "hoc ky",
    ]
    start_idx: int | None = None
    # Prefer the actual table header over a generic [BANG] marker. Many crawled
    # pages contain decorative/image tables before the PDF table, and starting
    # too early can cut off the rows that the reader needs.
    for idx, line in enumerate(lines):
        norm = _normalize_for_match(line)
        is_header = "|" in line and any(_term_in_text(term, norm) for term in header_terms)
        if is_header:
            start_idx = max(0, idx - 4)
            break

    if start_idx is None:
        for idx, line in enumerate(lines):
            norm = _normalize_for_match(line)
            if "[bang]" in norm:
                start_idx = max(0, idx - 2)
                break

    if start_idx is None:
        table_rows = [i for i, line in enumerate(lines) if "|" in line and line.count("|") >= 2]
        if len(table_rows) < 3:
            return ""
        start_idx = max(0, table_rows[0] - 3)

    selected: list[str] = []
    for line in lines[start_idx:]:
        candidate = "\n".join(selected + [line])
        if len(candidate) > max_chars:
            break
        selected.append(line)
    return "\n".join(selected).strip()


def _augment_rag_with_web_table_evidence(
    question: str,
    web_sources: list[WebSource],
    rag_sources: list[RagSource],
    *,
    max_sources: int = 2,
    max_chars: int = 12000,
) -> list[RagSource]:
    """Rescue exact admissions tables from web_sources when chunk ranking misses them."""
    if not web_sources:
        return rag_sources
    existing_table_by_url: dict[str, str] = {}
    for rag in rag_sources:
        url = rag.get("url", "")
        text = rag.get("text", "") or ""
        if url and _is_table_like_text(text):
            existing_table_by_url[url] = existing_table_by_url.get(url, "") + "\n" + text

    additions: list[RagSource] = []
    for idx, source in enumerate(web_sources, 1):
        url = source.get("url", "")
        if not url:
            continue
        table_text = _extract_admission_table_evidence(
            question, source.get("text", "") or "", max_chars=max_chars
        )
        if not table_text:
            continue
        existing_text = existing_table_by_url.get(url, "")
        if existing_text:
            existing_norm = _normalize_for_match(existing_text)
            table_norm = _normalize_for_match(table_text)
            existing_ratio = len(existing_text) / max(1, len(table_text))
            key_terms = [
                "dong/thang", "dong / thang", "dong/tin chi", "dong / tin chi",
                "hoc lai", "cai thien", "bang kep", "sau dai hoc", "thac si", "tien si",
            ]
            missing_key_terms = [
                term for term in key_terms
                if term in table_norm and term not in existing_norm
            ]
            if existing_ratio >= 0.8 and not missing_key_terms:
                continue
        additions.append({
            "chunk_index": -100000 - idx,
            "query": question,
            "title": source.get("title", ""),
            "url": url,
            "text": table_text,
            "content_type": "table",
        })
        if len(additions) >= max_sources:
            break
    if additions:
        print(
            f"[ReaderContextRescue] Added {len(additions)} table chunk(s) from web_sources "
            "because selected RAG missed admissions tables"
        )
        _debug_source_list("READER_RESCUED_TABLE_CHUNKS", additions)
    return additions + rag_sources


def _question_metric_kind(text: str) -> str:
    norm = _normalize_for_match(text)
    has_tuition = any(
        _term_in_text(term, norm)
        for term in ["hoc phi", "muc thu", "dong/tin chi", "dong/thang", "dong/nam", "trieu"]
    )
    has_score = any(
        _term_in_text(term, norm)
        for term in ["diem chuan", "diem trung tuyen", "diem xet tuyen", "diem san", "khoi a00", "a00"]
    )
    if has_tuition and not has_score:
        return "tuition"
    if has_score and not has_tuition:
        return "score"
    if has_tuition:
        return "tuition"
    if has_score:
        return "score"
    return ""


def _expected_metrics(question: str) -> list[str]:
    norm = _normalize_for_match(question)
    out: list[str] = []
    if any(_term_in_text(term, norm) for term in ["diem chuan", "diem trung tuyen", "diem xet tuyen", "diem san"]):
        out.append("score")
    if any(_term_in_text(term, norm) for term in ["hoc phi", "muc thu"]):
        out.append("tuition")
    return out


_SCHOOL_ALIAS_MAP: dict[str, list[str]] = {
    "UET": ["uet", "dai hoc cong nghe", "truong dai hoc cong nghe", "dhqghn", "vnu-uet"],
    "PTIT": ["ptit", "hoc vien cong nghe buu chinh vien thong", "buu chinh vien thong"],
    "HUST": ["hust", "dai hoc bach khoa ha noi", "bach khoa ha noi"],
    "HAUI": ["haui", "dai hoc cong nghiep ha noi", "cong nghiep ha noi"],
    "UTT": ["utt", "dai hoc cong nghe giao thong van tai", "cong nghe giao thong van tai"],
    "USTH": ["usth", "dai hoc khoa hoc va cong nghe ha noi"],
}


def _school_label_from_text(text: str) -> str:
    norm = _normalize_for_match(text)
    for label, aliases in _SCHOOL_ALIAS_MAP.items():
        if any(_term_in_text(alias, norm) for alias in aliases):
            return label
    return ""


def _expected_schools(question: str) -> list[str]:
    norm = _normalize_for_match(question)
    schools = [
        label
        for label, aliases in _SCHOOL_ALIAS_MAP.items()
        if any(_term_in_text(alias, norm) for alias in aliases)
    ]
    return schools


def _metric_answer_valid(metric: str, answer: str) -> bool:
    if _negative_or_empty_answer(answer):
        return False
    norm = _normalize_for_match(answer)
    if not re.search(r"\d", norm):
        return False
    money_signal = bool(
        re.search(r"\d[\d.,]*(?:\s*[-–]\s*\d[\d.,]*)?\s*(trieu|dong|vnd|vnđ)", norm)
        or _term_in_text("dong/tin chi", norm)
        or _term_in_text("dong/thang", norm)
        or _term_in_text("dong/nam", norm)
    )
    score_signal = _term_in_text("diem", norm) or bool(re.search(r"\b(?:1[5-9]|2[0-9]|30)(?:[,.]\d{1,2})?\b", norm))
    if metric == "tuition":
        return money_signal
    if metric == "score":
        return score_signal and not money_signal
    return True


def _line_matches_query_focus(line: str, query: str) -> int:
    line_norm = _normalize_for_match(line)
    query_norm = _normalize_for_match(query)
    score = 0
    focus_terms = [
        "khoa hoc may tinh",
        "cong nghe thong tin",
        "ky thuat may tinh",
        "tri tue nhan tao",
        "an toan thong tin",
        "he thong thong tin",
    ]
    for term in focus_terms:
        if _term_in_text(term, query_norm) and _term_in_text(term, line_norm):
            score += 5
    for label, aliases in _SCHOOL_ALIAS_MAP.items():
        if any(_term_in_text(alias, query_norm) for alias in aliases):
            if any(_term_in_text(alias, line_norm) for alias in aliases):
                score += 3
    if _term_in_text("a00", query_norm) and _term_in_text("a00", line_norm):
        score += 2
    if "|" in line:
        score += 1
    return score


def _extract_score_from_rag(query: str, rag_sources: list[RagSource]) -> str:
    best: tuple[int, float, str] | None = None
    for src in rag_sources[:12]:
        text = src.get("text", "") or ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or not re.search(r"\d", line):
                continue
            line_norm = _normalize_for_match(line)
            if not any(_term_in_text(term, line_norm) for term in ["diem chuan", "diem trung tuyen", "diem thi", "a00", "khoa hoc may tinh", "cong nghe thong tin"]):
                continue
            focus_score = _line_matches_query_focus(line, query)
            if focus_score <= 0:
                continue
            values: list[float] = []
            for token in re.findall(r"(?<!\d)(?:1[5-9]|2[0-9]|30)(?:[,.]\d{1,2})?(?!\d)", line_norm):
                try:
                    values.append(float(token.replace(",", ".")))
                except Exception:
                    pass
            if not values:
                continue
            value = max(values)
            candidate = (focus_score, value, line)
            if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] > best[1]):
                best = candidate
    if best is None:
        return ""
    value = best[1]
    return f"{value:g} diem"


def _extract_tuition_from_rag(query: str, rag_sources: list[RagSource]) -> str:
    money_re = re.compile(
        r"\d[\d.,]*(?:\s*[-–]\s*\d[\d.,]*)?\s*(?:trieu|dong|vnd|vnđ)(?:\s*/?\s*(?:nam|thang|tin chi|hoc ky))?",
        re.IGNORECASE,
    )
    best: tuple[int, str] | None = None
    for src in rag_sources[:12]:
        text = src.get("text", "") or ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line_norm = _normalize_for_match(line)
            if not money_re.search(line_norm):
                continue
            if _term_in_text("diem chuan", line_norm) and not _term_in_text("hoc phi", line_norm):
                continue
            focus_score = _line_matches_query_focus(line, query)
            if _term_in_text("hoc phi", line_norm) or _term_in_text("muc thu", line_norm):
                focus_score += 3
            if focus_score <= 0:
                continue
            candidate = (focus_score, line[:500])
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best[1] if best else ""


def _extract_checked_metric_fact(metric: str, query: str, result: Any) -> tuple[str, str]:
    raw_fact = (getattr(result, "fact", "") or "").strip()
    if raw_fact and _metric_answer_valid(metric, raw_fact):
        return raw_fact, "fact_extractor"
    rag_sources = getattr(result, "rag_sources", []) or []
    if metric == "score":
        parsed = _extract_score_from_rag(query, rag_sources)
    elif metric == "tuition":
        parsed = _extract_tuition_from_rag(query, rag_sources)
    else:
        parsed = ""
    if parsed and _metric_answer_valid(metric, parsed):
        return parsed, "parsed_from_rag"
    return "", "missing_or_unit_mismatch"


def _build_multihop_checked_context(question: str, trace: Any) -> tuple[list[RagSource], dict[str, Any]]:
    """Create a guarded synthetic context from multi-hop slots.

    This keeps the final reader from filling a missing tuition slot with a
    nearby admission score, and gives list/filter queries a deterministic
    "not enough evidence" signal when no tuition/score intersection exists.
    """
    if trace is None:
        return [], {"enabled": False}
    plan = getattr(trace, "plan", None)
    if plan is None or not getattr(plan, "sub_questions", None):
        return [], {"enabled": False}
    metrics = _expected_metrics(question)
    if not metrics:
        return [], {"enabled": False}

    q_norm = _normalize_for_match(question)
    is_comparison = any(_term_in_text(term, q_norm) for term in ["so sanh", "giua", "khac nhau"])
    is_filter_list = any(_term_in_text(term, q_norm) for term in ["danh sach", "loc", "cac truong", "nhung truong", "top", "xep hang"])

    slots: dict[str, dict[str, dict[str, str]]] = {}
    reasoning_fact = ""
    for sq in plan.sub_questions:
        res = trace.per_sub_q.get(sq.id)
        if not res:
            continue
        if sq.resolver == "reasoning":
            if (res.fact or "").strip():
                reasoning_fact = (res.fact or "").strip()
            continue
        metric = _question_metric_kind(f"{sq.text} {getattr(res, 'rewritten_text', '')}")
        if metric not in metrics:
            continue
        school_text = f"{sq.text} {getattr(res, 'rewritten_text', '')} "
        for src in (getattr(res, "rag_sources", []) or [])[:3]:
            school_text += f" {src.get('title', '')} {src.get('url', '')}"
        school = _school_label_from_text(school_text) or f"SQ#{sq.id}"
        fact, source = _extract_checked_metric_fact(metric, getattr(res, "rewritten_text", sq.text), res)
        slots.setdefault(school, {})[metric] = {
            "value": fact,
            "source": source,
            "sub_question": str(sq.id),
        }

    expected_schools = _expected_schools(question)
    if not expected_schools:
        expected_schools = [k for k in slots if not k.startswith("SQ#")]
    if not expected_schools and is_filter_list:
        expected_schools = sorted(slots.keys())

    missing: list[str] = []
    lines: list[str] = [
        "[Bang fact da kiem tra tu multi-hop]",
        "Quy tac: diem chuan phai co don vi diem; hoc phi phai co don vi tien. Khong dung diem chuan thay cho hoc phi.",
    ]
    if is_comparison and expected_schools:
        header = "| Truong | " + " | ".join("Diem chuan" if m == "score" else "Hoc phi" for m in metrics) + " | Trang thai |"
        sep = "|---" * (len(metrics) + 2) + "|"
        lines.extend([header, sep])
        for school in expected_schools:
            row_values: list[str] = []
            row_missing: list[str] = []
            for metric in metrics:
                item = slots.get(school, {}).get(metric, {})
                value = item.get("value", "")
                if value:
                    row_values.append(value)
                else:
                    label = "diem chuan" if metric == "score" else "hoc phi"
                    row_values.append("CHUA TIM THAY TRONG NGUON")
                    row_missing.append(label)
                    missing.append(f"{school}:{metric}")
            status = "du thong tin" if not row_missing else "thieu " + ", ".join(row_missing)
            lines.append(f"| {school} | " + " | ".join(row_values) + f" | {status} |")
    elif is_filter_list:
        reasoning_norm = _normalize_for_match(reasoning_fact)
        has_score_value = bool(re.search(r"\b(?:1[5-9]|2[0-9]|30)(?:[,.]\d{1,2})?\b", reasoning_norm))
        has_money_value = bool(
            re.search(r"\d[\d.,]*\s*(?:trieu|dong|vnd|vnđ)", reasoning_norm)
            or "dong/tin chi" in reasoning_norm
            or "dong/thang" in reasoning_norm
            or "dong/nam" in reasoning_norm
        )
        invalid_reasoning = (
            _negative_or_empty_answer(reasoning_fact)
            or any(term in reasoning_norm for term in ["unknown", "khong co thong tin", "khong duoc cung cap", "chua tim thay"])
            or ("score" in metrics and not has_score_value)
            or ("tuition" in metrics and not has_money_value)
        )
        if reasoning_fact and not invalid_reasoning:
            lines.append("Ket qua reasoning co bang chung:")
            lines.append(reasoning_fact)
        else:
            missing.append("final_filtered_list")
            lines.append(
                "CHUA DU BANG CHUNG DE LOC DANH SACH: can ca diem chuan va hoc phi hop le cho tung truong."
            )
    else:
        for school, by_metric in slots.items():
            for metric, item in by_metric.items():
                label = "diem chuan" if metric == "score" else "hoc phi"
                value = item.get("value") or "CHUA TIM THAY TRONG NGUON"
                if not item.get("value"):
                    missing.append(f"{school}:{metric}")
                lines.append(f"- {school} / {label}: {value}")

    if not slots and not reasoning_fact:
        return [], {"enabled": False}

    text = "\n".join(lines)
    chunk: RagSource = {
        "query": question,
        "url": "multihop://checked_slots",
        "title": "Bang fact multi-hop da kiem tra",
        "text": text,
        "chunk_index": -200000,
    }  # type: ignore[assignment]
    report = {
        "enabled": True,
        "metrics": metrics,
        "schools": expected_schools,
        "missing": missing,
        "partial": bool(missing) and is_comparison,
        "hard_no_answer": bool(missing) and is_filter_list,
        "summary": text,
    }
    return [chunk], report


class KeywordInfo(TypedDict):
    query: str
    priority: float
    info: str
    school: str
class KeywordModelProtocol(Protocol):
    async def keywords(self, question: str, params: GenerationParams, threshold: float = 0.5) -> list[KeywordInfo]: ...
class RouterModelProtocol(Protocol):
    async def route(self, question: str, params: GenerationParams) -> list[dict]: ...
    
MODEL_ID = "Qwen/Qwen3-4B"
# Retriever config
search_config = WebsearchConfig(
    page_timeout=15,
    file_timeout=15,
)
rag_config = RagConfig(
    embedding_name="intfloat/multilingual-e5-small",
    device="cpu"
)
splitter_config = SplitterConfig(
    tokenizer_name=MODEL_ID,
    chunk_size=512,
    chunk_overlap=0,
    device="cpu"
)
table_merge_config = MergeTableConfig(
    k_max_previous=5,
    k_max_next=5
)
neighbor_config = MergeNeighborConfig(
    k_previous_chunks=1,
    k_next_chunks=1
)
# Sampling Params
PAGE_RERANKER_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 4096
}
KEYWORDS_PARAMS = {
    "temperature": 0.5,
    "top_p": 0.9,
    "max_tokens": 4096
}
ROUTER_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 1024
}
DECOMPOSER_PARAMS = {
    "temperature": 0.3,  # low temp → output JSON ổn định
    "top_p": 0.9,
    "max_tokens": 1024
}
FACT_EXTRACTOR_PARAMS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 256
}
LIST_FACT_EXTRACTOR_PARAMS = {
    "temperature": 0.1,
    "top_p": 0.9,
    "max_tokens": 1024,  # cao hơn vì cần liệt kê đầy đủ entries
}
REASONER_PARAMS = {
    "temperature": 0.1,
    "top_p": 0.9,
    "max_tokens": 1024,  # đủ cho 1 câu trả lời tổng hợp + danh sách
}
MODELS: list[ModelInfo] = [
    {
        "name": "GPT 4o mini",
        "id": "gpt-4o-mini"
    },
    {
        "name": "Gemini 2.5 flash",
        "id": "gemini-2.5-flash"
    }
]
CLIENT_INFO: WorkerServerInfo = {
    "name": "Test API",
    "domain": "http://127.0.0.1:8002", # Auto change when run with ngrok
    "models": MODELS
}

from instruction import *

class LocalRetriever:
    """Search in static db"""
    def __init__(self) -> None:
        with open(f"{BASE_PATH}files/local.pkl", 'rb') as file:
            self.all_docs = pickle.load(file)
    def _filter_docs(self, school_id: str, section: str) -> list:
        school_docs = [doc for doc in self.all_docs if doc.metadata.get("school_id") == school_id]
        if school_docs:
            filtered_docs = [doc for doc in school_docs if doc.metadata.get("section") == section]
        else:
            filtered_docs = []
        return filtered_docs
    def retrieve(self, keywords: list[dict], params: GenerationParams | None = None) -> tuple[list[WebSource], list[RagSource]]:
        web_sources: list[WebSource] = []
        rag_sources: list[RagSource] = []
        log_step(
            "LOCAL_DB",
            "INPUT",
            {
                "query_count": len(keywords),
                "queries": keywords,
            },
            params or {},
        )
        try:
            for idx, kw in enumerate(keywords):
                school_id = kw.get("school_id")
                section = kw.get("section")
                if school_id and section:
                    docs = self._filter_docs(school_id, section)
                    title = f"Tìm trường ĐH-CĐ - Cốc Cốc ({school_id})"
                    combined_content = "\n\n".join([doc.page_content for doc in docs])
                    description = combined_content[:100] + "..." if len(combined_content) > 100 else combined_content
                    source_url = f"https://hoctap.coccoc.com/tim-truong-dh-cd#{school_id}-{section}"
                    web_source: WebSource = {
                        "query": f"{school_id}:{section}",
                        "title": title,
                        "description": description,
                        "url": source_url,
                        "text": combined_content,
                        "files": [],
                        "score": 1
                    }
                    rag_source: RagSource = {
                        "chunk_index": idx,
                        "query": f"{school_id}:{section}",
                        "title": title,
                        "url": source_url,
                        "text": combined_content,
                    }
                    web_sources.append(web_source)
                    rag_sources.append(rag_source)
        except:
            traceback.print_exc()
        _debug_source_list("LOCAL_WEB_SOURCES", web_sources)
        _debug_source_list("LOCAL_RAG_CHUNKS", rag_sources)
        log_step(
            "LOCAL_DB",
            "OUTPUT",
            {
                "web_sources": len(web_sources),
                "rag_sources": len(rag_sources),
                "web_preview": compact_sources(web_sources, "web", limit=5),
                "rag_preview": compact_sources(rag_sources, "rag", limit=5),
            },
            params or {},
        )
        return web_sources, rag_sources
        
class WebRetriever:
    """Search in web"""
    def __init__(self, llm_ranker: PageRerankModelProtocol, llm_keywords: KeywordModelProtocol) -> None:
        self.pipeline = DataRetrieverPipeline(
            llm_ranker,
            websearch_config=search_config,
            rag_config=rag_config,
            splitter_config=splitter_config,
            neighbor_merge_config=neighbor_config,
            table_merge_config=table_merge_config
        )
        self.llm_keywords = llm_keywords
        self.school_mapper = SchoolMapper(f"{BASE_PATH}files/school_name.json")

    async def start(self):
        """Initialize websearch"""
        await self.pipeline.start()

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE))

    def _domain_trust(self, url: str) -> float:
        from urllib.parse import urlparse

        domain = (urlparse(url).netloc or "").lower()
        if not domain:
            return 0.0
        if domain.endswith(".edu.vn") or domain.endswith(".gov.vn"):
            return 1.0
        if ".edu" in domain or ".gov" in domain:
            return 0.9
        if domain.endswith(".org"):
            return 0.75
        if any(bad in domain for bad in ["forum", "blogspot", "wordpress"]):
            return 0.35
        return 0.6

    def _relevance_score(self, question: str, title: str, description: str, url: str) -> float:
        query_tokens = self._tokenize(question)
        text_tokens = self._tokenize(f"{title} {description} {url}")
        if not query_tokens:
            return 0.0
        overlap = len(query_tokens.intersection(text_tokens))
        return overlap / max(1, len(query_tokens))

    def _relevance_score_with_text(self, question: str, title: str, description: str, url: str, text: str) -> float:
        query_tokens = self._tokenize(_normalize_for_match(question))
        text_tokens = self._tokenize(_normalize_for_match(f"{title} {description} {url} {text[:4000]}"))
        if not query_tokens:
            return 0.0
        return len(query_tokens.intersection(text_tokens)) / max(1, len(query_tokens))

    def _content_richness_score(self, text: str) -> float:
        text = text or ""
        token_count = len(self._tokenize(text))
        score = 0.0
        if token_count >= 40:
            score += 0.35
        elif token_count >= 20:
            score += 0.2
        if _is_table_like_text(text):
            score += 0.30
        if re.search(r"\b(?:19|20)\d{2}\b", text) or re.search(r"\d+(?:[,.]\d+)?", text):
            score += 0.20
        if len(text) >= 800:
            score += 0.15
        return min(1.0, score)

    def _education_level_mismatch_reason(self, question: str, candidate_text: str) -> str | None:
        question_norm = _normalize_for_match(question)
        candidate_norm = _normalize_for_match(candidate_text)
        graduate_terms = [
            "thac si", "cao hoc", "sau dai hoc", "nghien cuu sinh",
            "tien si", "dao tao thac si", "dao tao tien si",
            "master", "masters", "graduate", "postgraduate", "phd",
        ]
        if any(term in candidate_norm for term in graduate_terms) and not any(term in question_norm for term in graduate_terms):
            return "graduate_source_for_undergraduate_query"
        return None

    def _metric_mismatch_reason(self, question: str, candidate_text: str) -> str | None:
        question_norm = _normalize_for_match(question)
        candidate_norm = _normalize_for_match(candidate_text)
        score_like_query = any(term in question_norm for term in ["diem chuan", "diem trung tuyen", "diem xet tuyen"])
        tuition_like_query = any(term in question_norm for term in ["hoc phi", "tuition", "muc thu", "dong hoc phi"])
        plain_a00_query = (
            score_like_query
            and ("a00" in question_norm or "khoi a" in question_norm)
            and not any(
                term in question_norm
                for term in [
                    "hoc ba", "ccqt", "ket hop", "dgnl", "danh gia nang luc",
                    "dgtd", "danh gia tu duy", "tsa", "hsa", "aptitude",
                ]
            )
        )
        if plain_a00_query:
            non_thpt_method = any(
                term in candidate_norm
                for term in [
                    "hoc ba", "ccqt", "ket hop", "xet tuyen ket hop",
                    "dgnl", "danh gia nang luc", "dgtd", "danh gia tu duy",
                    "tsa", "hsa", "aptitude",
                ]
            )
            thpt_score_signal = any(
                term in candidate_norm
                for term in [
                    "diem thi thpt", "thi tot nghiep thpt", "tot nghiep thpt",
                    "xet tuyen thpt", "phuong thuc thpt",
                ]
            )
            if non_thpt_method and not thpt_score_signal:
                return "non_thpt_score_method_for_plain_a00_query"

        if tuition_like_query:
            asks_policy = any(term in question_norm for term in ["mien giam", "hoc bong", "ho tro", "tro cap"])
            waiver_terms = [
                "mien giam", "hoc bong", "ho tro chi phi", "chinh sach phat trien",
                "nguoi hoc tai nang", "tro cap", "cap bu hoc phi",
            ]
            schedule_terms = [
                "muc thu hoc phi", "dinh muc hoc phi", "thong bao hoc phi",
                "quy dinh hoc phi", "don gia hoc phi", "hoc phi nam hoc",
                "dong/tin chi", "dong / tin chi",
            ]
            waiver_hits = sum(1 for term in waiver_terms if term in candidate_norm)
            schedule_signal = any(term in candidate_norm for term in schedule_terms)
            if waiver_hits and not asks_policy and not schedule_signal:
                return "waiver_support_source_for_tuition_query"

        metric_groups = [
            ("tuition", ["hoc phi", "tuition", "muc thu", "dong hoc phi"]),
            ("admission_score", ["diem chuan", "diem trung tuyen", "diem xet tuyen"]),
            ("floor_score", ["diem san", "diem nhan ho so", "nguong dau vao"]),
            ("quota", ["chi tieu"]),
        ]
        required: list[tuple[str, list[str]]] = [
            (metric_name, terms)
            for metric_name, terms in metric_groups
            if any(term in question_norm for term in terms)
        ]
        if not required:
            return None
        if any(any(term in candidate_norm for term in terms) for _, terms in required):
            return None
        if _is_table_like_text(candidate_text) and re.search(r"\d", candidate_text or ""):
            return None
        return "missing_metric:" + ",".join(metric_name for metric_name, _ in required)

    def _canonical_url(self, url: str) -> str:
        from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

        if not url:
            return ""
        parsed = urlparse(url)
        filtered_query = [
            (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=False)
            if not k.lower().startswith("utm_")
        ]
        normalized = parsed._replace(path=(parsed.path.rstrip("/") or "/"), query=urlencode(filtered_query), fragment="")
        return urlunparse(normalized)

    def _source_safety_reason(self, question: str, source: dict) -> str | None:
        from urllib.parse import urlparse

        title = source.get("title", "") or ""
        url = source.get("url", "") or ""
        description = source.get("description", "") or ""
        text = source.get("text", "") or ""
        domain = (urlparse(url).netloc or "").lower()
        merged = _normalize_for_match(f"{title} {description} {url} {text[:3000]}")
        question_norm = _normalize_for_match(question)

        hard_bad_domains = [
            "vieclam", "topcv", "career", "jobs", "jobstreet", "timviec",
            "viec-lam", "vieclamtot", "123job", "itviec",
        ]
        job_terms = [
            "mong muốn tìm việc", "tìm việc", "việc làm", "tuyển dụng",
            "nhân viên", "thực tập sinh", "trưởng phòng nhân sự", "e-commerce executive",
        ]
        admission_terms = [
            "điểm chuẩn", "học phí", "tuyển sinh", "xét tuyển", "ngành đào tạo",
            "admission", "tuition",
        ]
        asks_admission = any(term in question_norm for term in admission_terms)
        asks_admission = asks_admission or any(term in question_norm for term in [
            "diem chuan", "diem trung tuyen", "diem xet tuyen", "diem san",
            "hoc phi", "tuyen sinh", "xet tuyen", "nganh dao tao",
        ])
        if asks_admission and any(bad in domain for bad in hard_bad_domains):
            return f"bad_domain_for_admission:{domain}"
        if asks_admission and sum(1 for term in job_terms if term in merged) >= 2:
            return "job_content_for_admission_query"
        if asks_admission:
            level_reason = self._education_level_mismatch_reason(question, merged)
            if level_reason:
                return level_reason
        if len(self._tokenize(text)) < 20 and not source.get("files"):
            return "too_little_content"
        return None

    def _source_safety_filter(
        self,
        question: str,
        web_sources: list[WebSource],
        rag_sources: list[RagSource],
        params: GenerationParams,
    ) -> tuple[list[WebSource], list[RagSource]]:
        quality_log = params.get("quality_log", False)
        accepted_web: list[WebSource] = []
        accepted_urls: set[str] = set()
        for idx, source in enumerate(web_sources, 1):
            reason = self._source_safety_reason(question, source)
            if reason:
                if quality_log:
                    print(f"[SourceSafetyGate] [{idx:02d}] DROP | reason={reason} | title={source.get('title', '')[:80]} | url={source.get('url', '')}")
                continue
            accepted_web.append(source)
            canonical = self._canonical_url(source.get("url", ""))
            if canonical:
                accepted_urls.add(canonical)

        accepted_rag: list[RagSource] = []
        for source in rag_sources:
            canonical = self._canonical_url(source.get("url", ""))
            if not web_sources or (canonical and canonical in accepted_urls):
                accepted_rag.append(source)
        return accepted_web, accepted_rag

    def _chunk_gate(
        self,
        question: str,
        web_sources: list[WebSource],
        rag_sources: list[RagSource],
        params: GenerationParams
    ) -> tuple[list[WebSource], list[RagSource]]:
        quality_log = params.get("quality_log", True)
        gate_label = "ChunkGate"
        min_score = float(params.get("chunk_gate_min_score", params.get("quality_min_score", 0.58)))
        min_relevance = float(params.get("quality_min_relevance", 0.25))
        min_trust = float(params.get("quality_min_trust", 0.50))
        max_docs = int(params.get("quality_max_docs", params.get("k_pages", len(web_sources) or 1)))
        strict = bool(params.get("quality_strict_mode", True))

        if quality_log:
            print(f"\n[{gate_label}] STEP 3/3 - Validate crawled sources and selected chunks")
            print(
                f"[{gate_label}] Thresholds -> final_score>={min_score:.2f}, "
                f"relevance>={min_relevance:.2f}, trust>={min_trust:.2f}, "
                f"strict={strict}, max_docs={max_docs}"
            )
        log_step(
            "CHUNK_GATE",
            "INPUT",
            {
                "question": question,
                "web_source_count": len(web_sources),
                "rag_source_count": len(rag_sources),
                "thresholds": {
                    "min_score": min_score,
                    "min_relevance": min_relevance,
                    "min_trust": min_trust,
                    "strict": strict,
                    "max_docs": max_docs,
                },
                "web_sources": compact_sources(web_sources, "web", limit=10),
                "rag_sources": compact_sources(rag_sources, "rag", limit=10),
            },
            params,
        )

        accepted_candidates: list[tuple[WebSource, float, str]] = []
        accepted_urls: set[str] = set()
        seen_urls: set[str] = set()
        for idx, source in enumerate(web_sources, 1):
            title = source.get("title", "")
            url = source.get("url", "")
            description = source.get("description", source.get("text", ""))
            canonical_url = self._canonical_url(url)
            if canonical_url and canonical_url in seen_urls:
                if quality_log:
                    print(f"[{gate_label}] [{idx:02d}] DROP | reason=duplicate_url | url={url}")
                continue
            safety_reason = self._source_safety_reason(question, source)
            if safety_reason:
                if quality_log:
                    print(f"[{gate_label}] [{idx:02d}] DROP | reason={safety_reason} | title={title[:80]} | url={url}")
                continue
            text = source.get("text", "") or ""
            candidate_text = f"{title} {description} {url} {text[:3000]}"
            metric_reason = self._metric_mismatch_reason(question, candidate_text)
            if metric_reason:
                if quality_log:
                    print(f"[{gate_label}] [{idx:02d}] DROP | reason={metric_reason} | title={title[:80]} | url={url}")
                continue
            relevance = self._relevance_score_with_text(question, title, description, url, text)
            trust = self._domain_trust(url)
            richness = self._content_richness_score(text)
            final_score = 0.50 * relevance + 0.30 * trust + 0.20 * richness
            is_ok = (
                final_score >= min_score and relevance >= min_relevance and trust >= min_trust
                if strict else
                final_score >= min_score or (relevance >= min_relevance and trust >= min_trust)
            )

            if quality_log:
                status = "PASS" if is_ok else "DROP"
                print(
                    f"[{gate_label}] [{idx:02d}] {status} | final={final_score:.3f} "
                    f"rel={relevance:.3f} trust={trust:.3f} rich={richness:.3f} | "
                    f"title={title[:80]} | url={url}"
                )

            if is_ok:
                accepted_candidates.append((source, final_score, canonical_url))
                if canonical_url:
                    seen_urls.add(canonical_url)

        accepted_candidates.sort(key=lambda item: item[1], reverse=True)
        if max_docs > 0:
            accepted_candidates = accepted_candidates[:max_docs]
        accepted_web_sources = [source for source, _, _ in accepted_candidates]
        accepted_urls = {canonical for _, _, canonical in accepted_candidates if canonical}

        accepted_rag_sources: list[RagSource] = []
        for idx, source in enumerate(rag_sources, 1):
            source_url = self._canonical_url(source.get("url", ""))
            if web_sources and (not source_url or source_url not in accepted_urls):
                if quality_log:
                    print(f"[{gate_label}] chunk#{idx:02d} DROP | reason=source_url_not_accepted | url={source.get('url', '')}")
                continue
            chunk_text = source.get("text", "") or ""
            chunk_surface = f"{source.get('title', '')} {source.get('url', '')} {chunk_text}"
            chunk_reason = self._education_level_mismatch_reason(question, chunk_surface)
            if not chunk_reason:
                chunk_reason = self._metric_mismatch_reason(question, chunk_surface)
            if not chunk_reason and len(self._tokenize(chunk_text)) < 12:
                chunk_reason = "chunk_too_little_content"
            if chunk_reason:
                if quality_log:
                    print(
                        f"[{gate_label}] chunk#{idx:02d} DROP | reason={chunk_reason} | "
                        f"title={source.get('title', '')[:80]} | url={source.get('url', '')}"
                    )
                continue
            accepted_rag_sources.append(source)

        if quality_log:
            print(
                f"[{gate_label}] Result -> web_sources={len(accepted_web_sources)}/{len(web_sources)}, "
                f"rag_sources={len(accepted_rag_sources)}/{len(rag_sources)}"
            )
        _debug_source_list("CHUNKGATE_ACCEPTED_WEB_SOURCES", accepted_web_sources)
        _debug_source_list("CHUNKGATE_ACCEPTED_CHUNKS", accepted_rag_sources)
        log_step(
            "CHUNK_GATE",
            "OUTPUT",
            {
                "accepted_web_sources": len(accepted_web_sources),
                "accepted_rag_sources": len(accepted_rag_sources),
                "dropped_web_sources": len(web_sources) - len(accepted_web_sources),
                "dropped_rag_sources": len(rag_sources) - len(accepted_rag_sources),
                "web_sources": compact_sources(accepted_web_sources, "web", limit=10),
                "rag_sources": compact_sources(accepted_rag_sources, "rag", limit=10),
            },
            params,
        )
        return accepted_web_sources, accepted_rag_sources

    async def retrive(self, question: str, params: GenerationParams) -> tuple[list[WebSource], list[RagSource]]:
        enable_quality_gate = params.get("enable_quality_gate", False)
        enable_chunk_gate = params.get("enable_chunk_gate", enable_quality_gate)
        rich_media_enabled = bool(params.get("include_pdf") or params.get("include_image"))
        if rich_media_enabled and params.get("quality_force_for_rich_media", True):
            if not enable_quality_gate:
                print("[QualityGate] Forced ON because include_pdf/include_image is enabled")
            enable_quality_gate = True
            enable_chunk_gate = True
            params["enable_quality_gate"] = True
            params["enable_pre_crawl_quality_gate"] = True
            params["enable_chunk_gate"] = True
        quality_log = params.get("quality_log", enable_quality_gate)
        params["quality_log"] = quality_log

        if quality_log:
            print("\n[QualityGate] STEP 1/3 - Send query to keyword extractor")
            print(f"[QualityGate] Question: {question}")
        log_step(
            "QUERY",
            "INPUT",
            {
                "question": question,
                "max_query": params.get("max_query", 1),
                "school_domain": params.get("school_domain", False),
                "purpose": "keyword_extractor_for_web_search",
            },
            params,
        )

        data = await self.llm_keywords.keywords(question, params)
        max_query = params.get("max_query", 1)
        queries = []
        school_restrict = params.get("school_domain", False)
        for item in data:
            if not school_restrict:
                queries.append(item["query"])
            else:
                school = item["school"]
                if school.strip() != "":
                    school_domains = self.school_mapper.domains_from_auto(school, 5)[:10]
                    print(f"[DOMAINS]", school_domains)
                    if len(school_domains) > 0:
                        queries.append([item["query"], school_domains])
        queries = queries[:max_query]
        if quality_log:
            print(f"[QualityGate] Extracted queries ({len(queries)}): {queries}")
            print("[QualityGate] STEP 2/3 - Run web search + page rerank + pre-crawl gate")
        log_step(
            "QUERY",
            "OUTPUT",
            {
                "raw_keyword_objects": data,
                "queries": queries,
                "query_count": len(queries),
            },
            params,
        )

        web_sources, rag_sources = await self.pipeline.retrieve(params, queries)
        if quality_log:
            print(
                f"[ChunkGate] Raw retrieve -> web_sources={len(web_sources)}, "
                f"rag_sources={len(rag_sources)}"
            )
        _debug_source_list("WEB_RAW_SOURCES", web_sources)
        _debug_source_list("RAG_RAW_CHUNKS", rag_sources)

        if not enable_chunk_gate:
            if quality_log:
                print("[ChunkGate] Disabled -> run SourceSafetyGate only")
            log_step(
                "CHUNK_GATE",
                "INPUT",
                {
                    "enabled": False,
                    "question": question,
                    "web_source_count": len(web_sources),
                    "rag_source_count": len(rag_sources),
                    "web_sources": compact_sources(web_sources, "web", limit=10),
                    "rag_sources": compact_sources(rag_sources, "rag", limit=10),
                },
                params,
            )
            log_step(
                "CHUNK_GATE",
                "OUTPUT",
                {
                    "enabled": False,
                    "reason": "chunk_gate_disabled",
                    "source_safety_filter": params.get("source_safety_filter", True),
                    "web_sources": len(web_sources),
                    "rag_sources": len(rag_sources),
                },
                params,
            )
            if params.get("source_safety_filter", True):
                return self._source_safety_filter(question, web_sources, rag_sources, params)
            return web_sources, rag_sources

        return self._chunk_gate(question, web_sources, rag_sources, params)
    
class RouterRetriever:
    def __init__(self, llm_router: RouterModelProtocol, web_retriever: WebRetriever, local_retriever: LocalRetriever) -> None:
        self.web_retriever = web_retriever
        self.local_retriever = local_retriever
        self.router = llm_router
    def _auto_apply_time_filter(self, question: str, params: GenerationParams) -> None:
        """Infer time filter (year) from question when user chưa set thủ công."""
        if params.get("time_metric") or params.get("time_range"):
            return
        # Ưu tiên năm gần nhất được nhắc tới
        year_candidates: list[int] = []
        # Range dạng 2022-2023
        for start, end in re.findall(r"(19\d{2}|20\d{2})\s*[-–]\s*(19\d{2}|20\d{2})", question):
            year_candidates.extend([int(start), int(end)])
        # Các năm đơn lẻ
        for year_str in re.findall(r"\b(19\d{2}|20\d{2})\b", question):
            year_candidates.append(int(year_str))
        if not year_candidates:
            return
        current_year = datetime.now().year
        past_years = [y for y in year_candidates if y <= current_year]
        if not past_years:
            return
        target_year = max(past_years)
        time_range = max(1, current_year - target_year + 1)
        time_range = min(time_range, 10)  # tránh range quá rộng
        params["time_metric"] = "y"
        params["time_range"] = time_range
        print(f"[AUTO TIME] Detected year {target_year} -> range={time_range}y")
    async def retrieve(self, question: str, params: GenerationParams) -> tuple[list[WebSource], list[RagSource]]:
        use_websearch = params.get("use_websearch", False) and params.get("max_query", 0) > 0 and params.get("k_docs", 0) > 0 and params.get("k_pages", 0) > 0
        use_localdb = params.get("use_localdb", False)
        _debug_block(
            "ROUTER_SOURCE_FLAGS",
            _debug_json({
                "question": question,
                "use_localdb": use_localdb,
                "use_websearch": use_websearch,
                "auto_source": params.get("auto_source"),
                "max_query": params.get("max_query"),
                "k_docs": params.get("k_docs"),
                "k_pages": params.get("k_pages"),
            }),
            limit=0,
        )
        log_step(
            "SOURCE_ROUTER",
            "INPUT",
            {
                "question": question,
                "use_localdb": use_localdb,
                "use_websearch": use_websearch,
                "auto_source": params.get("auto_source"),
                "source_mode": params.get("source_mode"),
                "max_query": params.get("max_query"),
                "k_docs": params.get("k_docs"),
                "k_pages": params.get("k_pages"),
            },
            params,
        )
        if use_websearch:
            self._auto_apply_time_filter(question, params)
        if use_websearch and use_localdb:
            local_queries = await self.router.route(question, params)
            hybrid_retrieval = bool(params.get("hybrid_retrieval", False))
            if len(local_queries) > 0:
                _debug_block(
                    "ROUTER_SOURCE_DECISION",
                    _debug_json({"mode": "hybrid" if hybrid_retrieval else "local_db", "reason": "router returned local queries", "local_queries": local_queries}),
                    limit=0,
                )
                local_web, local_rag = self.local_retriever.retrieve(local_queries, params)
                if hybrid_retrieval:
                    web_web, web_rag = await self.web_retriever.retrive(question, params)
                    log_step(
                        "SOURCE_ROUTER",
                        "OUTPUT",
                        {
                            "mode": "hybrid",
                            "reason": "router returned local queries and hybrid_retrieval=True",
                            "local_queries": local_queries,
                            "web_sources": len(local_web) + len(web_web),
                            "rag_sources": len(local_rag) + len(web_rag),
                        },
                        params,
                    )
                    return local_web + web_web, local_rag + web_rag
                log_step(
                    "SOURCE_ROUTER",
                    "OUTPUT",
                    {
                        "mode": "local_db",
                        "reason": "router returned local queries",
                        "local_queries": local_queries,
                        "web_sources": len(local_web),
                        "rag_sources": len(local_rag),
                    },
                    params,
                )
                return local_web, local_rag
            else:
                _debug_block(
                    "ROUTER_SOURCE_DECISION",
                    _debug_json({"mode": "web", "reason": "router returned no local queries"}),
                    limit=0,
                )
                web, rag = await self.web_retriever.retrive(question, params)
                log_step(
                    "SOURCE_ROUTER",
                    "OUTPUT",
                    {
                        "mode": "web",
                        "reason": "router returned no local queries",
                        "web_sources": len(web),
                        "rag_sources": len(rag),
                    },
                    params,
                )
                return web, rag
        elif use_localdb:
            local_queries = await self.router.route(question, params)
            if len(local_queries) > 0:
                _debug_block(
                    "ROUTER_SOURCE_DECISION",
                    _debug_json({"mode": "local_db", "reason": "local only, router returned queries", "local_queries": local_queries}),
                    limit=0,
                )
                web, rag = self.local_retriever.retrieve(local_queries, params)
                log_step(
                    "SOURCE_ROUTER",
                    "OUTPUT",
                    {
                        "mode": "local_db",
                        "reason": "local only, router returned queries",
                        "local_queries": local_queries,
                        "web_sources": len(web),
                        "rag_sources": len(rag),
                    },
                    params,
                )
                return web, rag
            else:
                _debug_block(
                    "ROUTER_SOURCE_DECISION",
                    _debug_json({"mode": "none", "reason": "local only, router returned no local queries"}),
                    limit=0,
                )
                log_step(
                    "SOURCE_ROUTER",
                    "OUTPUT",
                    {
                        "mode": "none",
                        "reason": "local only, router returned no local queries",
                        "web_sources": 0,
                        "rag_sources": 0,
                    },
                    params,
                )
                return [], []
        elif use_websearch:
            _debug_block(
                "ROUTER_SOURCE_DECISION",
                _debug_json({"mode": "web", "reason": "web only"}),
                limit=0,
            )
            web, rag = await self.web_retriever.retrive(question, params)
            log_step(
                "SOURCE_ROUTER",
                "OUTPUT",
                {
                    "mode": "web",
                    "reason": "web only",
                    "web_sources": len(web),
                    "rag_sources": len(rag),
                },
                params,
            )
            return web, rag
        else:
            _debug_block(
                "ROUTER_SOURCE_DECISION",
                _debug_json({"mode": "none", "reason": "both local and web disabled"}),
                limit=0,
            )
            log_step(
                "SOURCE_ROUTER",
                "OUTPUT",
                {
                    "mode": "none",
                    "reason": "both local and web disabled",
                    "web_sources": 0,
                    "rag_sources": 0,
                },
                params,
            )
            return [], []
        
import openai
class APIModelCore:
    def __init__(self) -> None:
        self.gpt_client = AsyncOpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))
        self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.logger = CmdLogger("Model")
    async def call(self, call_type: CallType, instruction: str, prompt: str, params: GenerationParams) -> AsyncGenerator[str, None]:
        print(f"[API] {call_type} | Instruction length: {len(instruction)} | Prompt length: {len(prompt)} | kwargs: {params.get('kwargs')}")
        if _debug_trace_enabled():
            prompt_limit = _debug_int_env("BOT_DEBUG_PROMPT_CHARS", 20000)
            _debug_block(
                f"LLM_CALL {call_type}",
                _debug_json({
                    "call_type": str(call_type),
                    "model_id": params.get("model_id"),
                    "max_tokens": params.get("max_tokens"),
                    "temperature": params.get("temperature"),
                    "top_p": params.get("top_p"),
                    "top_k": params.get("top_k"),
                    "params": dict(params),
                    "instruction_chars": len(instruction or ""),
                    "prompt_chars": len(prompt or ""),
                }),
                limit=0,
            )
            if _debug_full_prompt_enabled():
                _debug_block(f"LLM_SYSTEM_PROMPT {call_type}", instruction, limit=prompt_limit)
                _debug_block(f"LLM_USER_PROMPT {call_type}", prompt, limit=prompt_limit)
        model_id = params["model_id"]
        if "gpt" in model_id:
            while True:
                try:
                    stream = await self.gpt_client.chat.completions.create(
                        model=model_id, 
                        messages=[
                            {"role": "system", "content": instruction},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=params.get("max_tokens", 4096),
                        temperature=params.get("temperature", 0.7),
                        top_p=params.get("top_p", 0.9),
                        presence_penalty=params.get("presence_penalty", 0.1),
                        frequency_penalty=params.get("frequency_penalty", 0.0),
                        stream=True
                    )
                    total_text = ""
                    async for event in stream:
                        chunk = event.choices[0].delta.content
                        if chunk is not None:
                            total_text += chunk
                            yield chunk
                    return
                except openai.APIError as e:
                    print(e)
        else:
            gemini_config = types.GenerateContentConfig(
                temperature=params.get("temperature", 0.8),
                top_p=params.get("top_p", 0.9),
                top_k=params.get("top_k", 16),
                max_output_tokens=params.get("max_tokens", 4048)
            )
            stream = await self.gemini_client.aio.models.generate_content_stream(
                model=model_id,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=instruction)]
                    ),
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)]
                    )
                ],
                config=gemini_config
            )
            async for chunk in stream:
                if chunk.candidates:
                    if chunk.candidates[0].content:
                        if chunk.candidates[0].content.parts:
                            for part in chunk.candidates[0].content.parts:
                                if part.text:
                                    yield part.text
    async def __call__(self, call_type: CallType, instruction: str, prompt: str, params: GenerationParams) -> AsyncGenerator[str, None]:
        return self.call(call_type, instruction, prompt, params)
    
class APIModel(APIModelCore):
    async def decompose(self, question: str, params: GenerationParams):
        """Phân rã câu hỏi thành DecomposerPlan (multi-hop bước đầu).

        Trả về `DecomposerPlan | None`. None khi parse fail → caller sẽ fallback single-hop.
        """
        from data_retriever import parse_plan_json
        copy_params = copy.deepcopy(params)
        copy_params.update(DECOMPOSER_PARAMS)  # type: ignore
        prompt = DECOMPOSER_TEMPLATE.format(question=question)
        text = ""
        try:
            async for chunk in await self(
                call_type="decomposer",  # CallType enum chỉ có 4 giá trị sẵn; dùng string cho feature mới
                instruction=DECOMPOSER_INSTRUCTION + "\n\n" + DECOMPOSER_PREFIX,
                prompt=prompt,
                params=copy_params,
            ):
                text += chunk
        except Exception:
            traceback.print_exc()
            return None
        self.logger.log(f"[Decomposer raw] {text[:400]}...")
        _debug_block("DECOMPOSER_RAW_OUTPUT", text, limit=_debug_int_env("BOT_DEBUG_PROMPT_CHARS", 20000))
        plan = parse_plan_json(text, max_sub=5)
        if plan is None:
            self.logger.log("[Decomposer] parse failed → single-hop")
        return plan

    async def extract_fact(
        self,
        sub_q_text: str,
        rag_sources: list,
        params: GenerationParams,
        evidence_type: str = "factual",
    ) -> tuple[str, float]:
        """Trích fact từ top chunks cho bridge-entity hop tiếp theo.

        Chế độ chọn theo `evidence_type`:
        - factual            → 1 câu ≤25 từ (FACT_EXTRACTOR_*)
        - numeric/list/comparison/computation → liệt kê entries (LIST_FACT_EXTRACTOR_*)
        """
        if not rag_sources:
            return "", 0.0
        # Chọn prompt + params theo evidence_type
        try:
            instr, template = select_fact_extractor_prompt(evidence_type)
        except NameError:
            # Fallback nếu instruction module chưa có helper.
            instr, template = FACT_EXTRACTOR_INSTRUCTION, FACT_EXTRACTOR_TEMPLATE
        is_list_mode = (evidence_type or "factual").lower() in {
            "list", "numeric", "comparison", "computation"
        }
        # List mode: lấy nhiều chunks hơn + cắt dài hơn để model thấy đủ data.
        if is_list_mode:
            def _fact_chunk_key(c: dict) -> tuple[int, int, int]:
                text = c.get("text", "") or ""
                table_like = 1 if _is_table_like_text(text) else 0
                has_number = 1 if re.search(r"\d", text) else 0
                title_only = 1 if len(text.strip()) < 160 else 0
                return (table_like, has_number, -title_only)

            top = sorted(rag_sources[:12], key=_fact_chunk_key, reverse=True)[:8]
            chunk_max_chars = 4000
            run_params = LIST_FACT_EXTRACTOR_PARAMS
        else:
            top = rag_sources[:5]
            chunk_max_chars = 500
            run_params = FACT_EXTRACTOR_PARAMS
        context = "\n\n".join(
            (
                f"### Source {idx}: {c.get('title', '')} | {c.get('url', '')}\n"
                f"{_select_relevant_evidence_text(sub_q_text, c.get('text', '') or '', chunk_max_chars)}"
            )
            for idx, c in enumerate(top, 1)
        )
        prompt = template.format(question=sub_q_text, context=context)
        _debug_block(
            "FACT_EXTRACTOR_INPUT",
            _debug_json({
                "question": sub_q_text,
                "evidence_type": evidence_type,
                "source_count": len(rag_sources),
                "selected_count": len(top),
                "selected_sources": [
                    {
                        "title": c.get("title", ""),
                        "url": c.get("url", ""),
                        "chunk_index": c.get("chunk_index"),
                        "text_chars": len(c.get("text", "") or ""),
                    }
                    for c in top
                ],
            }),
            limit=0,
        )
        copy_params = copy.deepcopy(params)
        copy_params.update(run_params)  # type: ignore
        text = ""
        try:
            async for chunk in await self(
                call_type="fact_extractor",
                instruction=instr,
                prompt=prompt,
                params=copy_params,
            ):
                text += chunk
        except Exception:
            traceback.print_exc()
            return "", 0.0
        _debug_block("FACT_EXTRACTOR_RAW_OUTPUT", text, limit=_debug_int_env("BOT_DEBUG_PROMPT_CHARS", 20000))
        try:
            result = json.loads(extract_json(text))
            answer = str(result.get("answer", "")).strip()
            if isinstance(result.get("items"), list):
                items_answer = " ; ".join(
                    f"{item.get('name', '')}: {item.get('value', '')}".strip(": ")
                    for item in result.get("items", [])
                    if isinstance(item, dict) and (item.get("name") or item.get("value"))
                ).strip()
                if items_answer and (is_list_mode or not answer):
                    answer = items_answer
            conf = float(result.get("confidence", 0.0))
            conf = _calibrate_fact_confidence(answer, conf, evidence_type, sub_q_text, context)
            _debug_block("FACT_EXTRACTOR_PARSED", _debug_json({"answer": answer, "confidence": conf}), limit=0)
            return answer, conf
        except Exception:
            _debug_block("FACT_EXTRACTOR_PARSE_FAILED", text, limit=_debug_int_env("BOT_DEBUG_PROMPT_CHARS", 20000))
            return "", 0.0

    async def reason(
        self,
        original_question: str,
        sub_q_text: str,
        evidence_blocks: list,
        params: GenerationParams,
    ) -> tuple[str, float]:
        """Reasoning step: LLM tính toán/lọc/so sánh trên evidence từ deps.

        Trả `(answer_text, confidence)`. answer_text có thể là chuỗi nhiều dòng
        (mỗi dòng 1 entry) — Reader/hop sau sẽ hiểu được.
        """
        if not evidence_blocks:
            return "", 0.0
        try:
            instr = REASONER_INSTRUCTION
            template = REASONER_TEMPLATE
        except NameError:
            return "", 0.0
        # Concat tất cả evidence blocks; mỗi block đã được multi_hop cắt độ dài.
        joined = "\n\n".join(evidence_blocks)
        prompt = template.format(
            original_question=original_question,
            sub_question=sub_q_text,
            evidence=joined,
        )
        _debug_block(
            "REASONER_INPUT",
            _debug_json({
                "original_question": original_question,
                "sub_question": sub_q_text,
                "evidence_blocks": len(evidence_blocks),
                "evidence_chars": len(joined),
            }),
            limit=0,
        )
        _debug_block("REASONER_EVIDENCE", joined, limit=_debug_int_env("BOT_DEBUG_PROMPT_CHARS", 20000))
        copy_params = copy.deepcopy(params)
        copy_params.update(REASONER_PARAMS)  # type: ignore
        text = ""
        try:
            async for chunk in await self(
                call_type="reasoner",
                instruction=instr,
                prompt=prompt,
                params=copy_params,
            ):
                text += chunk
        except Exception:
            traceback.print_exc()
            return "", 0.0
        _debug_block("REASONER_RAW_OUTPUT", text, limit=_debug_int_env("BOT_DEBUG_PROMPT_CHARS", 20000))
        try:
            result = json.loads(extract_json(text))
            answer = str(result.get("answer", "")).strip()
            if isinstance(result.get("items"), list):
                items_answer = " ; ".join(
                    f"{item.get('name', '')}: {item.get('value', '')}".strip(": ")
                    for item in result.get("items", [])
                    if isinstance(item, dict) and (item.get("name") or item.get("value"))
                ).strip()
                if items_answer:
                    answer = items_answer
            conf = float(result.get("confidence", 0.0))
            conf = _calibrate_fact_confidence(answer, conf, "computation", f"{original_question} {sub_q_text}")
            grounded, grounding_reason = _reasoner_answer_grounded(
                answer, joined, f"{original_question} {sub_q_text}"
            )
            if not grounded:
                print(f"[ReasonerGuard] discard ungrounded answer: {grounding_reason}")
                answer = ""
                conf = 0.0
            _debug_block(
                "REASONER_PARSED",
                _debug_json({
                    "answer": answer,
                    "confidence": conf,
                    "grounded": grounded,
                    "grounding_reason": grounding_reason,
                }),
                limit=0,
            )
            return answer, conf
        except Exception:
            # Một số trường hợp LLM không trả JSON: dùng raw text như answer.
            _debug_block("REASONER_PARSE_FAILED", text, limit=_debug_int_env("BOT_DEBUG_PROMPT_CHARS", 20000))
            return "", 0.0

    async def route(self, question: str, params: GenerationParams) -> list[dict]:
        text = ""
        prompt = ROUTER_TEMPLATE.format(question=question)
        copy_params = copy.deepcopy(params)
        copy_params.update(ROUTER_PARAMS) #type:ignore 
        async for chunk in await self(
            call_type=CallType.ROUTER, 
            instruction=ROUTER_INSTRUCTION+ROUTER_PREFIX, 
            prompt=prompt, 
            params=copy_params
        ):
            text += chunk
        try:
            self.logger.log(text)
            _debug_block("ROUTER_RAW_OUTPUT", text, limit=_debug_int_env("BOT_DEBUG_PROMPT_CHARS", 20000))
            result = json.loads(extract_json(text))
            _debug_block("ROUTER_PARSED", _debug_json(result), limit=0)
            return result
        except:
            traceback.print_exc()
            return []
    async def _llm_rerank_page(self, pages: list[SearchResult], query: str, relative_threshold: float, params: GenerationParams) -> list[SearchResult]:
        if len(pages) == 0: return []
        text = ""    
        scores = [0.0 for _ in pages]
        prompt = self._construct_reranker_prompt(query, pages)
        copy_params = copy.deepcopy(params)
        copy_params.update(PAGE_RERANKER_PARAMS) #type:ignore
        async for chunk in await self(
            call_type=CallType.RANKER, 
            instruction=PAGE_RERANKER_INSTRUCTION+PAGE_RERANKER_PREFIX, 
            prompt=prompt, 
            params=copy_params
        ):
            text += chunk
        try:
            self.logger.log(text)
            result = json.loads(extract_json(text))
            if "output" in result:
                for item in result["output"]:
                    index = int(item["index"]) - 1
                    scores[index] = float(item["score"])
            else:
                # Fallback if model not provide intermediate step
                for item in result:
                    index = int(item["index"]) - 1
                    scores[index] = float(item["score"])
        except:
            traceback.print_exc()
        self.logger.log("-----Original-----")
        if self.logger._enable:
            for page in pages:
                self.logger.log(f'{page["score"]:.3f} + {page["title"]}')
        max_score = 0
        for score, search_result in zip(scores, pages):
            max_score = max(max_score, score)
            search_result["score"] = score
        if max_score == 0: return []
        
        results: list[SearchResult] = []
        for search_result in pages:
            if search_result["score"] >= max_score * relative_threshold:
                results.append(search_result)
        results = sorted(results, key=lambda r:r["score"], reverse=True)
        self.logger.log("-----Reorder-----")
        if self.logger._enable:
            for page in results:
                self.logger.log(f'{page["score"]:.3f} + {page["title"]}')
        return results
    def _construct_reranker_prompt(self, query: str, data: list[SearchResult]) -> str:
        candidates = [{
            "index": index+1, 
            "title": item["title"],
            "description": item["description"],
        } for index, item in enumerate(data)]
        candidates = "[" + ",\n".join([json.dumps(item, ensure_ascii=False) for item in candidates]) + "]"
        return PAGE_RERANKER_TEMPLATE.format(query=query, pages=candidates)
    async def keywords(self, question: str, params: GenerationParams, threshold: float = 0.5) -> list[KeywordInfo]:
        num_queries = params.get("max_query", 1)
        copy_params = copy.deepcopy(params)
        copy_params.update(KEYWORDS_PARAMS) #type:ignore
        prompt = KEYWORD_TEMPLATE.format(question=question)
        text = ""
        async for chunk in await self(
            call_type=CallType.KEYWORDS, 
            instruction=KEYWORDS_INTRUCTION, 
            prompt=KEYWORDS_PREFIX.replace("{num}", str(num_queries))+prompt, 
            params=copy_params
        ):
            text += chunk
        try:
            self.logger.log(text)
            _debug_block("KEYWORDS_RAW_OUTPUT", text, limit=_debug_int_env("BOT_DEBUG_PROMPT_CHARS", 20000))
            result: list[KeywordInfo] = json.loads(extract_json(text))
            _debug_block("KEYWORDS_PARSED", _debug_json(result), limit=0)
            for item in result:
                self.logger.log(item)
            return result
        except:
            print(text)
            traceback.print_exc()
            return []
    async def _heristic_rerank_page(self, pages: list[SearchResult], query: str, relative_threshold: float, params: GenerationParams) -> list[SearchResult]:
        """Rerank search results using embedding similarity"""
        self.logger.log("-----Original-----")
        if self.logger._enable:
            for page in pages:
                self.logger.log(f'{page["score"]:.3f} + {page["title"]}')
        import numpy as np
        def normalize_text(text: str) -> str:
            text = text.lower().strip()
            text = re.sub(r"[^a-zA-Z0-9\u00C0-\u1EF9\s\.,;]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        def detect_school(query: str, schools: dict) -> str | None:
            """Detect school from query using predefined keywords"""
            for school, aliases in schools.items():
                if any(alias in query for alias in aliases):
                    return school
            return None
        schools = {
            school: [normalize_text(alias) for alias in aliases]
            for school, aliases in json.load(open(f"{BASE_PATH}files/school_alias.json", "r", encoding="utf-8")).items()
        }
        embedding = ws_pipeline.retriever.web_retriever.pipeline._rag.embedding
        
        # If no embedding model available, return original results
        if not embedding:
            return pages
        
        query_norm = normalize_text(query)
        detected_school = detect_school(query_norm, schools)
        
        try:
            query_emb = embedding.embed_query(query_norm)
            max_score = 0
            for page in pages:
                title = page.get("title", "") or ""
                desc = page.get("description", "") or ""
                url = page.get("url", "") or ""

                # Chuẩn hóa
                title_norm = normalize_text(title)
                desc_norm = normalize_text(desc)
                url_norm = normalize_text(url)

                # Semantic embedding
                title_emb = embedding.embed_query(title_norm) if title_norm else None
                desc_emb = embedding.embed_query(desc_norm) if desc_norm else None
                url_emb = embedding.embed_query(url_norm) if url_norm else None

                # Cosine similarity
                def cos_sim(a, b):
                    norm_a = np.linalg.norm(a)
                    norm_b = np.linalg.norm(b)
                    if norm_a == 0 or norm_b == 0:
                        return 0.0
                    return float(np.dot(a, b) / (norm_a * norm_b))

                score = 0.0
                weights = {"title": 0.5, "desc": 0.3, "url": 0.2}
                if title_emb is not None:
                    score += cos_sim(query_emb, title_emb) * weights["title"]
                if desc_emb is not None:
                    score += cos_sim(query_emb, desc_emb) * weights["desc"]
                if url_emb is not None:
                    score += cos_sim(query_emb, url_emb) * weights["url"]

                # Heuristic ưu tiên trường trong query
                if detected_school:
                    aliases = [normalize_text(a) for a in schools.get(detected_school, [])]
                    if any(a in text for a in aliases for text in [url_norm, title_norm, desc_norm]):
                        score += 0.5
                    else:
                        for school, other_aliases in schools.items():
                            if school != detected_school:
                                other_aliases_norm = [normalize_text(a) for a in other_aliases]
                                if any(a in text for a in other_aliases_norm for text in [url_norm, title_norm, desc_norm]):
                                    score -= 0.5

                # Heuristic boost
                if any(kw in query_norm for kw in ["tuyển sinh", "ngành đào tạo"]):
                    if "tuyensinh247" in url_norm:
                        score += 0.1
                    if url_norm.endswith(".edu") or ".edu.vn" in url_norm:
                        score += 0.2
                page["score"] = score
                max_score = max(score, max_score)
            threshold_score = max_score * relative_threshold
            # Sort theo score giảm dần
            results = []
            for page in pages:
                if page["score"] >= threshold_score:
                    results.append(page)
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            self.logger.log("-----Reorder-----")
            if self.logger._enable:
                for page in results:
                    self.logger.log(f'{page["score"]:.3f} + {page["title"]}')
            return results
            
        except Exception:
            # If any error occurs, return original results
            return pages
    async def rerank_page(self, pages: list[SearchResult], query: str, relative_threshold: float, params: GenerationParams) -> list[SearchResult]:
        use_llm_rerank = params.get("llm_rerank", False)
        if use_llm_rerank:
            return await self._llm_rerank_page(pages, query, relative_threshold, params)
        else:
            return await self._heristic_rerank_page(pages, query, relative_threshold, params)
        
class CombinedProtocol(ModelProtocol, KeywordModelProtocol, PageRerankModelProtocol, RouterModelProtocol):
    pass


class _DecomposerAdapter:
    """Adapter để MultiHopOrchestrator gọi được `APIModel.decompose` theo protocol."""
    def __init__(self, api_model):
        self._m = api_model

    async def decompose(self, question: str, params):
        from data_retriever import DecomposerPlan
        plan = await self._m.decompose(question, params)
        return plan  # DecomposerPlan | None — Orchestrator tự xử lý None


class _FactExtractorAdapter:
    def __init__(self, api_model):
        self._m = api_model

    async def extract(self, sub_q_text, rag_sources, params, evidence_type: str = "factual"):
        return await self._m.extract_fact(
            sub_q_text, rag_sources, params, evidence_type=evidence_type
        )


class _ReasonerAdapter:
    """Adapter cho ReasonerProtocol — gọi APIModel.reason()."""
    def __init__(self, api_model):
        self._m = api_model

    async def reason(self, original_question, sub_q_text, evidence_blocks, params):
        return await self._m.reason(
            original_question, sub_q_text, evidence_blocks, params
        )


class CustomQA:
    def __init__(self, model_protocol: CombinedProtocol) -> None:
        self.logger = CmdLogger("QA")
        web_retriever = WebRetriever(model_protocol, model_protocol)
        local_retriever = LocalRetriever()
        self.retriever = RouterRetriever(
            model_protocol,
            web_retriever,
            local_retriever
        )
        self.llm_call = model_protocol
        # Sufficiency Gate: reuse cross-encoder đã load trong web pipeline để tiết kiệm VRAM
        self.sufficiency = SufficiencyGate(
            chunk_ranker=web_retriever.pipeline.chunk_ranker,
            config=SufficiencyConfig(),
        )
        # Multi-Hop Orchestrator (bước đầu xử lý câu hỏi phức tạp).
        # Opt-in: mặc định disabled; bật qua params["use_multi_hop"]=True.
        self.multi_hop = MultiHopOrchestrator(
            base_retriever=self.retriever,
            decomposer=_DecomposerAdapter(model_protocol),
            fact_extractor=_FactExtractorAdapter(model_protocol),
            reasoner=_ReasonerAdapter(model_protocol),
            config=MultiHopConfig(enabled=True, max_hops=3, max_sub_questions=5),
        )
    async def start(self):
        await self.retriever.web_retriever.start()
    async def inference(self, prompt: str, request: WorkerChatRequest) -> AsyncGenerator[str, None]:
        text = ""
        log_step(
            "FINAL_READER",
            "INPUT",
            {
                "prompt_chars": len(prompt or ""),
                "prompt_preview": trace_preview(prompt, 1200),
                "model_id": request["params"].get("model_id"),
                "temperature": request["params"].get("temperature"),
            },
            request["params"],
        )
        hard_answer = request["params"].get("_hard_no_answer_text")
        if hard_answer:
            text = str(hard_answer)
            yield text
            log_step(
                "FINAL_READER",
                "OUTPUT",
                {
                    "answer_chars": len(text),
                    "answer_preview": trace_preview(text, 1600),
                    "hard_no_answer": True,
                },
                request["params"],
            )
            return
        async for chunk in await self.llm_call(
            call_type=CallType.READER, 
            instruction=READER_UNTRAINED_INSTRUCTION+READER_UNTRAINED_PREFIX, 
            prompt=prompt, 
            params=request["params"]
        ):
            text += chunk
            yield chunk
        log_step(
            "FINAL_READER",
            "OUTPUT",
            {
                "answer_chars": len(text),
                "answer_preview": trace_preview(text, 1600),
            },
            request["params"],
        )
    async def pre_inference(
        self,
        question: str,
        stream_id: str,
        params: GenerationParams
    ) -> tuple[str, ModelPreOutput]:
        reset_trace()
        log_step(
            "REQUEST",
            "INPUT",
            {
                "stream_id": stream_id,
                "question": question,
                "params": dict(params),
            },
            params,
        )
        _debug_block(
            "REQUEST_START",
            _debug_json({
                "stream_id": stream_id,
                "question": question,
                "params": dict(params),
            }),
            limit=0,
        )
        # Multi-hop (opt-in). Khi tắt hoặc decomposer fail → Orchestrator tự fallback single-hop.
        web_sources, rag_sources, multi_hop_trace = await self.multi_hop.retrieve(
            question, params
        )
        if multi_hop_trace is not None:
            print(
                f"[MultiHop] hops={multi_hop_trace.hop_count} "
                f"sub_q={len(multi_hop_trace.plan.sub_questions)}"
            )
            for sq in multi_hop_trace.plan.sub_questions:
                r = multi_hop_trace.per_sub_q.get(sq.id)
                fact_preview = (r.fact[:80] + "...") if (r and r.fact) else "(no fact)"
                print(f"  SQ#{sq.id} [{sq.resolver}] → {fact_preview}")
        if multi_hop_trace is not None:
            _debug_block(
                "MULTI_HOP_TRACE",
                _debug_json({
                    "hop_count": multi_hop_trace.hop_count,
                    "sub_questions": [
                        {
                            "id": sq.id,
                            "text": sq.text,
                            "depends_on": sq.depends_on,
                            "resolver": sq.resolver,
                            "evidence_type": sq.evidence_type,
                            "rewritten_text": (multi_hop_trace.per_sub_q.get(sq.id).rewritten_text if multi_hop_trace.per_sub_q.get(sq.id) else ""),
                            "fact": (multi_hop_trace.per_sub_q.get(sq.id).fact if multi_hop_trace.per_sub_q.get(sq.id) else ""),
                            "confidence": (multi_hop_trace.per_sub_q.get(sq.id).confidence if multi_hop_trace.per_sub_q.get(sq.id) else 0.0),
                            "web_count": (len(multi_hop_trace.per_sub_q.get(sq.id).web_sources) if multi_hop_trace.per_sub_q.get(sq.id) else 0),
                            "rag_count": (len(multi_hop_trace.per_sub_q.get(sq.id).rag_sources) if multi_hop_trace.per_sub_q.get(sq.id) else 0),
                        }
                        for sq in multi_hop_trace.plan.sub_questions
                    ],
                }),
                limit=0,
            )
        else:
            _debug_block("MULTI_HOP_TRACE", "trace=None (single-hop fallback or disabled)")
        if multi_hop_trace is not None:
            q_norm = _normalize_for_match(question)
            strict_multi_hop = any(_term_in_text(t, q_norm) for t in ["so sanh", "danh sach", "xep hang", "top", "loc"])
            needs_metric = any(_term_in_text(t, q_norm) for t in ["diem chuan", "diem trung tuyen", "hoc phi"])
            missing_reasoning: list[int] = []
            missing_metric_subq: list[int] = []
            for sq in multi_hop_trace.plan.sub_questions:
                res = multi_hop_trace.per_sub_q.get(sq.id)
                fact = (getattr(res, "fact", "") or "").strip() if res else ""
                if _negative_or_empty_answer(fact):
                    fact = ""
                sq_norm = _normalize_for_match(sq.text)
                if sq.resolver == "reasoning" and not fact:
                    missing_reasoning.append(sq.id)
                elif any(_term_in_text(t, sq_norm) for t in ["diem chuan", "diem trung tuyen", "hoc phi"]) and not fact:
                    missing_metric_subq.append(sq.id)
            if strict_multi_hop and needs_metric and (missing_reasoning or missing_metric_subq):
                params["_hard_no_answer_text"] = (
                    "Tôi chưa tìm thấy đủ thông tin đáng tin cậy để trả lời đầy đủ câu hỏi này."
                )
                print(
                    f"[MultiHopHardStop] missing_reasoning={missing_reasoning} "
                    f"missing_metric_subq={missing_metric_subq}"
                )
                log_step(
                    "MULTIHOP_HARD_STOP",
                    "OUTPUT",
                    {
                        "missing_reasoning": missing_reasoning,
                        "missing_metric_subq": missing_metric_subq,
                        "answer": params["_hard_no_answer_text"],
                    },
                    params,
                )
        log_step(
            "FINAL_AGGREGATE",
            "INPUT",
            {
                "question": question,
                "web_sources_before_table_rescue": len(web_sources),
                "rag_sources_before_table_rescue": len(rag_sources),
                "table_rescue_max_sources": int(params.get("reader_table_rescue_max_sources", 3)),
                "table_rescue_max_chars": int(params.get("reader_table_rescue_max_chars", 12000)),
            },
            params,
        )
        rag_sources = _augment_rag_with_web_table_evidence(
            question,
            web_sources,
            rag_sources,
            max_sources=int(params.get("reader_table_rescue_max_sources", 3)),
            max_chars=int(params.get("reader_table_rescue_max_chars", 12000)),
        )
        checked_chunks, slot_report = _build_multihop_checked_context(question, multi_hop_trace)
        if checked_chunks:
            rag_sources = checked_chunks + rag_sources
            params["_multi_hop_slot_gate"] = slot_report
            if slot_report.get("partial"):
                params["_allow_partial_answer"] = True
            if slot_report.get("hard_no_answer"):
                params["_hard_no_answer_text"] = (
                    "Tôi chưa tìm thấy đủ thông tin điểm chuẩn và học phí đáng tin cậy "
                    "để lọc danh sách theo tất cả điều kiện."
                )
            log_step(
                "MULTIHOP_SLOT_GATE",
                "OUTPUT",
                {
                    "report": slot_report,
                    "checked_chunks": compact_sources(checked_chunks, "rag", limit=3),
                },
                params,
            )
            _debug_block("MULTIHOP_SLOT_GATE", _debug_json(slot_report), limit=0)
        log_step(
            "FINAL_AGGREGATE",
            "OUTPUT",
            {
                "question": question,
                "web_sources": len(web_sources),
                "rag_sources": len(rag_sources),
                "web_preview": compact_sources(web_sources, "web", limit=10),
                "rag_preview": compact_sources(rag_sources, "rag", limit=10),
            },
            params,
        )
        _debug_source_list("FINAL_WEB_SOURCES", web_sources)
        _debug_source_list("FINAL_RAG_CHUNKS", rag_sources)
        print("\n" + "=" * 80)
        print(f"[RAG CHUNKS] Total {len(rag_sources)} chunks selected for reader:")
        for idx, chunk in enumerate(rag_sources, 1):
            title = chunk.get("title", "N/A")
            url = chunk.get("url", "N/A")
            chunk_idx = chunk.get("chunk_index", "N/A")
            text_preview = chunk.get("text", "")[:400]
            print("-" * 80)
            print(f"[Chunk {idx}] title={title} | url={url} | chunk_index={chunk_idx}")
            print(f"Text preview:\n{text_preview}")
        if not rag_sources:
            print("[RAG CHUNKS] No chunks selected.")
        print("=" * 80)

        # Sufficiency Gate: đánh dấu low_confidence khi bằng chứng chưa đủ.
        try:
            log_step(
                "SUFFICIENCY_GATE",
                "INPUT",
                {
                    "question": question,
                    "chunk_count": len(rag_sources),
                    "chunks": compact_sources(rag_sources, "rag", limit=10),
                },
                params,
            )
            suff = self.sufficiency.check(question, rag_sources)
            params["sufficiency_score"] = suff.score
            strict_suff_min = float(params.get("strict_sufficiency_min_score", 0.35))
            strict_low_score = (
                _strict_evidence_required(question)
                and not bool(params.get("_allow_partial_answer"))
                and suff.score < strict_suff_min
            )
            params["low_confidence"] = (not suff.sufficient) or strict_low_score
            print(
                f"[SufficiencyGate] score={suff.score:.4f} sufficient={suff.sufficient} "
                f"used={suff.used_chunks} reason={suff.reason}"
            )
            log_step(
                "SUFFICIENCY_GATE",
                "OUTPUT",
                {
                    "score": suff.score,
                    "sufficient": suff.sufficient,
                    "used_chunks": suff.used_chunks,
                    "reason": suff.reason,
                    "low_confidence": params["low_confidence"],
                    "strict_low_score": strict_low_score,
                    "strict_sufficiency_min_score": strict_suff_min,
                },
                params,
            )
            if (
                params.get("low_confidence")
                and bool(params.get("strict_sufficiency_gate", True))
                and _strict_evidence_required(question)
                and not bool(params.get("_allow_partial_answer"))
            ):
                params["_hard_no_answer_text"] = "Tôi chưa tìm thấy thông tin phù hợp."
                log_step(
                    "SUFFICIENCY_GATE",
                    "OUTPUT",
                    {
                        "hard_stop": True,
                        "reason": "strict_evidence_required_but_insufficient",
                        "answer": params["_hard_no_answer_text"],
                    },
                    params,
                )
        except Exception as e:
            print(f"[SufficiencyGate] skipped due to error: {e}")
            params["low_confidence"] = False
            log_step(
                "SUFFICIENCY_GATE",
                "OUTPUT",
                {
                    "error": str(e),
                    "low_confidence": params["low_confidence"],
                },
                params,
            )

        log_step(
            "RAG_CONTEXT",
            "INPUT",
            {
                "chunk_count": len(rag_sources),
                "chunks": compact_sources(rag_sources, "rag", limit=10),
            },
            params,
        )
        context = SourceFormat()(rag_sources)
        prompt = READER_TEMPLATE.format(context=context, question=question)
        if params.get("_allow_partial_answer") and params.get("_multi_hop_slot_gate"):
            prompt = (
                "LUU Y BAT BUOC: hay uu tien [Bang fact da kiem tra tu multi-hop]. "
                "Neu o nao ghi CHUA TIM THAY TRONG NGUON thi phai noi ro la chua tim thay, "
                "khong lay diem chuan de dien vao hoc phi va khong tu suy doan.\n\n" + prompt
            )
        elif params.get("low_confidence"):
            prompt = (
                "LƯU Ý: bằng chứng truy xuất được có thể chưa đầy đủ. "
                "Nếu không đủ cơ sở hãy TRẢ LỜI RÕ 'Tôi chưa tìm thấy thông tin...' "
                "thay vì suy đoán.\n\n" + prompt
            )
        print("\n" + "=" * 80)
        log_step(
            "RAG_CONTEXT",
            "OUTPUT",
            {
                "context_chars": len(context),
                "context_preview": trace_preview(context, 1600),
                "prompt_chars": len(prompt),
                "low_confidence": params.get("low_confidence"),
            },
            params,
        )
        print("[RAG CONTEXT] Formatted chunks sent to reader:")
        print("-" * 80)
        print(context if context.strip() else "[Empty context]")
        print("=" * 80)
        if _debug_full_prompt_enabled():
            print("[FINAL PROMPT] Qwen4B input:")
            print("-" * 80)
            print(prompt)
            print("=" * 80 + "\n")
            _debug_block("READER_SYSTEM_PROMPT", READER_UNTRAINED_INSTRUCTION + READER_UNTRAINED_PREFIX, limit=_debug_int_env("BOT_DEBUG_PROMPT_CHARS", 20000))
            _debug_block("READER_CONTEXT", context, limit=_debug_int_env("BOT_DEBUG_PROMPT_CHARS", 20000))
            _debug_block("READER_FINAL_PROMPT", prompt, limit=_debug_int_env("BOT_DEBUG_PROMPT_CHARS", 20000))
        self.logger.start()
        pre_output: ModelPreOutput = {
            "generation_params": params,
            "web_sources": web_sources,
            "rag_sources": rag_sources,
            "extra_data": {
            },
            "result_url": stream_id,
        }
        log_step(
            "REQUEST",
            "OUTPUT",
            {
                "stream_id": stream_id,
                "web_sources": len(web_sources),
                "rag_sources": len(rag_sources),
                "prompt_chars": len(prompt),
                "low_confidence": params.get("low_confidence"),
            },
            params,
        )
        return prompt, pre_output
    
async def main():
    global ws_pipeline
    print("[Startup] Initializing API model...", flush=True)
    api_model = APIModel()

    print("[Startup] Building QA pipeline...", flush=True)
    ws_pipeline = CustomQA(api_model)
    print("[Startup] Starting retriever resources...", flush=True)
    await ws_pipeline.start()
    import uuid
    class ServerModelImplement(ServerModel):  
        def __init__(self) -> None:
            self.request_storage: dict[str, tuple[str, WorkerChatRequest, ModelPreOutput]] = {}
        async def pre_inference(self, request: WorkerChatRequest) -> ModelPreOutput:
            stream_id = str(uuid.uuid4())
            params = request["params"]

            def normalize_source_params() -> None:
                k_docs = int(params.get("k_docs") or 0)
                k_pages = int(params.get("k_pages") or 0)
                if k_docs <= 0 and k_pages <= 0 and not params.get("use_localdb"):
                    return

                raw_mode = str(params.get("source_mode") or "").lower()
                if raw_mode in {"auto", "local", "web", "hybrid"}:
                    source_mode = raw_mode
                elif bool(params.get("auto_source", True)):
                    source_mode = "auto"
                else:
                    use_local = bool(params.get("use_localdb"))
                    use_web = bool(params.get("use_websearch"))
                    if use_local and use_web:
                        source_mode = "hybrid"
                    elif use_local:
                        source_mode = "local"
                    elif use_web:
                        source_mode = "web"
                    else:
                        source_mode = "auto"

                params["source_mode"] = source_mode
                params["auto_source"] = source_mode == "auto"
                if source_mode in {"auto", "hybrid"}:
                    params["use_localdb"] = True
                    params["use_websearch"] = True
                elif source_mode == "local":
                    params["use_localdb"] = True
                    params["use_websearch"] = False
                elif source_mode == "web":
                    params["use_localdb"] = False
                    params["use_websearch"] = True

            normalize_source_params()
            params.setdefault("enable_quality_gate", False)
            params.setdefault("enable_pre_crawl_quality_gate", params["enable_quality_gate"])
            params.setdefault("enable_chunk_gate", params["enable_quality_gate"])
            params.setdefault("quality_log", params["enable_quality_gate"])
            params.setdefault("test_trace", params["quality_log"])
            params.setdefault("quality_min_score", 0.58)
            params.setdefault("quality_min_relevance", 0.25)
            params.setdefault("quality_min_trust", 0.50)
            params.setdefault("quality_semantic_weight", 0.65)
            params.setdefault("pre_crawl_quality_min_score", params["quality_min_score"])
            params.setdefault("chunk_gate_min_score", params["quality_min_score"])
            params.setdefault("quality_strict_mode", True)
            params.setdefault("quality_max_docs", params.get("k_pages", 3))
            params.setdefault("source_safety_filter", True)
            params.setdefault("quality_force_for_rich_media", True)
            params.setdefault("hybrid_retrieval", True)
            params.setdefault("auto_multi_hop", True)
            params.setdefault("multi_hop_complexity_threshold", 2)
            params.setdefault("reader_table_rescue_max_sources", 3)
            params.setdefault("reader_table_rescue_max_chars", 12000)
            _debug_block(
                "SERVER_REQUEST",
                _debug_json({
                    "stream_id": stream_id,
                    "text": request.get("text"),
                    "params": dict(params),
                    "forward_kwargs": request.get("forward_kwargs"),
                }),
                limit=0,
            )
            print(params)
            print(
                f"[QualityGate] enable_quality_gate={params['enable_quality_gate']} | "
                f"pre_crawl={params['enable_pre_crawl_quality_gate']} | "
                f"chunk_gate={params['enable_chunk_gate']} | "
                f"quality_log={params['quality_log']} | "
                f"min_relevance={params['quality_min_relevance']} | "
                f"min_trust={params['quality_min_trust']}"
            )
            prompt, pre_output = await ws_pipeline.pre_inference(
                request["text"],
                stream_id,
                request["params"]
            ) 
            self.request_storage[stream_id] = (prompt, request, pre_output)
            return pre_output
        async def inference(self, stream_id: str) -> AsyncGenerator[str, None]:
            prompt, request, pre_output = self.request_storage.pop(stream_id)
            generator = ws_pipeline.inference(prompt, request)
            total = ""
            try:
                async for chunk in generator:
                    total += chunk
                    yield chunk
            finally:
                # Store chat data when finish
                model_output: ModelOutput = {
                    **pre_output,
                    "text": total
                }
                data: WorkerStoreChatData = {
                    "forward_kwargs": request["forward_kwargs"],
                    "model_output": model_output
                }
                _debug_block(
                    "FINAL_RESPONSE",
                    _debug_json({
                        "stream_id": stream_id,
                        "response_chars": len(total),
                        "response": total,
                        "rag_sources": len(pre_output.get("rag_sources", [])),
                        "web_sources": len(pre_output.get("web_sources", [])),
                    }),
                    limit=0,
                )
                _debug_block(
                    "STORE_CHAT_DATA_SUMMARY",
                    _debug_json({
                        "stream_id": stream_id,
                        "forward_kwargs": request.get("forward_kwargs"),
                        "model_output_keys": list(model_output.keys()),
                    }),
                    limit=0,
                )
                await self.store(data)
                
    server_model = ServerModelImplement()
    app = construct_app(
        server_domain=DOMAIN,
        info=CLIENT_INFO,
        server_model=server_model,
        init_tasks=[],
        shutdown_tasks=[],
        is_local=IS_LOCAL,
        deploy_url=DEPLOY_URL
    )
    # CORS policy
    from fastapi.middleware.cors import CORSMiddleware
    origins = [
        "http://127.0.0.1:8000",
        "https://uniadmission.me",
        DOMAIN
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    import uvicorn

    uvicorn_config = uvicorn.Config(app, port=NGROK_PORT)
    uvicorn_server = uvicorn.Server(uvicorn_config)
    await uvicorn_server.serve()
    
asyncio.run(main())
