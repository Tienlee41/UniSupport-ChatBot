from typing import Any
import re
from langchain_text_splitters.markdown import MarkdownTextSplitter, MarkdownHeaderTextSplitter
from transformers import AutoTokenizer
import tiktoken

TABLE_MARKER = "[BANG]"
TABLE_PATTERN = re.compile(rf'({TABLE_MARKER}\n(?:\|[^\n]+\|\n?)+)', re.MULTILINE)

class TokenMarkdownSplitter(MarkdownTextSplitter):
    def __init__(self, **kwargs: Any) -> None:
        tokenizer_name: str | None = kwargs.pop("tokenizer_name", None)
        device = kwargs.pop("device", "cpu")
        if tokenizer_name is None:
            raise ValueError("TokenMarkdownSplitter need tokenizer_name")
        if tokenizer_name.startswith("gpt"):
            self.tokenizer = tiktoken.encoding_for_model(tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, device=device)
        headers_to_split_on = kwargs.pop(
            "headers_to_split_on",
            [("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")]
        )
        kwargs["length_function"] = self._get_length
        super().__init__(**kwargs)
        self.header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        self._bold_pattern = re.compile(r"^\s*\*\*(.+?)\*\*\s*:?\s*$")
        self._table_line = re.compile(r"^\s*\|.*\|\s*$")

    def _get_length(self, text: str) -> int:
        if isinstance(self.tokenizer, tiktoken.Encoding):
            return len(self.tokenizer.encode(text))
        return len(self.tokenizer.encode(text, return_tensors=None))

    def _normalize_headings(self, text: str) -> str:
        lines = []
        for line in text.splitlines():
            m = self._bold_pattern.match(line.strip())
            lines.append(f"#### {m.group(1).strip()}" if m else line)
        return "\n".join(lines)

    def _mark_tables(self, text: str) -> str:
        """Mark tables with [BANG] marker, avoid duplicates"""
        lines, out, buf = text.splitlines(), [], []
        for line in lines:
            stripped = line.strip()
            # Skip existing [BANG] markers
            if stripped == TABLE_MARKER:
                continue
            if self._table_line.match(stripped):
                buf.append(line)
            else:
                if buf:
                    # Only add marker if last non-empty line isn't already [BANG]
                    last_non_empty = next((l for l in reversed(out) if l.strip()), "")
                    if last_non_empty != TABLE_MARKER:
                        if out and out[-1].strip():
                            out.append("")
                        out.append(TABLE_MARKER)
                    out.extend(buf)
                    out.append("")
                    buf.clear()
                out.append(line)
        if buf:
            last_non_empty = next((l for l in reversed(out) if l.strip()), "")
            if last_non_empty != TABLE_MARKER:
                if out and out[-1].strip():
                    out.append("")
                out.append(TABLE_MARKER)
            out.extend(buf)
        return "\n".join(out)

    def _extract_tables(self, text: str) -> tuple[list[str], str]:
        """Tach tables thanh atomic chunks, tra ve (table_chunks, remaining_text)"""
        tables = [m.group(1).strip() for m in TABLE_PATTERN.finditer(text)]
        remaining = TABLE_PATTERN.sub('\n\n', text)
        return tables, re.sub(r'\n{3,}', '\n\n', remaining)

    def _build_section(self, content: str, metadata: dict) -> str:
        headers = [metadata[h].strip() for h in ["H1","H2","H3","H4","H5","H6"] if metadata.get(h)]
        text = content.strip()
        if not headers:
            return text
        prefix = "#" * min(len(headers) + 2, 6) + " " + " > ".join(headers)
        return f"{prefix}\n\n{text}" if text else prefix

    def split_text(self, text: str) -> list[str]:
        text = self._normalize_headings(text)
        text = self._mark_tables(text)
        sections = self.header_splitter.split_text(text)
        if not sections:
            return MarkdownTextSplitter.split_text(self, text)
        
        chunks: list[str] = []
        for section in sections:
            section_text = self._build_section(
                section.page_content,
                getattr(section, "metadata", {}) or {}
            )
            if not section_text.strip():
                continue
            # Atomic table chunking - tach table ra rieng
            tables, remaining = self._extract_tables(section_text)
            if remaining.strip():
                chunks.extend(MarkdownTextSplitter.split_text(self, remaining))
            chunks.extend(tables)  # Tables khong bi split
        return chunks
