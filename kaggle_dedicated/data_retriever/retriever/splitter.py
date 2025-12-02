from ..schema import WebSource, RagSource
from ..config import SplitterConfig
from .utils import TokenMarkdownSplitter, TABLE_MARKER

class Splitter:
    def __init__(self, config: SplitterConfig) -> None:
        self.splitter = TokenMarkdownSplitter(
            tokenizer_name=config.tokenizer_name, 
            device=config.device,
            chunk_size=config.chunk_size, 
            chunk_overlap=config.chunk_overlap
        )
        self.logging = True

    def _detect_type(self, text: str) -> str:
        if TABLE_MARKER in text or text.count("|") > 5:
            return "table"
        if text.strip().startswith("#"):
            return "heading"
        return "text"

    def split(self, web_source: WebSource) -> list[RagSource]:
        """Split WebSource to RagSource with rich metadata"""
        chunks = self.splitter.split_text(web_source["text"])
        total = len(chunks)
        if self.logging:
            print(f'Split {web_source["title"]}: {len(web_source["text"])} chars -> {total} chunks')
        
        return [
            {
                "query": web_source["query"],
                "title": web_source["title"],
                "url": web_source["url"],
                "text": chunk,
                "chunk_index": i,
                "total_chunks": total,
                "content_type": self._detect_type(chunk),
                "position": i / max(total - 1, 1)
            }
            for i, chunk in enumerate(chunks)
        ]