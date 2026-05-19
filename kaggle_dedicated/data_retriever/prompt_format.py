import re
from .schema import RagSource

class SourceFormat:
    """Format RAG sources to compact LLM context"""
    
    def __init__(self, use_separators: bool = True, compact: bool = True) -> None:
        self.use_separators = use_separators
        self.compact = compact
    
    def __call__(self, sources: list[RagSource]) -> str:
        if not sources:
            return ""
        
        # Group by URL
        url_groups: dict[str, list[RagSource]] = {}
        for s in sources:
            url_groups.setdefault(s["url"], []).append(s)
        
        result: list[str] = []
        
        for idx, (url, chunks) in enumerate(url_groups.items(), 1):
            chunks = sorted(chunks, key=lambda x: x["chunk_index"])
            title = chunks[0]["title"][:60]  # Truncate long titles
            
            # Compact header
            if self.compact:
                result.append(f"[{idx}] {title}")
                result.append(f"URL: {url}")
            else:
                result.append(f"### [{title}]({url})")
            
            # Combine chunks, prioritize tables
            tables = []
            texts = []
            for chunk in chunks:
                text = chunk["text"].strip()
                if "[BANG]" in text or text.count("|") >= 3:
                    tables.append(text)
                else:
                    texts.append(text)
            
            # Tables usually contain the exact admissions facts, so keep them
            # before surrounding prose in the reader context.
            for t in tables:
                result.append(t)
            for t in texts:
                result.append(t)
            
            if self.use_separators:
                result.append("---")
        
        formatted = "\n\n".join(result)
        
        # Cleanup
        formatted = re.sub(r'\n{3,}', '\n\n', formatted)
        formatted = re.sub(r'(\[BANG\]\s*)+', '[BANG]\n', formatted)
        formatted = re.sub(r'\[BANG\]\s*\[BANG\]', '[BANG]', formatted)
        # Remove empty table markers
        formatted = re.sub(r'\[BANG\]\s*\n\s*---', '---', formatted)
        
        return formatted.strip()
            
