"""Chunk post-processing: deduplication, compression, smart ranking"""
import re
from ..schema import RagSource
from ..config import ChunkProcessorConfig, EDU_DOMAINS, OFFICIAL_DOMAINS

class ChunkProcessor:
    def __init__(self, config: ChunkProcessorConfig | None = None) -> None:
        self.config = config or ChunkProcessorConfig()
        self._number_pattern = re.compile(r'\d[\d.,]*')
    
    def process(self, chunks: list[RagSource], query: str) -> list[RagSource]:
        """Full processing pipeline: dedupe -> rank -> compress"""
        if not chunks:
            return []
        chunks = self._deduplicate(chunks)
        chunks = self._smart_rank(chunks, query)
        chunks = self._compress(chunks)
        return chunks
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Fast similarity using word sets"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    def _deduplicate(self, chunks: list[RagSource]) -> list[RagSource]:
        """Remove duplicate/near-duplicate chunks"""
        if len(chunks) <= 1:
            return chunks
        
        unique: list[RagSource] = []
        seen_texts: list[str] = []
        
        for chunk in chunks:
            text = chunk["text"]
            is_duplicate = False
            
            for seen in seen_texts:
                if self._jaccard_similarity(text, seen) > self.config.similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(chunk)
                seen_texts.append(text)
        
        return unique
    
    def _get_chunk_score(self, chunk: RagSource, query: str) -> float:
        """Calculate chunk importance score"""
        score = 1.0
        text = chunk["text"]
        url = chunk.get("url", "")
        content_type = chunk.get("content_type", "text")
        
        # Table boost
        if content_type == "table" or "[BANG]" in text or text.count("|") >= 5:
            score *= self.config.table_boost
        
        # Numeric data boost
        numbers = self._number_pattern.findall(text)
        if len(numbers) >= 3:
            score *= self.config.numeric_boost
        
        # Domain boost
        for domain in EDU_DOMAINS:
            if domain in url:
                score *= self.config.edu_domain_boost
                break
        for domain in OFFICIAL_DOMAINS:
            if domain in url:
                score *= self.config.edu_domain_boost * 0.9  # Slightly less than edu
                break
        
        # Query relevance boost (simple keyword matching)
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        overlap = len(query_words & text_words)
        if overlap > 0:
            score *= 1 + (overlap / len(query_words)) * 0.3
        
        # Position boost (earlier chunks often have summaries)
        position = chunk.get("position", 0.5)
        if position < 0.3:
            score *= 1.1
        
        return score
    
    def _smart_rank(self, chunks: list[RagSource], query: str) -> list[RagSource]:
        """Rank chunks by importance"""
        scored = [(chunk, self._get_chunk_score(chunk, query)) for chunk in chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored]
    
    def _compress(self, chunks: list[RagSource]) -> list[RagSource]:
        """Limit chunks to max_total_chunks and approximate token limit"""
        if len(chunks) <= self.config.max_total_chunks:
            result = chunks
        else:
            result = chunks[:self.config.max_total_chunks]
        
        # Approximate token counting (4 chars ~ 1 token)
        total_chars = 0
        max_chars = self.config.max_tokens * 4
        final: list[RagSource] = []
        
        for chunk in result:
            chunk_chars = len(chunk["text"])
            if total_chars + chunk_chars > max_chars:
                break
            final.append(chunk)
            total_chars += chunk_chars
        
        return final


