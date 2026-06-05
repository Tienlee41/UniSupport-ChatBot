"""Chunk post-processing: deduplication, compression, smart ranking.

Nâng cấp v5 (phát triển cho câu hỏi phức tạp):
- Dense deduplication (cosine trên embedding) thay thế / bổ sung Jaccard.
- Entity-aware boost/penalty theo `school` / `major` trong query.
- Time-aware decay theo khoảng cách năm giữa chunk và query.
- MMR (Maximal Marginal Relevance) để đảm bảo diversity.
- Per-sub-question budgeting khi xử lý multi-query.
"""
import math
import re
import unicodedata
from typing import Optional

from ..schema import RagSource
from ..config import ChunkProcessorConfig, EDU_DOMAINS, OFFICIAL_DOMAINS


class ChunkProcessor:
    def __init__(
        self,
        config: Optional[ChunkProcessorConfig] = None,
        embedding=None,  # HuggingFaceEmbeddings-like: .embed_query, .embed_documents
    ) -> None:
        self.config = config or ChunkProcessorConfig()
        self.embedding = embedding
        self._number_pattern = re.compile(r"\d[\d.,]*")
        self._year_pattern = re.compile(r"\b(19\d{2}|20\d{2})\b")
        # Một số school-id phổ biến (đồng bộ với router.py). Dùng để phát hiện entity.
        self._school_tokens = {
            "uet", "hus", "hust", "neu", "ftu", "tmu", "hlu", "hnue", "hmu",
            "haui", "ls", "ueb", "ulis", "ussh", "utc", "ptit", "act", "kma",
            "ajc", "vmu", "yds", "vnu",
        }

    def _norm(self, text: str) -> str:
        normalized = unicodedata.normalize("NFD", text or "")
        without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        return re.sub(r"\s+", " ", without_marks.lower()).strip()

    def _is_table_like(self, text: str, content_type: str = "") -> bool:
        return content_type == "table" or "[BANG]" in (text or "") or (text or "").count("|") >= 5

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────
    def process(
        self,
        chunks: list[RagSource],
        query: str,
        queries: Optional[list[str]] = None,
    ) -> list[RagSource]:
        """Pipeline đầy đủ: dedupe -> smart_rank -> MMR -> compress.

        Args:
            chunks: all retrieved chunks (đã gom dedup cơ bản bên ngoài).
            query: query gốc (hoặc combined). Dùng fallback khi không có `queries`.
            queries: danh sách sub-queries cho per-sub-q budgeting.
        """
        if not chunks:
            return []

        chunks = self._deduplicate(chunks)

        if queries and len(queries) > 1 and self.config.use_per_subq_budget:
            chunks = self._rank_with_per_subq(chunks, queries)
        else:
            chunks = self._smart_rank(chunks, query)

        if self.config.use_mmr and self.embedding is not None:
            chunks = self._mmr_select(chunks, query, self.config.max_total_chunks)

        chunks = self._compress(chunks)
        return chunks

    # ──────────────────────────────────────────────────────────────────
    # Deduplication
    # ──────────────────────────────────────────────────────────────────
    def _deduplicate(self, chunks: list[RagSource]) -> list[RagSource]:
        if len(chunks) <= 1:
            return chunks
        # Table chunks often share the same headers/units, so dense or Jaccard
        # deduplication can incorrectly collapse different rows of one PDF table.
        # Keep table-like chunks distinct by (url, chunk_index) and only dedupe
        # prose chunks aggressively.
        table_chunks: list[RagSource] = []
        prose_chunks: list[RagSource] = []
        seen_table_keys: set[tuple[str, int]] = set()
        for chunk in chunks:
            if self._is_table_like(chunk.get("text", ""), chunk.get("content_type", "")):
                key = (chunk.get("url", ""), int(chunk.get("chunk_index", -1)))
                if key not in seen_table_keys:
                    table_chunks.append(chunk)
                    seen_table_keys.add(key)
            else:
                prose_chunks.append(chunk)

        if not prose_chunks:
            return table_chunks

        if self.config.use_dense_dedup and self.embedding is not None:
            return table_chunks + self._dense_deduplicate(prose_chunks)
        return table_chunks + self._jaccard_deduplicate(prose_chunks)

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _jaccard_deduplicate(self, chunks: list[RagSource]) -> list[RagSource]:
        unique: list[RagSource] = []
        seen_texts: list[str] = []
        threshold = self.config.similarity_threshold
        for chunk in chunks:
            text = chunk["text"]
            if any(self._jaccard_similarity(text, seen) > threshold for seen in seen_texts):
                continue
            unique.append(chunk)
            seen_texts.append(text)
        return unique

    def _dense_deduplicate(self, chunks: list[RagSource]) -> list[RagSource]:
        try:
            texts = [c["text"] for c in chunks]
            embs = self.embedding.embed_documents(texts)
        except Exception as e:
            print(f"[ChunkProcessor] Dense dedup failed ({e}), fallback Jaccard")
            return self._jaccard_deduplicate(chunks)

        import numpy as np
        mat = np.asarray(embs, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat /= norms

        threshold = self.config.similarity_threshold
        unique: list[RagSource] = []
        kept_idx: list[int] = []
        for i, chunk in enumerate(chunks):
            is_dup = False
            for j in kept_idx:
                if float(mat[i] @ mat[j]) > threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(chunk)
                kept_idx.append(i)
        return unique

    # ──────────────────────────────────────────────────────────────────
    # Entity / Time boost
    # ──────────────────────────────────────────────────────────────────
    def _extract_query_entities(self, query: str) -> dict:
        q_lower = query.lower()
        schools = [s for s in self._school_tokens if re.search(rf"\b{re.escape(s)}\b", q_lower)]
        years = [int(y) for y in self._year_pattern.findall(query)]
        return {"schools": schools, "years": years}

    def _entity_time_multiplier(self, chunk: RagSource, entities: dict) -> float:
        mul = 1.0
        text_lower = chunk.get("text", "").lower()
        url_lower = chunk.get("url", "").lower()
        haystack = f"{text_lower} {url_lower}"

        # Entity (school) match
        if entities["schools"]:
            hits = [s for s in entities["schools"] if re.search(rf"\b{re.escape(s)}\b", haystack)]
            if hits:
                mul *= self.config.entity_match_boost
            else:
                mul *= self.config.entity_mismatch_penalty

        # Time decay
        if entities["years"]:
            target_year = max(entities["years"])  # năm "mục tiêu" là năm mới nhất được nhắc
            years_in_chunk = [int(y) for y in self._year_pattern.findall(chunk.get("text", ""))]
            if years_in_chunk:
                age_gap = min(abs(y - target_year) for y in years_in_chunk)
                decay = max(
                    self.config.time_decay_floor,
                    1.0 - self.config.time_decay_per_year * age_gap,
                )
                mul *= decay
        return mul

    # ──────────────────────────────────────────────────────────────────
    # Smart rank
    # ──────────────────────────────────────────────────────────────────
    def _get_chunk_score(self, chunk: RagSource, query: str, entities: Optional[dict] = None) -> float:
        score = 1.0
        text = chunk["text"]
        url = chunk.get("url", "")
        content_type = chunk.get("content_type", "text")
        text_norm = self._norm(text)
        query_norm = self._norm(query)

        # Table boost
        if content_type == "table" or "[BANG]" in text or "[BẢNG]" in text or text.count("|") >= 5:
            score *= self.config.table_boost

        # Numeric data boost
        numbers = self._number_pattern.findall(text)
        if len(numbers) >= 3:
            score *= self.config.numeric_boost

        score_terms = ["diem chuan", "diem trung tuyen", "diem xet tuyen", "diem san"]
        tuition_terms = ["hoc phi", "muc thu", "tin chi", "trieu", "dong/nam", "nam hoc"]
        if any(term in query_norm for term in score_terms):
            if any(term in text_norm for term in score_terms) or "a00" in text_norm or "to hop" in text_norm:
                score *= 1.45
        if any(term in query_norm for term in tuition_terms):
            if any(term in text_norm for term in tuition_terms):
                score *= 1.55
        if len(text.strip()) < 140 and not self._is_table_like(text, content_type) and len(numbers) < 2:
            score *= 0.35

        # Domain boost
        for domain in EDU_DOMAINS:
            if domain in url:
                score *= self.config.edu_domain_boost
                break
        for domain in OFFICIAL_DOMAINS:
            if domain in url:
                score *= self.config.edu_domain_boost * 0.9
                break

        # Query-overlap boost (keyword matching)
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        overlap = len(query_words & text_words)
        if overlap > 0 and query_words:
            score *= 1 + (overlap / len(query_words)) * 0.3

        # Position boost
        position = chunk.get("position", 0.5)
        if position < 0.3:
            score *= 1.1

        # Entity & Time (NEW)
        if entities is None:
            entities = self._extract_query_entities(query)
        score *= self._entity_time_multiplier(chunk, entities)

        return score

    def _smart_rank(self, chunks: list[RagSource], query: str) -> list[RagSource]:
        entities = self._extract_query_entities(query)
        scored = [(chunk, self._get_chunk_score(chunk, query, entities)) for chunk in chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored]

    def _rank_with_per_subq(self, chunks: list[RagSource], queries: list[str]) -> list[RagSource]:
        """Rank đảm bảo mỗi sub-query có ít nhất `min_chunks_per_subq` chunks top."""
        min_per = max(1, self.config.min_chunks_per_subq)
        reserved: list[RagSource] = []
        seen_ids = set()

        for q in queries:
            entities = self._extract_query_entities(q)
            scored = [(c, self._get_chunk_score(c, q, entities)) for c in chunks if id(c) not in seen_ids]
            scored.sort(key=lambda x: x[1], reverse=True)
            for chunk, _ in scored[:min_per]:
                reserved.append(chunk)
                seen_ids.add(id(chunk))

        # Xếp các chunk còn lại theo combined query
        combined = " ".join(queries)
        remaining = [c for c in chunks if id(c) not in seen_ids]
        combined_entities = self._extract_query_entities(combined)
        remaining_scored = [(c, self._get_chunk_score(c, combined, combined_entities)) for c in remaining]
        remaining_scored.sort(key=lambda x: x[1], reverse=True)

        return reserved + [c for c, _ in remaining_scored]

    # ──────────────────────────────────────────────────────────────────
    # MMR
    # ──────────────────────────────────────────────────────────────────
    def _mmr_select(self, chunks: list[RagSource], query: str, k: int) -> list[RagSource]:
        if len(chunks) <= k or self.embedding is None:
            return chunks[:k] if len(chunks) > k else chunks

        try:
            import numpy as np
            texts = [c["text"] for c in chunks]
            q_emb = np.asarray(self.embedding.embed_query(query), dtype=np.float32)
            c_embs = np.asarray(self.embedding.embed_documents(texts), dtype=np.float32)
        except Exception as e:
            print(f"[ChunkProcessor] MMR failed ({e}), skip")
            return chunks[:k]

        def _norm(v):
            n = (v @ v) ** 0.5
            return v / n if n > 0 else v

        q_emb = _norm(q_emb)
        c_norms = [_norm(v) for v in c_embs]

        rel = [float(q_emb @ cv) for cv in c_norms]

        lam = self.config.mmr_lambda
        selected_idx: list[int] = []
        remaining = list(range(len(chunks)))

        while remaining and len(selected_idx) < k:
            best, best_score = remaining[0], -1e9
            for i in remaining:
                if not selected_idx:
                    mmr_score = lam * rel[i]
                else:
                    max_sim = max(float(c_norms[i] @ c_norms[j]) for j in selected_idx)
                    mmr_score = lam * rel[i] - (1 - lam) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best = i
            selected_idx.append(best)
            remaining.remove(best)

        return [chunks[i] for i in selected_idx]

    # ──────────────────────────────────────────────────────────────────
    # Compression
    # ──────────────────────────────────────────────────────────────────
    def _compress(self, chunks: list[RagSource]) -> list[RagSource]:
        if len(chunks) > self.config.max_total_chunks:
            chunks = chunks[: self.config.max_total_chunks]

        total_chars = 0
        max_chars = self.config.max_tokens * 4
        final: list[RagSource] = []
        for chunk in chunks:
            n = len(chunk["text"])
            if total_chars + n > max_chars:
                if not final:
                    kept = dict(chunk)
                    kept["text"] = chunk["text"][:max_chars]
                    final.append(kept)  # type: ignore[arg-type]
                continue
            final.append(chunk)
            total_chars += n
        return final
