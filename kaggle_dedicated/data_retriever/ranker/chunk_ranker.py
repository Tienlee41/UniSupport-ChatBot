from sentence_transformers import CrossEncoder
import torch

from ..schema import RagSource
from ..config import ChunkRankerConfig


class ChunkRanker:
    """Cross-encoder reranker for Vietnamese (BAAI/bge-reranker-v2-m3 mặc định)."""

    def __init__(self, chunk_config: ChunkRankerConfig, shared_ranker=None, shared_ranker_device=None) -> None:
        self.chunk_config = chunk_config

        # Tận dụng model đã load để tiết kiệm VRAM
        if shared_ranker is not None:
            self.ranker = shared_ranker
            self.device = shared_ranker_device or chunk_config.device
            print(f"[ChunkRanker] Reusing shared reranker model: {chunk_config.ranker_name} on {self.device}")
            return

        # Resolve runtime device: CUDA-or-CPU fallback.
        device = (chunk_config.device or "cpu").lower()
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                device = "cpu"
                print("[ChunkRanker] CUDA requested but unavailable, falling back to CPU")
            else:
                try:
                    test_tensor = torch.zeros(1, device="cuda")
                    del test_tensor
                    torch.cuda.empty_cache()
                except RuntimeError:
                    device = "cpu"
                    print("[ChunkRanker] CUDA out of memory, falling back to CPU")

        self.ranker = CrossEncoder(
            chunk_config.ranker_name,
            max_length=chunk_config.max_length,
            device=device,
        )
        self.device = device
        print(f"[ChunkRanker] Loaded model: {chunk_config.ranker_name} on {device}")

    # ──────────────────────────────────────────────────────────────────
    # Low-level: score pairs
    # ──────────────────────────────────────────────────────────────────
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Cho một list (query, text), trả score cross-encoder.

        Dùng cho SufficiencyGate và các component muốn reuse cross-encoder đã load.
        """
        if not pairs:
            return []
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        scores = self.ranker.predict(pairs)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [float(s) for s in scores]

    # ──────────────────────────────────────────────────────────────────
    # High-level: rerank chunks
    # ──────────────────────────────────────────────────────────────────
    def rerank_chunks(
        self,
        sources: list[RagSource],
        query: str,
        relative_threshold: float = 0.5,
        use_relative_threshold: bool = True,
        sort_by_score: bool = False,
    ) -> list[RagSource]:
        """Perform cross-encoder rerank (per-page hoặc per-subquery).

        Args:
            sources: RAG sources cần rerank.
            query: chuỗi truy vấn.
            relative_threshold: threshold value.
            use_relative_threshold:
                True  -> threshold = max_score * relative_threshold (tương đối, dùng cho web).
                False -> threshold = relative_threshold (tuyệt đối, dùng cho local DB).
            sort_by_score: nếu True bỏ `keep_order` của config, giữ thứ tự theo score
                           (giúp mitigate "lost-in-the-middle" cho reader).
        """
        if not sources:
            return []

        print(f"[ChunkRanker] Query: {query[:80]}...")
        print(f"[ChunkRanker] Input chunks: {len(sources)}")

        pairs = [(query, source["text"]) for source in sources]
        scores = self.score_pairs(pairs)
        scored_sources = list(zip(sources, scores))

        max_score = max(scores) if scores else 0.0
        if use_relative_threshold:
            score_threshold = max_score * relative_threshold
            threshold_type = "relative"
        else:
            score_threshold = relative_threshold
            threshold_type = "fixed"

        print(f"[ChunkRanker] Max score: {max_score:.4f}, Threshold: {score_threshold:.4f} ({threshold_type})")

        sorted_by_score = sorted(scored_sources, key=lambda x: x[1], reverse=True)
        print("[ChunkRanker] Top chunks by score:")
        for i, (src, score) in enumerate(sorted_by_score[:5]):
            text_preview = src["text"][:60].replace("\n", " ")
            status = "KEEP" if score >= score_threshold else "DROP"
            print(f"  [{i+1}] {score:.4f} ({status}) | {text_preview}...")

        # Lọc theo threshold. Danh sách đã sort theo score desc.
        valid_sources = [src for src, score in sorted_by_score if score >= score_threshold]

        # Nếu sort_by_score=True => giữ nguyên thứ tự score desc.
        # Ngược lại theo config.keep_order thì restore thứ tự gốc.
        if not sort_by_score and self.chunk_config.keep_order and valid_sources:
            source_to_idx = {id(src): idx for idx, src in enumerate(sources)}
            valid_sources.sort(key=lambda src: source_to_idx.get(id(src), 0))

        print(f"[ChunkRanker] Output chunks: {len(valid_sources)} (filtered {len(sources) - len(valid_sources)})")
        return valid_sources
