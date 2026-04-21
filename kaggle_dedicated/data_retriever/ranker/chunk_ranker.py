from sentence_transformers import CrossEncoder
import torch

from ..schema import RagSource
from ..config import ChunkRankerConfig

class ChunkRanker:
    """Cross-encoder reranker for Vietnamese"""
    def __init__(self, chunk_config: ChunkRankerConfig, shared_ranker=None, shared_ranker_device=None) -> None:
        self.chunk_config = chunk_config
        
        # If shared reranker is provided, reuse it to save VRAM
        if shared_ranker is not None:
            self.ranker = shared_ranker
            self.device = shared_ranker_device or chunk_config.device
            print(f"[ChunkRanker] Reusing shared reranker model: {chunk_config.ranker_name} on {self.device}")
            return
        
        # Resolve runtime device: if CUDA is requested but unavailable/OOM, fallback to CPU.
        device = (chunk_config.device or "cpu").lower()
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                device = "cpu"
                print("[ChunkRanker] CUDA requested but unavailable, falling back to CPU")
            else:
                try:
                    # Try a tiny allocation to catch low-memory GPUs at startup.
                    test_tensor = torch.zeros(1, device="cuda")
                    del test_tensor
                    torch.cuda.empty_cache()
                except RuntimeError:
                    device = "cpu"
                    print("[ChunkRanker] CUDA out of memory, falling back to CPU")
        
        self.ranker = CrossEncoder(
            chunk_config.ranker_name, 
            max_length=chunk_config.max_length,
            device=device
        )
        self.device = device  # Store actual device used
        print(f"[ChunkRanker] Loaded model: {chunk_config.ranker_name} on {device}")
    
    def rerank_chunks(self, sources: list[RagSource], query: str, relative_threshold: float = 0.5, use_relative_threshold: bool = True) -> list[RagSource]:
        """
        Perform cross-encoder rerank (Per page).
        
        Args:
            sources: List of RAG sources to rerank
            query: Query string
            relative_threshold: Threshold value (0.5 by default)
            use_relative_threshold: If True, threshold = max_score * relative_threshold (relative)
                                   If False, threshold = relative_threshold (fixed absolute)
        """
        if not sources:
            return []
        
        print(f"[ChunkRanker] Query: {query[:80]}...")
        print(f"[ChunkRanker] Input chunks: {len(sources)}")
        
        # Prepare pairs for cross-encoder
        pairs = [(query, source["text"]) for source in sources]
        
        # Get scores from cross-encoder
        # Clear CUDA cache before prediction to avoid OOM
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        scores = self.ranker.predict(pairs)
        
        # Clear cache after prediction
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Combine sources with scores
        scored_sources = list(zip(sources, scores))
        
        # Calculate threshold based on mode
        max_score = max(scores) if scores.size > 0 else 0
        if use_relative_threshold:
            # Relative threshold: threshold = max_score * relative_threshold
            score_threshold = max_score * relative_threshold
            threshold_type = "relative"
        else:
            # Fixed absolute threshold: threshold = relative_threshold
            score_threshold = relative_threshold
            threshold_type = "fixed"
        
        print(f"[ChunkRanker] Max score: {max_score:.4f}, Threshold: {score_threshold:.4f} ({threshold_type})")
        
        # Log top scores
        sorted_by_score = sorted(scored_sources, key=lambda x: x[1], reverse=True)
        print("[ChunkRanker] Top chunks by score:")
        for i, (src, score) in enumerate(sorted_by_score[:5]):
            text_preview = src["text"][:60].replace("\n", " ")
            status = "KEEP" if score >= score_threshold else "DROP"
            print(f"  [{i+1}] {score:.4f} ({status}) | {text_preview}...")
        
        # Sort by score descending
        scored_sources.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold
        valid_sources = [src for src, score in scored_sources if score >= score_threshold]
        
        # Restore original order if needed
        if self.chunk_config.keep_order and valid_sources:
            source_to_idx = {id(src): idx for idx, src in enumerate(sources)}
            valid_sources.sort(key=lambda src: source_to_idx.get(id(src), 0))
        
        print(f"[ChunkRanker] Output chunks: {len(valid_sources)} (filtered {len(sources) - len(valid_sources)})")
        
        return valid_sources
