from sentence_transformers import CrossEncoder
import torch

from ..schema import RagSource
from ..config import ChunkRankerConfig

class ChunkRanker:
    """Cross-encoder reranker for Vietnamese"""
    def __init__(self, chunk_config: ChunkRankerConfig) -> None:
        self.chunk_config = chunk_config
        # Auto-detect device: use CPU if CUDA is out of memory, otherwise use configured device
        device = chunk_config.device
        if device == "cuda" and torch.cuda.is_available():
            try:
                # Try to allocate a small tensor to check if CUDA has memory
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.empty_cache()
            except RuntimeError:
                # CUDA out of memory, fallback to CPU
                device = "cpu"
                print(f"[ChunkRanker] CUDA out of memory, falling back to CPU")
        
        self.ranker = CrossEncoder(
            chunk_config.ranker_name, 
            max_length=chunk_config.max_length,
            device=device
        )
        self.device = device  # Store actual device used
        print(f"[ChunkRanker] Loaded model: {chunk_config.ranker_name} on {device}")
    
    def rerank_chunks(self, sources: list[RagSource], query: str, relative_threshold: float = 0.5) -> list[RagSource]:
        """Perform cross-encoder rerank (Per page)."""
        if not sources:
            return []
        
        print(f"[ChunkRanker] Query: {query[:80]}...")
        print(f"[ChunkRanker] Input chunks: {len(sources)}")
        
        # Prepare pairs for cross-encoder
        pairs = [(query, source["text"]) for source in sources]
        
        # Get scores from cross-encoder
        # Clear CUDA cache before prediction to avoid OOM
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        scores = self.ranker.predict(pairs)
        
        # Clear cache after prediction
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Combine sources with scores
        scored_sources = list(zip(sources, scores))
        
        # Use fixed absolute threshold of 0.5 instead of relative threshold
        max_score = max(scores) if scores.size > 0 else 0
        score_threshold = 0.5  # Fixed absolute threshold
        
        print(f"[ChunkRanker] Max score: {max_score:.4f}, Threshold: {score_threshold:.4f} (fixed)")
        
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
