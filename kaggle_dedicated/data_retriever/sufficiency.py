"""Sufficiency Gate: kiểm tra bằng chứng có đủ trước khi đưa vào reader.

Dùng chính CrossEncoder (BGE-reranker) đã load trong ChunkRanker để chấm điểm
cặp (question, evidence_concat) rồi so với threshold.

Nếu không đủ, pipeline có thể:
  - Gắn cờ `low_confidence` lên params để reader biết và thêm disclaimer.
  - (Optional) Trigger retry retrieval với rewrite (HyDE / step-back) — sẽ làm ở P3.
"""
from dataclasses import dataclass
from typing import Optional

from .schema import RagSource
from .config import SufficiencyConfig


@dataclass
class SufficiencyResult:
    sufficient: bool
    score: float
    used_chunks: int
    reason: str = ""


class SufficiencyGate:
    def __init__(self, chunk_ranker, config: Optional[SufficiencyConfig] = None):
        """
        Args:
            chunk_ranker: instance có method `score_pairs(list[tuple[str,str]]) -> list[float]`.
                          Reuse cross-encoder đã load để tiết kiệm VRAM.
            config: SufficiencyConfig.
        """
        self.ranker = chunk_ranker
        self.config = config or SufficiencyConfig()

    def check(self, question: str, chunks: list[RagSource]) -> SufficiencyResult:
        if not self.config.enabled:
            return SufficiencyResult(True, 1.0, 0, "disabled")
        if not chunks:
            return SufficiencyResult(False, 0.0, 0, "no_chunks")

        # Bỏ synthetic reasoning chunks khi check (chúng là tổng hợp, không
        # phải bằng chứng gốc, dễ làm cross-encoder chấm thấp).
        synth_prefix = (self.config.skip_synthetic_url_prefix or "").strip()
        real_chunks = chunks
        if synth_prefix:
            filtered = [
                c for c in chunks
                if not (c.get("url", "") or "").startswith(synth_prefix)
            ]
            # Chỉ apply filter khi còn ít nhất 1 chunk thật, để tránh wipe sạch.
            if filtered:
                real_chunks = filtered

        top = real_chunks[: self.config.top_k_for_check]
        concat = "\n\n".join(c.get("text", "") for c in top)
        if len(concat) > self.config.max_concat_chars:
            concat = concat[: self.config.max_concat_chars]

        try:
            scores = self.ranker.score_pairs([(question, concat)])
        except Exception as e:
            return SufficiencyResult(True, 0.0, len(top), f"error_fallback_{type(e).__name__}")

        score = scores[0] if scores else 0.0
        sufficient = score >= self.config.threshold
        reason = "ok" if sufficient else "below_threshold"

        # Override: cross-encoder thường chấm thấp với câu hỏi nhiều điều kiện
        # (so sánh, lọc, tổng hợp). Nếu evidence thật nhiều và max_score đủ
        # nhỏ-positive → vẫn coi là sufficient để reader tự tổng hợp.
        if (
            not sufficient
            and len(real_chunks) >= int(self.config.override_min_chunks)
            and score >= float(self.config.override_min_score)
        ):
            sufficient = True
            reason = (
                f"override_by_evidence_count={len(real_chunks)}"
                f"_score={score:.3f}"
            )
        return SufficiencyResult(sufficient, score, len(top), reason)
