from dataclasses import dataclass
from typing import Callable, Awaitable, TypedDict, Literal

FILE_PREFIX = "[{title}]({url}):\n"

# Domain priority for source ranking
EDU_DOMAINS = [".edu.vn", ".edu", ".ac."]
OFFICIAL_DOMAINS = [".gov.vn", ".gov"]

@dataclass
class WebsearchConfig:
    page_timeout: float = 8  # Giam tu 10 -> 8
    file_timeout: float = 5
    max_file_per_page: int = 5  # Giam tu 10 -> 5

@dataclass
class MergeNeighborConfig:
    k_previous_chunks: int = 1
    k_next_chunks: int = 1

@dataclass
class MergeTableConfig:
    k_max_previous: int = 5
    k_max_next: int = 5
    separator_threshold: int = 1

@dataclass
class RagConfig:
    embedding_name: str = "intfloat/multilingual-e5-small"
    device: str = "cuda"
    keep_order: bool = True
    use_hybrid: bool = True
    hybrid_alpha: float = 0.7

@dataclass
class SplitterConfig:
    tokenizer_name: str = "Qwen/Qwen3-4B"
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_length: int = 5
    device: str = "cuda"

@dataclass
class ChunkRankerConfig:
    ranker_name: str = "BAAI/bge-reranker-v2-m3"
    max_length: int = 512
    keep_order: bool = True
    device: str = "cuda"

@dataclass 
class ChunkProcessorConfig:
    # Deduplication
    similarity_threshold: float = 0.85  # Chunks giong > 85% se bi loai
    use_dense_dedup: bool = True        # Dung cosine-sim tren embedding thay Jaccard khi co embedding
    # Compression
    max_total_chunks: int = 20  # Gioi han tong chunks
    max_tokens: int = 6000  # Gioi han tokens cho context
    # Smart ranking
    table_boost: float = 1.5  # Boost score cho table chunks
    numeric_boost: float = 1.2  # Boost score cho chunks co so lieu
    edu_domain_boost: float = 1.3  # Boost cho nguon .edu.vn
    # Entity-aware boost (NEW)
    entity_match_boost: float = 1.4     # Nhan khi chunk co chua school/major duoc hoi
    entity_mismatch_penalty: float = 0.3  # Nhan khi query co school cu the nhung chunk khong nhac toi
    # Time-aware boost (NEW)
    time_decay_per_year: float = 0.15   # Moi nam lech giam 15%, san 0.3
    time_decay_floor: float = 0.3
    # MMR diversity (NEW)
    use_mmr: bool = True
    mmr_lambda: float = 0.7             # 1.0 = tat (chi theo score), 0.0 = toi da diversity
    # Per-sub-question budgeting (NEW)
    use_per_subq_budget: bool = True
    min_chunks_per_subq: int = 2

@dataclass
class SufficiencyConfig:
    enabled: bool = True
    threshold: float = 0.10             # Hạ tu 0.35 -> 0.10: cross-encoder kho cham cao voi cau hoi co dieu kien phuc tap
    top_k_for_check: int = 8            # Tang tu 5 -> 8: lay nhieu evidence hon
    max_concat_chars: int = 4000        # Tang tu 3000 -> 4000
    # Bo qua chunks tong hop (synthetic) cua reasoning step.
    skip_synthetic_url_prefix: str = "multihop://"
    # Khi so chunks evidence that >= override_min_chunks va max_score >= override_min_score
    # -> coi la sufficient ngay du tong score thap.
    override_min_chunks: int = 6
    override_min_score: float = 0.05

@dataclass
class DataRetrieverConcurrentConig:
    engine_query_limit: int = 2  # Tang tu 1 -> 2 de search nhanh hon
    page_rerank_limit: int = 4  # Tang tu 2 -> 4
    page_download_limit: int = 8  # Tang tu 4 -> 8
    file_download_limit: int = 16
