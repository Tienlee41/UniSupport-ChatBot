from typing import Literal, Optional, NotRequired
from typing_extensions import TypedDict

# All from core.types

SearchEngineType = Literal["google", "brave"]
ChatMessageRole = Literal["user", "bot"] # System instruction should stored per user message

class RagSource(TypedDict):
    query: str
    url: str
    title: str
    text: str
    chunk_index: int
    # Rich metadata
    total_chunks: NotRequired[int]
    content_type: NotRequired[str]  # text, table, heading
    position: NotRequired[float]  # 0.0 = dau, 1.0 = cuoi
    # File info
    file_url: NotRequired[str]
    file_title: NotRequired[str]
    file_type: NotRequired[str]
    

class FileSource(TypedDict):
    file_url: str
    file_title: str
    file_type: str
    text: str
    
class WebSource(TypedDict):
    query: str
    url: str
    title: str
    description: str
    text: str
    files: list[FileSource]
    score: float

class ModelInfo(TypedDict):
    name: str
    id: str

class GenerationParams(TypedDict):
    # Model
    model_id: str
    # Search
    use_websearch: NotRequired[bool]
    use_localdb: NotRequired[bool]
    # Source policy for retrieval:
    # - auto: multi-hop decomposer chooses local_db / web / hybrid per sub-question.
    # - local/web/hybrid: force every retrieval sub-question to the selected source mode.
    source_mode: NotRequired[Literal["auto", "local", "web", "hybrid"]]
    # Backward-compatible flag. True maps to source_mode=auto; False derives mode
    # from use_websearch/use_localdb when source_mode is not provided.
    auto_source: NotRequired[bool]
    max_query: NotRequired[int]
    query_score_threshold: NotRequired[float]
    engine_type: NotRequired[SearchEngineType] # google/brave
    domain_restrict: NotRequired[bool]
    school_domain: NotRequired[bool]
    time_metric: NotRequired[Literal["m", "y", "d"]]
    time_range: NotRequired[int]
    time_year: NotRequired[int]
    time_year_start: NotRequired[int]
    time_year_end: NotRequired[int]
    # Rerank
    llm_rerank: NotRequired[bool]
    page_score_threshold: NotRequired[float]
    chunk_score_threshold: NotRequired[float]
    # Retrieve
    k_docs: NotRequired[int]
    k_pages: NotRequired[int]
    page_rerank: NotRequired[bool]
    chunk_rerank: NotRequired[bool]
    include_pdf: NotRequired[bool]
    include_image: NotRequired[bool]
    merge_table: NotRequired[bool]
    merge_neighbor: NotRequired[bool]
    # Retrieval gates: QualityGate runs before crawl, ChunkGate runs after chunking.
    enable_quality_gate: NotRequired[bool]
    enable_pre_crawl_quality_gate: NotRequired[bool]
    enable_chunk_gate: NotRequired[bool]
    quality_log: NotRequired[bool]
    test_trace: NotRequired[bool]
    quality_min_score: NotRequired[float]
    quality_min_relevance: NotRequired[float]
    quality_min_trust: NotRequired[float]
    quality_semantic_weight: NotRequired[float]
    pre_crawl_quality_min_score: NotRequired[float]
    chunk_gate_min_score: NotRequired[float]
    quality_strict_mode: NotRequired[bool]
    quality_max_docs: NotRequired[int]
    # Sampling
    max_tokens: NotRequired[int]
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    top_k: NotRequired[int]
    # Other
    max_history: NotRequired[int]
    
class ModelPreOutput(TypedDict):
    """
    `result_url` should store id only when created. As it would later formatted to url in router.
    """
    generation_params: GenerationParams
    web_sources: list[WebSource]
    rag_sources: list[RagSource]
    extra_data: dict
    result_url: str 
    
class ModelOutput(ModelPreOutput):
    text: str
    
class ChatMessage(TypedDict):
    role: ChatMessageRole
    text: str
    
class WorkerServerInfo(TypedDict):
    name: str
    domain: str
    models: list[ModelInfo]   
    
class WorkerPreInferenceResponse(TypedDict):
    pre_output: ModelPreOutput
    info: WorkerServerInfo
    
class WorkerChatRequest(TypedDict):
    text: str
    params: GenerationParams
    history: list[ChatMessage]
    forward_kwargs: dict
    
class WorkerStoreChatData(TypedDict):
    forward_kwargs: dict
    model_output: ModelOutput
