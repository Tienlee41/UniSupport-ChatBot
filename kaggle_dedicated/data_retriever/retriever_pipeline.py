from .schema import WebSource, RagSource, FileSource, AbstractSearchEngine, SearchResult, HtmlResult
from .search_engines import BraveSearchEngine, GoogleSearchEngine
from .ranker import PageRerankModelProtocol, ChunkRanker, ChunkProcessor
from .downloader import PageDowloader
from .extractor import ContentExtractor
from .retriever import Splitter, FaissRetriever, Merger
from .config import *
from .retriever.utils import CmdLogger
from .snippet_checker import HeuristicSnippetChecker, SnippetCheckerProtocol
import time
import math
from server import GenerationParams
import asyncio
import aiohttp
from typing import cast
from concurrent.futures import ThreadPoolExecutor
import re
import unicodedata
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from .test_trace import compact_sources, log_step, preview

class DataRetrieverPipeline:
    def __init__(
        self, 
        page_ranker_model: PageRerankModelProtocol, 
        concurrent_config: DataRetrieverConcurrentConig | None = None, 
        websearch_config: WebsearchConfig | None = None,
        splitter_config: SplitterConfig | None = None,
        rag_config: RagConfig | None = None,
        table_merge_config: MergeTableConfig | None = None,
        neighbor_merge_config: MergeNeighborConfig | None = None,
        chunk_ranker_config: ChunkRankerConfig | None = None,
        chunk_processor_config: ChunkProcessorConfig | None = None
    ) -> None:
        # Config
        self._concurrent_config = concurrent_config or DataRetrieverConcurrentConig()
        self._query_semaphore = asyncio.Semaphore(self._concurrent_config.engine_query_limit)
        
        # Thread pool for CPU-bound tasks
        self._thread_pool = ThreadPoolExecutor(max_workers=8)
        
        # Websearch
        self._websearch_config = websearch_config or WebsearchConfig()
        self._brave_search_engine: AbstractSearchEngine = BraveSearchEngine()
        self._google_search_engine: AbstractSearchEngine = GoogleSearchEngine()
        
        # Page ranker
        self._page_ranker_model = page_ranker_model

        # Splitter
        self._splitter = Splitter(splitter_config or SplitterConfig())
        self._rag = FaissRetriever(rag_config or RagConfig())
        self._merger = Merger(
            neighbor_merge_config or MergeNeighborConfig(),
            table_merge_config or MergeTableConfig()
        )
        
        # Chunk ranker + processor
        # Try to reuse shared reranker from page_ranker_model to save VRAM
        shared_ranker = None
        shared_ranker_device = None
        if hasattr(page_ranker_model, 'shared_reranker'):
            try:
                shared_ranker = page_ranker_model.shared_reranker
                shared_ranker_device = getattr(page_ranker_model, 'shared_reranker_device', None)
                print(f"[DataRetrieverPipeline] Using shared reranker from page_ranker_model to save VRAM")
            except Exception as e:
                print(f"[DataRetrieverPipeline] Failed to get shared reranker: {e}, will load separately")
        
        self._chunk_ranker = ChunkRanker(
            chunk_ranker_config or ChunkRankerConfig(),
            shared_ranker=shared_ranker,
            shared_ranker_device=shared_ranker_device
        )
        # Chia sẻ embedding đã load trong FaissRetriever để MMR/dense-dedup không nạp lại model
        self._chunk_processor = ChunkProcessor(
            chunk_processor_config or ChunkProcessorConfig(),
            embedding=getattr(self._rag, "embedding", None),
        )
        
        # Snippet checker
        self._snippet_checker: SnippetCheckerProtocol = HeuristicSnippetChecker(
            min_snippet_length=150,
            min_keyword_match_ratio=0.4
        )
        
        self.logger = CmdLogger("Retriever")

    @property
    def chunk_ranker(self) -> "ChunkRanker":
        """Expose cho SufficiencyGate tái sử dụng cross-encoder đã load."""
        return self._chunk_ranker
    
    def _preview_text(self, text: str, limit: int = 200) -> str:
        text = text.replace("\n", " ").strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    def _normalize_gate_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFD", text or "")
        without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        without_marks = without_marks.replace("đ", "d").replace("Đ", "D")
        return re.sub(r"\s+", " ", without_marks.lower()).strip()

    def _gate_tokens(self, text: str) -> set[str]:
        return set(re.findall(r"\w+", self._normalize_gate_text(text), flags=re.UNICODE))

    def _canonical_url(self, url: str) -> str:
        if not url:
            return ""
        parsed = urlparse(url)
        filtered_query = [
            (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=False)
            if not k.lower().startswith("utm_")
        ]
        normalized = parsed._replace(path=(parsed.path.rstrip("/") or "/"), query=urlencode(filtered_query), fragment="")
        return urlunparse(normalized)

    def _domain_trust(self, url: str) -> float:
        domain = (urlparse(url).netloc or "").lower()
        if not domain:
            return 0.0
        if domain.endswith(".edu.vn") or domain.endswith(".gov.vn"):
            return 1.0
        if ".edu" in domain or ".gov" in domain:
            return 0.9
        if domain.endswith(".org"):
            return 0.75
        if any(bad in domain for bad in ["forum", "blogspot", "wordpress"]):
            return 0.35
        return 0.6

    def _surface_overlap(self, query: str, result: SearchResult) -> float:
        query_tokens = self._gate_tokens(query)
        if not query_tokens:
            return 0.0
        surface = f"{result.get('title', '')} {result.get('description', '')} {result.get('url', '')}"
        surface_tokens = self._gate_tokens(surface)
        return len(query_tokens.intersection(surface_tokens)) / max(1, len(query_tokens))

    def _education_level_mismatch_reason(self, query: str, surface: str) -> str | None:
        query_norm = self._normalize_gate_text(query)
        surface_norm = self._normalize_gate_text(surface)
        graduate_terms = [
            "thac si", "cao hoc", "sau dai hoc", "nghien cuu sinh",
            "tien si", "dao tao thac si", "dao tao tien si",
            "master", "masters", "graduate", "postgraduate", "phd",
        ]
        if any(term in surface_norm for term in graduate_terms) and not any(term in query_norm for term in graduate_terms):
            return "graduate_source_for_undergraduate_query"
        return None

    def _pre_crawl_safety_reason(self, query: str, result: SearchResult) -> str | None:
        domain = (urlparse(result.get("url", "")).netloc or "").lower()
        surface = f"{result.get('title', '')} {result.get('description', '')} {result.get('url', '')}"
        surface_norm = self._normalize_gate_text(surface)
        query_norm = self._normalize_gate_text(query)

        hard_bad_domains = [
            "vieclam", "topcv", "career", "jobs", "jobstreet", "timviec",
            "viec-lam", "vieclamtot", "123job", "itviec",
        ]
        admission_terms = [
            "diem chuan", "diem trung tuyen", "diem xet tuyen", "diem san",
            "hoc phi", "tuyen sinh", "xet tuyen", "nganh dao tao",
            "admission", "tuition",
        ]
        job_terms = [
            "tim viec", "viec lam", "tuyen dung", "nhan vien",
            "thuc tap sinh", "e-commerce executive",
        ]
        asks_admission = any(term in query_norm for term in admission_terms)
        if asks_admission and any(bad in domain for bad in hard_bad_domains):
            return f"bad_domain_for_admission:{domain}"
        if asks_admission and sum(1 for term in job_terms if term in surface_norm) >= 2:
            return "job_content_for_admission_query"
        if asks_admission:
            work_abroad_terms = [
                "eps", "xuat canh", "nguoi lao dong", "lao dong ngoai nuoc",
                "han quoc", "visa", "ky quy", "giao duc dinh huong",
                "dao tao dinh huong", "phai cu", "hop dong lao dong",
            ]
            if any(term in surface_norm for term in work_abroad_terms):
                return "work_abroad_training_source_for_admission_query"
            if "ha noi" in query_norm:
                outside_hanoi_terms = [
                    "dai hoc hoa sen", "hoasen", "tp ho chi minh", "thanh pho ho chi minh",
                    "dai hoc da nang", "viet han", "da nang", "tp.hcm", "tphcm",
                ]
                if any(term in surface_norm for term in outside_hanoi_terms):
                    return "outside_hanoi_source_for_hanoi_query"
        level_reason = self._education_level_mismatch_reason(query, surface)
        if asks_admission and level_reason:
            return level_reason
        asks_score = any(term in query_norm for term in ["diem chuan", "diem trung tuyen", "diem xet tuyen"])
        if asks_score and ("a00" in query_norm or "khoi a" in query_norm):
            explicit_method = any(term in query_norm for term in [
                "hoc ba", "ccqt", "ket hop", "dgnl", "danh gia nang luc",
                "dgtd", "danh gia tu duy", "tsa", "hsa", "aptitude",
            ])
            non_thpt_method = any(term in surface_norm for term in [
                "hoc ba", "ccqt", "ket hop", "xet tuyen ket hop",
                "dgnl", "danh gia nang luc", "dgtd", "danh gia tu duy",
                "tsa", "hsa", "aptitude",
            ])
            thpt_score_signal = any(term in surface_norm for term in [
                "diem thi thpt", "thi tot nghiep thpt", "tot nghiep thpt",
                "xet tuyen thpt", "phuong thuc thpt",
            ])
            if non_thpt_method and not explicit_method and not thpt_score_signal:
                return "non_thpt_score_method_for_plain_a00_query"
        asks_tuition = any(term in query_norm for term in ["hoc phi", "tuition", "muc thu"])
        if asks_tuition:
            asks_policy = any(term in query_norm for term in ["mien giam", "hoc bong", "ho tro", "tro cap"])
            waiver_hits = sum(1 for term in [
                "mien giam", "hoc bong", "ho tro chi phi", "chinh sach phat trien",
                "nguoi hoc tai nang", "tro cap", "cap bu hoc phi",
            ] if term in surface_norm)
            schedule_signal = any(term in surface_norm for term in [
                "muc thu hoc phi", "dinh muc hoc phi", "thong bao hoc phi",
                "quy dinh hoc phi", "don gia hoc phi", "hoc phi nam hoc",
                "dong/tin chi", "dong / tin chi",
            ])
            if waiver_hits and not asks_policy and not schedule_signal:
                return "waiver_support_source_for_tuition_query"
        return None

    def _pre_crawl_quality_gate(
        self,
        params: GenerationParams,
        query: str,
        search_results: list[SearchResult],
    ) -> list[SearchResult]:
        enabled = bool(params.get("enable_pre_crawl_quality_gate", params.get("enable_quality_gate", False)))
        log_step(
            "QUALITY_GATE",
            "INPUT",
            {
                "query": query,
                "enabled": enabled,
                "candidate_count": len(search_results),
                "candidates": compact_sources(search_results, "search", limit=10),
            },
            params,
        )
        if not enabled or not search_results:
            log_step(
                "QUALITY_GATE",
                "OUTPUT",
                {
                    "enabled": enabled,
                    "reason": "disabled" if not enabled else "empty_candidates",
                    "passed_count": len(search_results),
                    "passed": compact_sources(search_results, "search", limit=10),
                },
                params,
            )
            return search_results

        quality_log = bool(params.get("quality_log", enabled))
        alpha = max(0.0, min(1.0, float(params.get("quality_semantic_weight", 0.65))))
        min_score = float(params.get("pre_crawl_quality_min_score", params.get("quality_min_score", 0.58)))
        min_relevance = float(params.get("quality_min_relevance", 0.25))
        min_trust = float(params.get("quality_min_trust", 0.50))
        strict = bool(params.get("quality_strict_mode", True))
        score_filter_enabled = bool(params.get("llm_rerank", False))

        max_page_score = max((float(result.get("score", 0.0) or 0.0) for result in search_results), default=0.0)
        accepted: list[SearchResult] = []
        seen: set[str] = set()

        if quality_log:
            self.logger.log(
                "[QualityGate] Pre-crawl after page rerank -> "
                f"min_score>={min_score:.2f}, relevance>={min_relevance:.2f}, "
                f"trust>={min_trust:.2f}, alpha={alpha:.2f}, score_filter={score_filter_enabled}"
            )

        for idx, result in enumerate(search_results, 1):
            url = result.get("url", "")
            canonical = self._canonical_url(url)
            if canonical and canonical in seen:
                if quality_log:
                    self.logger.log(f"[QualityGate] [{idx:02d}] DROP | reason=duplicate_url | url={url}")
                continue
            safety_reason = self._pre_crawl_safety_reason(query, result)
            if safety_reason:
                if quality_log:
                    self.logger.log(
                        f"[QualityGate] [{idx:02d}] DROP | reason={safety_reason} | "
                        f"title={result.get('title', '')[:80]} | url={url}"
                    )
                continue

            semantic = float(result.get("score", 0.0) or 0.0)
            semantic_norm = semantic / max_page_score if max_page_score > 0 else 0.0
            semantic_norm = max(0.0, min(1.0, semantic_norm))
            overlap = self._surface_overlap(query, result)
            relevance = alpha * semantic_norm + (1.0 - alpha) * overlap
            trust = self._domain_trust(url)
            final_score = 0.75 * relevance + 0.25 * trust
            if score_filter_enabled:
                is_ok = (
                    final_score >= min_score and relevance >= min_relevance and trust >= min_trust
                    if strict else
                    final_score >= min_score or (relevance >= min_relevance and trust >= min_trust)
                )
            else:
                # When LLM page rerank is disabled, Google/Brave scores are not a semantic
                # confidence signal. Keep pre-crawl gate as a safety/dedupe gate only.
                is_ok = True

            if quality_log:
                status = "PASS" if is_ok else "DROP"
                self.logger.log(
                    f"[QualityGate] [{idx:02d}] {status} | final={final_score:.3f} "
                    f"rel={relevance:.3f} sem={semantic_norm:.3f} overlap={overlap:.3f} "
                    f"trust={trust:.3f} | title={result.get('title', '')[:80]} | url={url}"
                )

            if is_ok:
                accepted.append(result)
                if canonical:
                    seen.add(canonical)

        if quality_log:
            self.logger.log(f"[QualityGate] Pre-crawl result -> pages={len(accepted)}/{len(search_results)}")
        log_step(
            "QUALITY_GATE",
            "OUTPUT",
            {
                "enabled": enabled,
                "input_count": len(search_results),
                "passed_count": len(accepted),
                "dropped_count": len(search_results) - len(accepted),
                "passed": compact_sources(accepted, "search", limit=10),
            },
            params,
        )
        return accepted
    
    def _log_chunks(self, prefix: str, title: str, chunks: list[RagSource], max_samples: int = 3):
        if not chunks:
            self.logger.log(f"{prefix} {title}: no chunks")
            return
        self.logger.log(f"{prefix} {title}: total {len(chunks)} chunks")
        for chunk in chunks[:max_samples]:
            preview = self._preview_text(chunk.get("text", ""))
            self.logger.log(f"{prefix} chunk#{chunk.get('chunk_index', 0)}: {preview}")
        if len(chunks) > max_samples:
            self.logger.log(f"{prefix} ... (+{len(chunks) - max_samples} more)")
    
    def _prioritize_table_chunks(
        self,
        total_sources: list[RagSource],
        retrieved_sources: list[RagSource],
        max_additional: int = 8
    ) -> list[RagSource]:
        """Ensure chunks that contain tables are always included"""
        table_chunks: list[RagSource] = []
        existing_indexes = {source.get("chunk_index") for source in retrieved_sources}
        for source in total_sources:
            if self._is_table_chunk(source):
                if source.get("chunk_index") not in existing_indexes:
                    table_chunks.append(source)
            if len(table_chunks) >= max_additional:
                break
        if table_chunks:
            self.logger.log(f"[Prioritize] Added {len(table_chunks)} table chunks before retrieval output")
        return table_chunks + retrieved_sources

    def _is_table_chunk(self, source: RagSource) -> bool:
        text = source.get("text", "") or ""
        return (
            source.get("content_type") == "table"
            or "[BANG]" in text
            or text.count("|") >= 3
        )

    def _is_table_query(self, query: str) -> bool:
        q = self._normalize_gate_text(query)
        table_terms = [
            "hoc phi", "muc thu", "tin chi", "diem chuan", "diem trung tuyen",
            "diem xet tuyen", "diem san", "chi tieu", "ma nganh", "to hop",
            "danh sach", "so sanh", "xep hang", "top",
        ]
        return any(term in q for term in table_terms)

    def _expand_table_context(
        self,
        total_sources: list[RagSource],
        selected_sources: list[RagSource],
        query: str,
        max_table_chunks: int = 12,
    ) -> list[RagSource]:
        """Keep sibling table chunks for broad table questions."""
        if not selected_sources or not self._is_table_query(query):
            return selected_sources
        if not any(self._is_table_chunk(source) for source in selected_sources):
            return selected_sources
        if not any(self._is_table_chunk(source) for source in total_sources):
            return selected_sources

        kept = {(source.get("url", ""), source.get("chunk_index")) for source in selected_sources}
        expanded: list[RagSource] = []
        for source in sorted(total_sources, key=lambda item: item.get("chunk_index", 0)):
            if not self._is_table_chunk(source):
                continue
            key = (source.get("url", ""), source.get("chunk_index"))
            if key in kept:
                continue
            expanded.append(source)
            kept.add(key)
            if len(expanded) >= max_table_chunks:
                break

        if expanded:
            self.logger.log(
                f"[TableContext] Expanded table context with {len(expanded)} chunk(s) "
                f"for query: {query[:120]}"
            )
        return expanded + selected_sources

    def _restore_protected_table_chunks(
        self,
        before_rerank: list[RagSource],
        after_rerank: list[RagSource],
        max_restore: int = 8,
    ) -> list[RagSource]:
        kept = {(source.get("url", ""), source.get("chunk_index")) for source in after_rerank}
        restored: list[RagSource] = []
        for source in before_rerank:
            key = (source.get("url", ""), source.get("chunk_index"))
            if key in kept or not self._is_table_chunk(source):
                continue
            restored.append(source)
            kept.add(key)
            if len(restored) >= max_restore:
                break
        if restored:
            self.logger.log(f"[Prioritize] Restored {len(restored)} table chunks after chunk rerank")
        return restored + after_rerank
    
    async def start(self):
        self._aio_session = aiohttp.ClientSession()
        # Page downloader
        self._page_downloader = PageDowloader(self._aio_session, self._concurrent_config.page_download_limit, self._websearch_config.page_timeout)
        # Page extractor
        self._page_extractor = ContentExtractor(self._aio_session, self._concurrent_config.file_download_limit, self._websearch_config.file_timeout, self._websearch_config.max_file_per_page)
    async def stop(self):
        await self._aio_session.close()
        self._thread_pool.shutdown(wait=False)
    async def retrieve_sep(
        self,
        params: GenerationParams,
        queries_and_domains: list[str | tuple[str, list[str]]]
    ) -> tuple[list[WebSource], list[RagSource]]:
        if len(queries_and_domains) == 0: return [], []
        tasks = []
        async def task(query: str, school_domains: list[str]):
            async with self._query_semaphore:
                return await self.retrieve_single_page(params, query, school_domains)
        for item in queries_and_domains: #type:ignore
            if isinstance(item, str):
                # Only query, no shool domain
                tasks.append(asyncio.create_task(task(item, [])))
            else:
                tasks.append(asyncio.create_task(task(item[0], item[1])))
        results = await asyncio.gather(*tasks)
        web_sources: list[WebSource] = []
        rag_sources: list[RagSource] = []
        for item in results:
            web_sources.extend(item[0])
            rag_sources.extend(item[1])
        return web_sources, rag_sources
    async def retrieve(
        self,
        params: GenerationParams,
        queries_and_domains: list[str | tuple[str, list[str]]]
    ) -> tuple[list[WebSource], list[RagSource]]:
        query_count = len(queries_and_domains)
        if query_count == 0:
            return [], []

        include_pdf = params.get("include_pdf", False)
        # Clone params to avoid mutating caller reference
        params = cast(GenerationParams, {**params})
        if "llm_rerank" not in params:
            params["llm_rerank"] = query_count > 1
            mode = "ON" if params["llm_rerank"] else "OFF"
            self.logger.log(
                f"Auto llm_rerank={mode} (query_count={query_count})"
            )
        else:
            mode = "ON" if params["llm_rerank"] else "OFF"
            self.logger.log(
                f"Manual llm_rerank={mode} (query_count={query_count})"
            )
        if "chunk_rerank" not in params:
            params["chunk_rerank"] = query_count > 1
            mode = "ON" if params["chunk_rerank"] else "OFF"
            self.logger.log(
                f"Auto chunk_rerank={mode} (query_count={query_count})"
            )
        else:
            mode = "ON" if params["chunk_rerank"] else "OFF"
            self.logger.log(
                f"Manual chunk_rerank={mode} (query_count={query_count})"
            )
        log_step(
            "RETRIEVAL_CONFIG",
            "INPUT",
            {
                "queries": [
                    item if isinstance(item, str) else {"query": item[0], "school_domains": item[1]}
                    for item in queries_and_domains
                ],
                "query_count": query_count,
                "engine_type": params.get("engine_type", "brave"),
                "llm_rerank": params.get("llm_rerank"),
                "chunk_rerank": params.get("chunk_rerank"),
                "k_pages": params.get("k_pages"),
                "k_docs": params.get("k_docs"),
                "include_pdf": params.get("include_pdf"),
                "include_image": params.get("include_image"),
                "quality_gate": params.get("enable_pre_crawl_quality_gate", params.get("enable_quality_gate")),
                "chunk_gate": params.get("enable_chunk_gate", params.get("enable_quality_gate")),
            },
            params,
        )

        tasks = []
        async def search_and_rerank_task(query: str, school_domains: list[str]):
            async with self._query_semaphore:
                return await self._search_and_rerank(params, query, school_domains)
        for item in queries_and_domains: #type:ignore
            if isinstance(item, str):
                # Only query, no shool domain
                tasks.append(asyncio.create_task(search_and_rerank_task(item, [])))
            else:
                tasks.append(asyncio.create_task(search_and_rerank_task(item[0], item[1])))
                
        search_results_list: list[list[SearchResult]] = await asyncio.gather(*tasks)
        html_results_list = await self._download(params, search_results_list)
        
        # PARALLEL: Process và RAG chạy song song cho các queries khác nhau
        # Tạo tasks cho process và RAG
        async def process_and_rag_task(
            html_results: list[HtmlResult], 
            query: str, 
            include_pdf: bool,
            include_image: bool,
        ) -> tuple[list[WebSource], list[list[RagSource]]]:
            # Process pages
            web_sources = await self._process(html_results, include_pdf, include_image, params)
            # RAG processing
            rag_sources_list = await self._split_rag_merge(query, web_sources, params)
            return web_sources, rag_sources_list
        
        # Chạy process và RAG song song cho tất cả queries
        process_rag_tasks = [
            asyncio.create_task(
                process_and_rag_task(
                    html_results,
                    item if isinstance(item, str) else item[0],
                    include_pdf,
                    params.get("include_image", False),
                )
            )
            for item, html_results in zip(queries_and_domains, html_results_list)
        ]
        
        process_rag_results = await asyncio.gather(*process_rag_tasks)
        web_sources_list = [result[0] for result in process_rag_results]
        rag_sources_list_list = [result[1] for result in process_rag_results]
        
        web_sources_list = self._pages_list_reorder(web_sources_list)
        
        # Per-sub-question pipeline: pass both combined query (fallback) và list queries gốc.
        query_list = [
            item if isinstance(item, str) else item[0]
            for item in queries_and_domains
        ]
        combined_query = " ".join(query_list)
        rag_sources = await self._merge_rag_source(
            rag_sources_list_list, combined_query, queries=query_list, params=params
        )
        web_sources = await self._merge_web_sources(web_sources_list)
        log_step(
            "RETRIEVAL_CONFIG",
            "OUTPUT",
            {
                "web_sources": len(web_sources),
                "rag_sources": len(rag_sources),
                "web_preview": compact_sources(web_sources, "web", limit=5),
                "rag_preview": compact_sources(rag_sources, "rag", limit=5),
            },
            params,
        )
        return web_sources, rag_sources
    def _pages_list_reorder(
        self,
        web_sources_list: list[list[WebSource]]
    ) -> list[list[WebSource]]:
        web_sources_list = sorted(
            web_sources_list,
            key=lambda web_sources: web_sources[0]["score"] if len(web_sources) > 0 else 0,
            reverse=True
        )
        return web_sources_list
    async def _search_and_rerank(
        self,
        params: GenerationParams,
        query: str,
        school_domains: list[str]
    ) -> list[SearchResult]:
        # Websearch
        self.logger.start()
        engine_type = params.get("engine_type", "brave")
        domain_restrict = params.get("domain_restrict", False)
        time_metric = params.get("time_metric")
        time_range = params.get("time_range")
        time_year = params.get("time_year")
        time_year_start = params.get("time_year_start")
        time_year_end = params.get("time_year_end")
        search_results: list[SearchResult] = []
        if engine_type == "brave":
            search_func = self._brave_search_engine.search
        else:
            search_func = self._google_search_engine.search
        log_step(
            "WEB_SEARCH",
            "INPUT",
            {
                "query": query,
                "engine_type": engine_type,
                "domain_restrict": domain_restrict,
                "school_domains": school_domains,
                "time_metric": time_metric,
                "time_range": time_range,
                "time_year": time_year,
                "time_year_start": time_year_start,
                "time_year_end": time_year_end,
            },
            params,
        )
        search_results = await search_func(
            query=query,
            domain_restrict=domain_restrict,
            school_domains=school_domains,
            time_metric=time_metric,
            time_range=time_range,
            time_year=time_year,
            time_year_start=time_year_start,
            time_year_end=time_year_end
        )
        self.logger.end("Websearch")
        log_step(
            "WEB_SEARCH",
            "OUTPUT",
            {
                "query": query,
                "result_count": len(search_results),
                "results": compact_sources(search_results, "search", limit=10),
            },
            params,
        )
        # Rerank Page
        use_rerank = params.get("llm_rerank", True)
        log_step(
            "WEB_SEARCH_LLM_RERANK",
            "INPUT",
            {
                "query": query,
                "enabled": use_rerank,
                "relative_threshold": params.get("page_score_threshold", 0.51),
                "candidate_count": len(search_results),
                "candidates": compact_sources(search_results, "search", limit=10),
            },
            params,
        )
        if use_rerank:
            self.logger.start()
            page_score_threshold = params.get("page_score_threshold", 0.51)
            search_results = await self._page_ranker_model.rerank_page(
                pages=search_results,
                query=query,
                relative_threshold=page_score_threshold,
                params=params
            )
            self.logger.end("Rerank")
        else:
            self.logger.log("Rerank: Skip (llm_rerank=False)")
        log_step(
            "WEB_SEARCH_LLM_RERANK",
            "OUTPUT",
            {
                "query": query,
                "enabled": use_rerank,
                "result_count": len(search_results),
                "results": compact_sources(search_results, "search", limit=10),
            },
            params,
        )
        search_results = self._pre_crawl_quality_gate(params, query, search_results)
        return search_results
    async def _download(
        self,
        params: GenerationParams,
        search_results_list: list[list[SearchResult]],
    ) -> list[list[HtmlResult]]:
        """Optimized parallel download with snippet optimization"""
        use_snippet_optimization = params.get("use_snippet_optimization", True)
        k_pages = params.get("k_pages", 3)
        include_pdf = params.get("include_pdf", False)
        include_image = params.get("include_image", False)
        log_step(
            "CRAWL_EXTRACT",
            "INPUT",
            {
                "query_groups": len(search_results_list),
                "k_pages_per_query": k_pages,
                "include_pdf": include_pdf,
                "include_image": include_image,
                "search_results": [
                    compact_sources(group, "search", limit=10)
                    for group in search_results_list
                ],
            },
            params,
        )
        
        # Dedupe URLs across all lists
        seen_urls: set[str] = set()
        filtered_list: list[list[SearchResult]] = []
        for search_results in search_results_list:
            filtered = [sr for sr in search_results if sr["url"] not in seen_urls]
            for sr in filtered:
                seen_urls.add(sr["url"])
            filtered_list.append(filtered)
        
        # Flatten all search results for parallel processing
        all_results: list[SearchResult] = [sr for srs in filtered_list for sr in srs]
        
        snippet_results: dict[str, HtmlResult] = {}
        crawl_results: list[SearchResult] = []
        
        if use_snippet_optimization and all_results:
            self.logger.start()
            log_step(
                "SNIPPET_CHECK",
                "INPUT",
                {
                    "enabled": True,
                    "candidate_count": len(all_results),
                    "candidates": compact_sources(all_results, "search", limit=10),
                },
                params,
            )
            # Check ALL snippets in parallel at once
            async def check_one(sr: SearchResult) -> tuple[SearchResult, bool]:
                is_ok = await self._snippet_checker.is_sufficient(
                    snippet=sr.get("description", ""),
                    title=sr.get("title", ""),
                    query=sr.get("query", ""),
                    url=sr.get("url", ""),
                    params=params
                )
                return sr, is_ok
            
            checks = await asyncio.gather(*[check_one(sr) for sr in all_results])
            
            for sr, is_sufficient in checks:
                if is_sufficient:
                    snippet_results[sr["url"]] = {
                        **sr,
                        "html": f"{sr.get('title', '')}\n\n{sr.get('description', '')}",
                        "score": sr.get("score", 1.0)
                    }
                else:
                    crawl_results.append(sr)
            
            self.logger.end("Snippet Check")
            log_step(
                "SNIPPET_CHECK",
                "OUTPUT",
                {
                    "snippet_accepted_count": len(snippet_results),
                    "crawl_needed_count": len(crawl_results),
                    "snippet_accepted": compact_sources(list(snippet_results.values()), "html", limit=10),
                    "crawl_needed": compact_sources(crawl_results, "search", limit=10),
                },
                params,
            )
        else:
            crawl_results = all_results
            log_step(
                "SNIPPET_CHECK",
                "OUTPUT",
                {
                    "enabled": False,
                    "reason": "disabled_or_empty",
                    "crawl_needed_count": len(crawl_results),
                },
                params,
            )
        
        # Crawl all needed URLs in ONE batch (parallel)
        crawl_start = time.time()
        crawled: dict[str, HtmlResult] = {}
        
        if crawl_results:
            # Chỉ gửi TOP k_pages URL (theo thứ tự đã rerank) sang ScrapingBee
            max_pages = min(k_pages * len(search_results_list), len(crawl_results))
            self.logger.log(f"[Download] Crawling {max_pages} URLs (from {len(crawl_results)} candidates)...")
            html_list = await self._page_downloader.download(
                crawl_results, max_pages, include_pdf, include_image
            )
            for hr in html_list:
                crawled[hr["url"]] = hr
        
        crawl_time = time.time() - crawl_start
        
        # Merge all results
        all_html = {**crawled, **snippet_results}
        
        # Build final results respecting k_pages per query
        final_list: list[list[HtmlResult]] = []
        for search_results in search_results_list:
            final: list[HtmlResult] = []
            for sr in search_results:
                if sr["url"] in all_html and len(final) < k_pages:
                    final.append(all_html[sr["url"]])
            final_list.append(final)
        
        self.logger.log(f"[Download] Snippet: {len(snippet_results)}, Crawled: {len(crawled)} in {crawl_time:.2f}s")
        log_step(
            "CRAWL_EXTRACT",
            "OUTPUT",
            {
                "snippet_count": len(snippet_results),
                "crawled_count": len(crawled),
                "crawl_time_sec": round(crawl_time, 3),
                "html_groups": [
                    compact_sources(group, "html", limit=10)
                    for group in final_list
                ],
            },
            params,
        )
        return final_list
    async def _process(
        self,
        html_results: list[HtmlResult],
        include_pdf: bool,
        include_image: bool,
        params: GenerationParams,
    ) -> list[WebSource]:
        # Process page
        self.logger.start()
        log_step(
            "CONTENT_EXTRACT",
            "INPUT",
            {
                "html_count": len(html_results),
                "include_pdf": include_pdf,
                "include_image": include_image,
                "html_results": compact_sources(html_results, "html", limit=10),
            },
            params,
        )
        web_sources: list[WebSource] = await self._page_extractor.extract(
            html_results, include_pdf, include_image
        )
        self.logger.end("Process")
        log_step(
            "CONTENT_EXTRACT",
            "OUTPUT",
            {
                "web_source_count": len(web_sources),
                "web_sources": compact_sources(web_sources, "web", limit=10),
            },
            params,
        )
        return web_sources
    async def _split_rag_merge(
        self,
        query: str,
        web_sources: list[WebSource],
        params: GenerationParams
    ) -> list[list[RagSource]]:
        """Split, RAG, merge - fully parallel using thread pool"""
        self.logger.start()
        merge_table = params.get("merge_table", True)
        merge_neighbor = params.get("merge_neighbor", True)
        chunk_score_threshold = params.get("chunk_score_threshold", 0.5)
        chunk_rerank_enabled = params.get("chunk_rerank", True)
        k_docs = params.get("k_docs", 5)
        
        if not web_sources:
            self.logger.end("RAG")
            return []
        
        scores = [source["score"] for source in web_sources]
        total_scores = sum(scores)
        if total_scores == 0:
            page_k_docs = [max(1, k_docs // len(scores))] * len(scores)
        else:
            page_k_docs = [max(1, math.ceil(s / total_scores * k_docs)) for s in scores]
        
        loop = asyncio.get_event_loop()
        
        def process_single_source(web_source: WebSource, page_k_doc: int) -> list[RagSource]:
            """Sync function to run in thread pool"""
            log_step(
                "CHUNKING",
                "INPUT",
                {
                    "query": query,
                    "page_k_doc": page_k_doc,
                    "source": {
                        "title": web_source.get("title", ""),
                        "url": web_source.get("url", ""),
                        "score": web_source.get("score"),
                        "text_chars": len(web_source.get("text", "") or ""),
                        "text_preview": preview(web_source.get("text", ""), 220),
                    },
                },
                params,
            )
            rag_sources = self._splitter.split(web_source)
            log_step(
                "CHUNKING",
                "OUTPUT",
                {
                    "query": query,
                    "source_url": web_source.get("url", ""),
                    "raw_chunk_count": len(rag_sources),
                    "raw_chunks_preview": compact_sources(rag_sources, "rag", limit=5),
                },
                params,
            )
            if not rag_sources:
                return []
            relavent = self._rag.retrieve(rag_sources, query, page_k_doc)
            relavent = self._prioritize_table_chunks(rag_sources, relavent)
            relavent = self._merger.merge(rag_sources, relavent, merge_table, merge_neighbor)
            log_step(
                "RERANK_CHUNK",
                "INPUT",
                {
                    "query": query,
                    "enabled": chunk_rerank_enabled,
                    "relative_threshold": chunk_score_threshold,
                    "candidate_count": len(relavent),
                    "candidates": compact_sources(relavent, "rag", limit=8),
                },
                params,
            )
            if chunk_rerank_enabled and relavent:
                before_rerank = relavent
                # For web: use relative threshold (threshold = max_score * chunk_score_threshold)
                relavent = self._chunk_ranker.rerank_chunks(relavent, query, relative_threshold=chunk_score_threshold, use_relative_threshold=True)
                relavent = self._restore_protected_table_chunks(before_rerank, relavent)
            relavent = self._expand_table_context(rag_sources, relavent, query)
            log_step(
                "RERANK_CHUNK",
                "OUTPUT",
                {
                    "query": query,
                    "enabled": chunk_rerank_enabled,
                    "result_count": len(relavent),
                    "results": compact_sources(relavent, "rag", limit=8),
                },
                params,
            )
            return relavent
        
        # Run all in parallel using thread pool
        tasks = [
            loop.run_in_executor(self._thread_pool, process_single_source, ws, kd)
            for ws, kd in zip(web_sources, page_k_docs)
        ]
        rag_sources_list = await asyncio.gather(*tasks)
        
        self.logger.end("RAG")
        return list(rag_sources_list)
    async def _merge_rag_source(
        self,
        rag_sources_list_list: list[list[list[RagSource]]],
        query: str,
        queries: list[str] | None = None,
        params: GenerationParams | None = None,
    ) -> list[RagSource]:
        """Combine all ragsource, deduplicate, rank, compress.

        Nâng cấp v5: truyền cả `queries` (danh sách sub-query) xuống ChunkProcessor
        để thực hiện per-sub-question budgeting + entity/time/MMR boost.
        """
        # Step 1: Flatten and remove exact duplicates by (url, chunk_index)
        url_indexes: dict[str, set[int]] = {}
        all_chunks: list[RagSource] = []
        for rag_sources_list in rag_sources_list_list:
            for rag_sources in rag_sources_list:
                for rag_source in rag_sources:
                    chunk_index = rag_source["chunk_index"]
                    url = rag_source["url"]
                    if url not in url_indexes:
                        url_indexes[url] = {chunk_index}
                        all_chunks.append(rag_source)
                    elif chunk_index not in url_indexes[url]:
                        url_indexes[url].add(chunk_index)
                        all_chunks.append(rag_source)
        log_step(
            "DEDUP",
            "INPUT",
            {
                "combined_query": query,
                "query_count": len(queries) if queries else 1,
                "group_count": len(rag_sources_list_list),
                "candidate_chunks_after_exact_dedup": len(all_chunks),
                "candidates": compact_sources(all_chunks, "rag", limit=12),
            },
            params or {},
        )

        # Step 2: Apply chunk processor (dedupe, rank, MMR, compress)
        loop = asyncio.get_event_loop()
        processed = await loop.run_in_executor(
            self._thread_pool,
            lambda: self._chunk_processor.process(all_chunks, query, queries=queries),
        )

        self.logger.log(
            f"[ChunkProcessor] {len(all_chunks)} -> {len(processed)} chunks "
            f"(queries={len(queries) if queries else 1})"
        )
        log_step(
            "DEDUP",
            "OUTPUT",
            {
                "input_chunks": len(all_chunks),
                "output_chunks": len(processed),
                "queries": queries or [query],
                "chunks": compact_sources(processed, "rag", limit=12),
            },
            params or {},
        )
        return processed
    async def _merge_web_sources(
        self,
        web_sources_list: list[list[WebSource]]
    ) -> list[WebSource]:
        urls = set()
        final_web_sources: list[WebSource] = []
        for web_sources in web_sources_list:
            for web_source in web_sources:
                if web_source["url"] not in urls:
                    urls.add(web_source["url"])
                    final_web_sources.append(web_source)
        return final_web_sources
    async def retrieve_single_page(
        self,
        params: GenerationParams,
        query: str,
        school_domains: list[str]
    ) -> tuple[list[WebSource], list[RagSource]]:
        # Websearch
        self.logger.start()
        engine_type = params.get("engine_type", "brave")
        domain_restrict = params.get("domain_restrict", False)
        time_metric = params.get("time_metric")
        time_range = params.get("time_range")         
        search_results: list[SearchResult] = []
        if engine_type == "brave":
            search_func = self._brave_search_engine.search
        else:
            search_func = self._google_search_engine.search
        log_step(
            "WEB_SEARCH",
            "INPUT",
            {
                "query": query,
                "engine_type": engine_type,
                "domain_restrict": domain_restrict,
                "school_domains": school_domains,
                "time_metric": time_metric,
                "time_range": time_range,
            },
            params,
        )
        search_results = await search_func(
            query=query,
            domain_restrict=domain_restrict,
            school_domains=school_domains,
            time_metric=time_metric,
            time_range=time_range
        )
        self.logger.end("Websearch")
        log_step(
            "WEB_SEARCH",
            "OUTPUT",
            {
                "query": query,
                "result_count": len(search_results),
                "results": compact_sources(search_results, "search", limit=10),
            },
            params,
        )
        # Rerank Page
        use_rerank = params.get("llm_rerank", True)
        log_step(
            "WEB_SEARCH_LLM_RERANK",
            "INPUT",
            {
                "query": query,
                "enabled": use_rerank,
                "relative_threshold": params.get("page_score_threshold", 0.5),
                "candidate_count": len(search_results),
                "candidates": compact_sources(search_results, "search", limit=10),
            },
            params,
        )
        if use_rerank:
            self.logger.start()
            page_score_threshold = params.get("page_score_threshold", 0.5)
            search_results = await self._page_ranker_model.rerank_page(
                pages=search_results,
                query=query,
                relative_threshold=page_score_threshold,
                params=params
            )
            self.logger.end("Rerank")
        else:
            self.logger.log("Rerank: Skip (llm_rerank=False)")
        log_step(
            "WEB_SEARCH_LLM_RERANK",
            "OUTPUT",
            {
                "query": query,
                "enabled": use_rerank,
                "result_count": len(search_results),
                "results": compact_sources(search_results, "search", limit=10),
            },
            params,
        )
        search_results = self._pre_crawl_quality_gate(params, query, search_results)
        # Download page
        self.logger.start()
        k_pages = params.get("k_pages", 3) # Todo: Split by query priority
        include_pdf = params.get("include_pdf", False)
        include_image = params.get("include_image", False)
        log_step(
            "CRAWL_EXTRACT",
            "INPUT",
            {
                "query": query,
                "k_pages": k_pages,
                "include_pdf": include_pdf,
                "include_image": include_image,
                "search_results": compact_sources(search_results, "search", limit=10),
            },
            params,
        )
        html_results: list[HtmlResult] = await self._page_downloader.download(
            search_results, 
            k_pages, 
            include_pdf, 
            include_image
        )
        self.logger.end("Download")
        log_step(
            "CRAWL_EXTRACT",
            "OUTPUT",
            {
                "html_count": len(html_results),
                "html_results": compact_sources(html_results, "html", limit=10),
            },
            params,
        )
        # Process page
        self.logger.start()
        log_step(
            "CONTENT_EXTRACT",
            "INPUT",
            {
                "html_count": len(html_results),
                "include_pdf": include_pdf,
                "include_image": include_image,
                "html_results": compact_sources(html_results, "html", limit=10),
            },
            params,
        )
        web_sources: list[WebSource] = await self._page_extractor.extract(
            html_results, include_pdf, include_image
        )
        self.logger.end("Process")
        log_step(
            "CONTENT_EXTRACT",
            "OUTPUT",
            {
                "web_source_count": len(web_sources),
                "web_sources": compact_sources(web_sources, "web", limit=10),
            },
            params,
        )
        # Split, rag, merge
        self.logger.start()
        merge_table = params.get("merge_table", True)
        merge_neighbor = params.get("merge_neighbor", True)
        chunk_score_threshold = params.get("chunk_score_threshold", 0.5)
        chunk_rerank_enabled = params.get("chunk_rerank", True)
        k_docs = params.get("k_docs", 5) # Todo: Split by query priority
        
        rag_sources: list[RagSource] = []
        scores = [source["score"] for source in web_sources]
        total_scores = sum(scores) 
        if total_scores == 0: # When reranker fail
            page_k_docs = [math.ceil(k_docs/len(scores)) for _ in scores]
        else:
            page_k_docs = [math.ceil(confidence/total_scores*k_docs) for confidence in scores]
        for web_source, page_k_doc in zip(web_sources, page_k_docs):
            log_step(
                "CHUNKING",
                "INPUT",
                {
                    "query": query,
                    "page_k_doc": page_k_doc,
                    "source": {
                        "title": web_source.get("title", ""),
                        "url": web_source.get("url", ""),
                        "score": web_source.get("score"),
                        "text_chars": len(web_source.get("text", "") or ""),
                        "text_preview": preview(web_source.get("text", ""), 220),
                    },
                },
                params,
            )
            rag_sources = self._splitter.split(web_source)
            log_step(
                "CHUNKING",
                "OUTPUT",
                {
                    "query": query,
                    "source_url": web_source.get("url", ""),
                    "raw_chunk_count": len(rag_sources),
                    "raw_chunks_preview": compact_sources(rag_sources, "rag", limit=5),
                },
                params,
            )
            relavent_sources = self._rag.retrieve(rag_sources, query, page_k_doc)
            relavent_sources = self._prioritize_table_chunks(rag_sources, relavent_sources)
            relavent_sources = self._merger.merge(rag_sources, relavent_sources, merge_table, merge_neighbor)
            log_step(
                "RERANK_CHUNK",
                "INPUT",
                {
                    "query": query,
                    "enabled": chunk_rerank_enabled,
                    "relative_threshold": chunk_score_threshold,
                    "candidate_count": len(relavent_sources),
                    "candidates": compact_sources(relavent_sources, "rag", limit=8),
                },
                params,
            )
            if chunk_rerank_enabled:
                before_rerank = relavent_sources
                # For web: use relative threshold (threshold = max_score * chunk_score_threshold)
                relavent_sources = self._chunk_ranker.rerank_chunks(relavent_sources, query, relative_threshold=chunk_score_threshold, use_relative_threshold=True)
                relavent_sources = self._restore_protected_table_chunks(before_rerank, relavent_sources)
            relavent_sources = self._expand_table_context(rag_sources, relavent_sources, query)
            log_step(
                "RERANK_CHUNK",
                "OUTPUT",
                {
                    "query": query,
                    "enabled": chunk_rerank_enabled,
                    "result_count": len(relavent_sources),
                    "results": compact_sources(relavent_sources, "rag", limit=8),
                },
                params,
            )
            rag_sources = relavent_sources
        
        self.logger.end("RAG")
        return web_sources, rag_sources 
