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
        self._chunk_processor = ChunkProcessor(chunk_processor_config or ChunkProcessorConfig())
        
        # Snippet checker
        self._snippet_checker: SnippetCheckerProtocol = HeuristicSnippetChecker(
            min_snippet_length=150,
            min_keyword_match_ratio=0.4
        )
        
        self.logger = CmdLogger("Retriever")
    
    def _preview_text(self, text: str, limit: int = 200) -> str:
        text = text.replace("\n", " ").strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "..."
    
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
        max_additional: int = 3
    ) -> list[RagSource]:
        """Ensure chunks that contain tables are always included"""
        table_chunks: list[RagSource] = []
        existing_indexes = {source["chunk_index"] for source in retrieved_sources}
        for source in total_sources:
            text = source.get("text", "")
            if "[BẢNG]" in text or text.count("|") >= 3:
                if source["chunk_index"] not in existing_indexes:
                    table_chunks.append(source)
            if len(table_chunks) >= max_additional:
                break
        if table_chunks:
            self.logger.log(f"[Prioritize] Added {len(table_chunks)} table chunks before retrieval output")
        return table_chunks + retrieved_sources
    
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
            web_sources = await self._process(html_results, include_pdf, include_image)
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
        
        # Combine queries for chunk processing
        combined_query = " ".join(
            item if isinstance(item, str) else item[0] 
            for item in queries_and_domains
        )
        rag_sources = await self._merge_rag_source(rag_sources_list_list, combined_query) 
        web_sources = await self._merge_web_sources(web_sources_list)
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
        # Rerank Page
        use_rerank = params.get("llm_rerank", True)
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
        else:
            crawl_results = all_results
        
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
        return final_list
    async def _process(
        self,
        html_results: list[HtmlResult],
        include_pdf: bool,
        include_image: bool,
    ) -> list[WebSource]:
        # Process page
        self.logger.start()
        web_sources: list[WebSource] = await self._page_extractor.extract(
            html_results, include_pdf, include_image
        )
        self.logger.end("Process")
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
            rag_sources = self._splitter.split(web_source)
            if not rag_sources:
                return []
            relavent = self._rag.retrieve(rag_sources, query, page_k_doc)
            relavent = self._prioritize_table_chunks(rag_sources, relavent)
            relavent = self._merger.merge(rag_sources, relavent, merge_table, merge_neighbor)
            if chunk_rerank_enabled and relavent:
                # For web: use relative threshold (threshold = max_score * chunk_score_threshold)
                relavent = self._chunk_ranker.rerank_chunks(relavent, query, relative_threshold=chunk_score_threshold, use_relative_threshold=True)
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
        query: str
    ) -> list[RagSource]:
        """Combine all ragsource, deduplicate, rank, compress"""
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
        
        # Step 2: Apply chunk processor (dedupe, rank, compress)
        processed = await asyncio.get_event_loop().run_in_executor(
            self._thread_pool,
            self._chunk_processor.process,
            all_chunks,
            query
        )
        
        self.logger.log(f"[ChunkProcessor] {len(all_chunks)} -> {len(processed)} chunks")
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
        search_results = await search_func(
            query=query,
            domain_restrict=domain_restrict,
            school_domains=school_domains,
            time_metric=time_metric,
            time_range=time_range
        )
        self.logger.end("Websearch")
        # Rerank Page
        use_rerank = params.get("llm_rerank", True)
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
        # Download page
        self.logger.start()
        k_pages = params.get("k_pages", 3) # Todo: Split by query priority
        include_pdf = params.get("include_pdf", False)
        include_image = params.get("include_image", False)
        html_results: list[HtmlResult] = await self._page_downloader.download(
            search_results, 
            k_pages, 
            include_pdf, 
            include_image
        )
        self.logger.end("Download")
        # Process page
        self.logger.start()
        web_sources: list[WebSource] = await self._page_extractor.extract(
            html_results, include_pdf, include_image
        )
        self.logger.end("Process")
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
            rag_sources = self._splitter.split(web_source)
            relavent_sources = self._rag.retrieve(rag_sources, query, page_k_doc)
            relavent_sources = self._merger.merge(rag_sources, relavent_sources, merge_table, merge_neighbor)
            if chunk_rerank_enabled:
                # For web: use relative threshold (threshold = max_score * chunk_score_threshold)
                relavent_sources = self._chunk_ranker.rerank_chunks(relavent_sources, query, relative_threshold=chunk_score_threshold, use_relative_threshold=True)
            rag_sources = relavent_sources
        
        self.logger.end("RAG")
        return web_sources, rag_sources 