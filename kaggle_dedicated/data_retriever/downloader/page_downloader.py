import aiohttp
import os
import asyncio
import random
from typing import Awaitable, TYPE_CHECKING
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Comment

if TYPE_CHECKING:
    CoroutineType = asyncio._CoroutineLike
else:
    CoroutineType = Awaitable

from ..schema import SearchResult, HtmlResult

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
]
SCRAPINGBEE_API = "https://app.scrapingbee.com/api/v1/"


class PageDowloader:
    def __init__(self, session: aiohttp.ClientSession, concurrent_page_download: int, timeout: float) -> None:
        self.timeout = aiohttp.ClientTimeout(timeout)
        self.session = session
        self.semaphore = asyncio.Semaphore(concurrent_page_download)
        self._scrapingbee_key = os.getenv("SCRAPINGBEE_API_KEY")
        self._max_retry = int(os.getenv("SCRAPINGBEE_MAX_RETRY", "1"))
        self._delay_min = float(os.getenv("SCRAPINGBEE_DELAY_MIN", "1"))
        self._delay_max = float(os.getenv("SCRAPINGBEE_DELAY_MAX", "3"))
        self._api_call_count = 0
        self._premium_api_call_count = 0
        self._cache: dict[str, HtmlResult] = {}
        self._premium_domains = {"uet.edu.vn", "uet.vnu.edu.vn", "www.uet.vnu.edu.vn", "hust.edu.vn", "www.hust.edu.vn", "vnu.edu.vn", "www.vnu.edu.vn"}
        self._wp_selectors = ["article", "div.entry-content", "div.td-post-content", "div.single-content", "div.post-content"]
    async def _run_jobs(self, jobs: list[CoroutineType], k_pages: int):
        """Chạy jobs song song, dừng khi đủ k_pages thành công"""
        initial_count = self._api_call_count
        max_concurrent = min(k_pages, len(jobs))
        print(f"[Page download] Attempt to download {k_pages} pages from {len(jobs)} urls (PARALLEL, tối đa {max_concurrent} request cùng lúc)")
        page_count = 0
        results: list[None | HtmlResult] = [None] * len(jobs)
        pending: dict[asyncio.Task, int] = {}
        next_job_idx = 0

        def fill_tasks():
            nonlocal next_job_idx
            while next_job_idx < len(jobs) and len(pending) < max_concurrent and page_count < k_pages:
                task = asyncio.create_task(jobs[next_job_idx])
                pending[task] = next_job_idx
                next_job_idx += 1

        try:
            fill_tasks()
            while pending and page_count < k_pages:
                done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    idx = pending.pop(task)
                    try:
                        result = await task
                        results[idx] = result
                        if result and result.get("html", "").strip():
                            page_count += 1
                    except (asyncio.CancelledError, Exception) as e:
                        results[idx] = None
                        if not isinstance(e, asyncio.CancelledError):
                            print(f"[Page download] Task {idx} error: {str(e)[:50]}")
                    finally:
                        if page_count < k_pages:
                            fill_tasks()

                if page_count >= k_pages:
                    cancelled = [t for t in pending.keys() if not t.done()]
                    for t in cancelled:
                        t.cancel()
                    await asyncio.gather(*pending.keys(), return_exceptions=True)
                    if cancelled:
                        print(f"[Page download] Đã cancel {len(cancelled)} tasks (có thể đã gửi request, ScrapingBee vẫn tính credits)")
                    break
        finally:
            if pending:
                await asyncio.gather(*pending.keys(), return_exceptions=True)

        print(f"[Page download] Tổng requests đã gửi: {self._api_call_count - initial_count} (bao gồm cả cancelled)")
        return results
    async def download(self, search_results: list[SearchResult], k_pages: int, include_pdf: bool, include_image: bool) -> list[HtmlResult]:
        """Download up to k_pages từ danh sách URLs"""
        initial_api = self._api_call_count
        initial_premium = self._premium_api_call_count
        ssl = os.getenv("WEB_SEARCH_SSL", "True").lower() in ("true", "1")
        
        max_jobs = min(k_pages, len(search_results))
        jobs = []
        for result in search_results[:max_jobs]:
            if result["url"].endswith(".pdf"):
                jobs.append(self._handle_pdf(result) if include_pdf else self._null_task())
            else:
                jobs.append(self._download_task(ssl, result))
        
        print(f"[Page download] Chi tao {len(jobs)} jobs (k_pages={k_pages}, total_urls={len(search_results)})")
        results = await self._run_jobs(jobs, k_pages)
        html_results = [r for r in results if r]
        
        credits = self._api_call_count - initial_api
        premium = self._premium_api_call_count - initial_premium
        regular = credits - premium
        print(f"[Page download] Success: {len(html_results)}/{k_pages} pages")
        print(f"[Page download] Credits: {regular * 1 + premium * 10} (regular: {regular}, premium: {premium}x10)")
        return html_results

    async def _handle_pdf(self, search_result: SearchResult) -> HtmlResult:
        return {**search_result, "html": f'[{search_result["title"]}]({search_result["url"]})'}

    async def _null_task(self):
        return None
    def _clean_html(self, html: str) -> str:
        """Clean HTML nhưng giữ nguyên structure để preserve table format"""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside"]):
            tag.decompose()
        for comment in soup(text=lambda x: isinstance(x, Comment)):
            comment.extract()

        def has_content(elem):
            return elem.find("table") is not None or len(elem.get_text(separator=" ", strip=True)) > 200

        for selector in self._wp_selectors:
            target = soup.select_one(selector)
            if target and has_content(target):
                return str(target)

        best = None
        best_score = 0
        for candidate in soup.find_all(["div", "main", "section", "article"], recursive=True):
            score = len(candidate.get_text(separator=" ", strip=True)) + (1000 if candidate.find("table") else 0)
            if score > best_score:
                best_score = score
                best = candidate

        return str(best if best and best_score > 200 else soup.find("main") or soup.body or soup)

    def _needs_premium_proxy(self, url: str) -> bool:
        try:
            domain = urlparse(url).netloc.lower()
            return any(d in domain for d in self._premium_domains)
        except:
            return False
    
    async def _scrapingbee_fetch(self, search_result: SearchResult) -> HtmlResult | None:
        if not self._scrapingbee_key:
            print("[Page downloader] Missing SCRAPINGBEE_API_KEY")
            return None

        url = search_result["url"]
        if url in self._cache:
            print(f"[Page download] Cache hit: {url[:60]}...")
            return self._cache[url]

        headers = {"User-Agent": random.choice(USER_AGENTS)}
        use_premium_first = self._needs_premium_proxy(url)

        if not use_premium_first:
            result = await self._try_fetch(url, search_result, headers, premium=False)
            if result:
                return result

        for attempt in range(1, self._max_retry + 1):
            try:
                result = await self._try_fetch(url, search_result, headers, premium=True)
                if result:
                    return result
                if attempt < self._max_retry:
                    await asyncio.sleep(random.uniform(self._delay_min, self._delay_max))
            except asyncio.CancelledError:
                print(f"[Page downloader] Cancelled (đã gửi request): {url[:60]}...")
                raise
            except asyncio.TimeoutError:
                print(f"[Page downloader] Timeout via ScrapingBee: {url} (không retry)")
                return None
            except Exception as e:
                if attempt < self._max_retry:
                    print(f"[Page downloader] Error via ScrapingBee (attempt {attempt}/{self._max_retry}): {str(e)[:100]}")
                else:
                    print(f"[Page downloader] Error via ScrapingBee (final): {str(e)[:100]}")
        return None

    async def _try_fetch(self, url: str, search_result: SearchResult, headers: dict, premium: bool) -> HtmlResult | None:
        self._api_call_count += 1
        if premium:
            self._premium_api_call_count += 1

        params = {
            "api_key": self._scrapingbee_key,
            "url": url,
            "render_js": "false",
            "premium_proxy": "true" if premium else "false",
        }

        async with self.session.get(SCRAPINGBEE_API, params=params, headers=headers, timeout=self.timeout) as response:
            html = await response.text()
            if response.status == 200:
                cleaned = self._clean_html(html)
                if cleaned:
                    result = {**search_result, "html": cleaned}
                    self._cache[url] = result
                    print(f"[Page download] ✓ Thành công ({'premium' if premium else 'không premium'}): {url[:60]}...")
                    return result
                print(f"[Page downloader] Empty body after cleaning: {url}")
                return None
            elif response.status in [401, 402]:
                print(f"[Page download] Error {response.status} via ScrapingBee: {url[:60]}...")
                if response.status == 401:
                    print(f"[Page download] ScrapingBee error details: {html[:200]}")
                return None
            elif not premium and response.status in [402, 403, 429]:
                print(f"[Page download] Bị chặn/rate limit (status {response.status}), thử premium_proxy: {url[:60]}...")
            else:
                print(f"[Page download] Error {response.status} via ScrapingBee: {url[:60]}... (không retry)")
        return None

    async def _download_task(self, ssl: bool, search_result: SearchResult) -> HtmlResult | None:
        async with self.semaphore:
            return await self._scrapingbee_fetch(search_result)