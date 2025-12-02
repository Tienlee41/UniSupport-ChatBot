import asyncio, aiohttp, unicodedata
from urllib.parse import urljoin
from bs4 import BeautifulSoup

from ..schema import FileSource
from .pdf_to_text import PDFProcessor

KEYWORDS = ["dinh kem", "file dinh kem", "tai ve", "download", "phu luc", "quy dinh", "huong dan"]

def strip_accents(s: str) -> str:
    if not s: return ""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()

def normalize_link(href: str, base: str) -> str:
    if not href: return ""
    href = href.strip()
    if not href or href.startswith(("javascript", "mailto:", "#")): return ""
    if href.startswith("//"): return "https:" + href
    if href.startswith("http"): return href
    if href.startswith("/"): return base.rstrip("/") + href
    return urljoin(base, href)

class PdfFinder:
    def __init__(self, session: aiohttp.ClientSession, timeout: aiohttp.ClientTimeout, 
                 concurrent: int, max_per_page: int) -> None:
        self.session, self.timeout = session, timeout
        self._sem = asyncio.Semaphore(concurrent)
        self._max = max_per_page
        self.pdf_to_text = PDFProcessor()

    def _find_pdf_links(self, html: str, base_url: str) -> list[dict]:
        """Find all PDF links: direct .pdf links + keyword links that end with .pdf"""
        soup, seen, results = BeautifulSoup(html, "html.parser"), set(), []
        for a in soup.find_all("a", href=True):
            href = a.get("href", "").strip()
            url = normalize_link(href, base_url)
            if not url or url in seen: continue
            seen.add(url)
            title = a.get_text(" ", strip=True) or url.split("/")[-1]
            # Direct PDF link
            if url.lower().endswith(".pdf"):
                results.append({"title": title, "url": url})
            # Keyword link that is PDF
            else:
                meta = strip_accents(" ".join([title, a.get("title", ""), href]))
                if any(kw in meta for kw in KEYWORDS) and url.lower().endswith(".pdf"):
                    results.append({"title": title, "url": url})
        return results

    async def _download(self, ssl: bool, title: str, url: str) -> FileSource | None:
        async with self._sem:
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                           "Accept": "application/pdf,*/*"}
                async with self.session.get(url=url, timeout=self.timeout, ssl=ssl, headers=headers) as r:
                    if r.ok:
                        data = await r.content.read()
                        if data: return {"file_title": title, "file_url": url, "file_type": "pdf", 
                                        "text": self.pdf_to_text.extract_text(data)}
            except: pass
            return None

    async def find_pdfs(self, html: str, url: str, ssl: bool) -> tuple[list[FileSource], list[str]]:
        """Find and download PDFs. Returns (files, urls)"""
        pdf_links = self._find_pdf_links(html, url)[:self._max]
        if not pdf_links: return [], []
        results = await asyncio.gather(*[self._download(ssl, p["title"], p["url"]) for p in pdf_links])
        files = [r for r in results if r]
        urls = [p["url"] for p in pdf_links]
        return files, urls
