import re
import os
import asyncio
import time
import aiohttp
from urllib.parse import urljoin
from bs4 import BeautifulSoup, NavigableString, Tag, Comment
from crawl4ai import DefaultMarkdownGenerator
from ..schema import HtmlResult, WebSource, FileSource
from .pdf_finder import PdfFinder
from .image_ocr import ImageOCR
from .image_finder import ImageFinder

WS = re.compile(r'\s+', re.DOTALL)
URL_RE = re.compile(r'\[([^\]]+)\]\(.*?\)')
PIPE_LINE = re.compile(r'^[\|\s]+$')

class ContentExtractor:
    def __init__(self, session: aiohttp.ClientSession, concurrent: int, timeout: float, max_file: int, min_len: int = 5):
        self._min_len = min_len
        self.session = session
        self.timeout = aiohttp.ClientTimeout(timeout)
        self.pdf_finder = PdfFinder(session, self.timeout, concurrent, max_file)
        self.image_ocr = ImageOCR()
        self.image_finder = ImageFinder(max_per_page=max_file)
        self._image_semaphore = asyncio.Semaphore(min(concurrent * 2, 10))
        self.md_gen = DefaultMarkdownGenerator(options={"ignore_links": False, "escape_html": True, "skip_internal_links": True, "ignore_images": True})

    def _clean(self, t): 
        return re.sub(r'\s+', ' ', str(t).strip()) if t else ""
    
    def _is_footer(self, line):
        s = line.strip()
        if not s:
            return False
        return PIPE_LINE.match(s) or (s.count('|') >= 2 and len(s.replace('|', '').replace(' ', '')) < 10)

    def _table_md(self, table: Tag) -> str:
        trs = table.find_all("tr")
        if not trs:
            return ""
        max_c = max((sum(int(c.get("colspan", 1)) for c in tr.find_all(["td", "th"])) for tr in trs), default=0)
        if not max_c:
            return ""
        
        grid, tracker = [], {}
        for tr in trs:
            row, idx = [""] * max_c, 0
            for c in range(max_c):
                if c in tracker:
                    v, r = tracker[c]
                    row[c] = v
                    tracker[c] = (v, r-1) if r > 1 else tracker.pop(c, None) or (v, 0)
                    if c in tracker and tracker[c][1] <= 0:
                        del tracker[c]
            for cell in tr.find_all(["td", "th"]):
                while idx < max_c and row[idx]:
                    idx += 1
                if idx >= max_c:
                    break
                txt = re.sub(r'\*\*', '', self._clean(cell.get_text(" ", strip=True))).replace("|", "\\|")
                cs, rs = int(cell.get("colspan", 1)), int(cell.get("rowspan", 1))
                for c in range(idx, min(idx + cs, max_c)):
                    row[c] = txt
                    if rs > 1:
                        tracker[c] = (txt, rs - 1)
                idx += cs
            grid.append(row)
        
        if not grid:
            return ""
        cols = [i for i in range(max_c) if any(r[i].strip() for r in grid)]
        if not cols:
            return ""
        grid = [[r[i] for i in cols] for r in grid]
        h = grid[0]
        sep = " | ".join(["---"] * len(h))
        return "\n".join([f"| {' | '.join(h)} |", f"| {sep} |"] + 
                        [f"| {' | '.join((r + ['']*len(h))[:len(h)])} |" for r in grid[1:]])

    def _walk(self, node, out):
        if isinstance(node, NavigableString):
            if t := str(node).strip():
                out.append(t)
            return
        if not isinstance(node, Tag):
            return
        n = node.name.lower()
        if n in ["script", "style", "meta", "header", "footer", "nav", "noscript"]:
            return
        if n in ["h1", "h2", "h3", "h4"]:
            if t := self._clean(node.get_text(" ", strip=True)):
                out.append("#" * int(n[1]) + " " + t)
            return
        if n == "p":
            if t := self._clean(node.get_text(" ", strip=True)):
                out.append(t)
            return
        if n in ["ul", "ol"]:
            for i, li in enumerate(node.find_all("li", recursive=False), 1):
                if t := self._clean(li.get_text(" ", strip=True)):
                    out.append(f"{i}. {t}" if n == "ol" else f"- {t}")
            return
        if n == "table":
            if md := self._table_md(node):
                out.extend(["[BANG]"] + md.splitlines())
            return
        for c in node.children:
            self._walk(c, out)

    def _html_to_text(self, html):
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "meta", "header", "footer", "nav", "noscript", "svg"]):
            t.decompose()
        for e in soup(text=lambda x: isinstance(x, Comment)):
            e.extract()
        for sel in ["[class*=breadcrumb]", ".site-footer", "footer", ".sidebar", "#sidebar", ".related-posts", ".share-buttons"]:
            for n in soup.select(sel):
                n.decompose()
        out = []
        self._walk(soup, out)
        return "\n".join(c for c in out if c)

    def _clean_footer(self, text):
        lines, result, i = text.splitlines(), [], 0
        while i < len(lines):
            line = lines[i].strip()
            if PIPE_LINE.match(line) or line == '|':
                parts = []
                while i < len(lines):
                    cur = lines[i].strip()
                    if PIPE_LINE.match(cur) or cur in ['|', '']:
                        i += 1
                    elif cur and not self._is_footer(cur):
                        parts.append(cur)
                        i += 1
                    else:
                        i += 1
                        break
                if parts:
                    result.append(' | '.join(parts))
            else:
                result.append(lines[i])
                i += 1
        while result and not result[-1].strip():
            result.pop()
        return '\n'.join(result)

    def _process(self, text, min_len):
        lines, result, in_tbl = text.splitlines(), [], False
        markers = ('#', '- ', '* ', '+ ', '|', '[BANG]')
        for line in lines:
            s = line.strip()
            if self._is_footer(s):
                continue
            solid = re.sub(WS, '', s)
            if s == '[BANG]':
                if not (result and result[-1].strip() == '[BANG]'):
                    in_tbl = True
                    if result and result[-1].strip():
                        result.append('')
                    result.append('[BANG]')
            elif in_tbl:
                if s.startswith('|') and '---' not in s:
                    cells = [re.sub(r'\*\*', '', re.sub(r'\s+', ' ', c.strip())) for c in s.split('|')]
                    while cells and not cells[0].strip():
                        cells.pop(0)
                    while cells and not cells[-1].strip():
                        cells.pop()
                    if cells:
                        result.append('|' + '|'.join(cells) + '|')
                        if len(result) >= 2 and result[-2] == '[BANG]':
                            result.append('|' + '|'.join(['---'] * len([c for c in cells if c.strip()])) + '|')
                elif s:
                    in_tbl = False
                    if (solid and len(solid) > min_len) or s.startswith(markers):
                        result.append(re.sub(r'\s+', ' ', s))
            elif s.startswith(markers) or (solid and len(solid) > min_len):
                result.append(s)
        return re.sub(r'\n{3,}', '\n\n', self._clean_footer('\n'.join(result))).strip()

    def _extract(self, html, url):
        if not ("<" in html and ">" in html and re.search(r'<\s*/?\s*[a-zA-Z][^>]*>', html)):
            return self._process(html, self._min_len)
        text = self._html_to_text(html)
        if not text.strip():
            soup = BeautifulSoup(html, "html.parser")
            ph = {}
            for i, t in enumerate(soup.find_all("table")):
                k = f"[TABLE_{i}]"
                ph[k] = self._table_md(t)
                t.replace_with(BeautifulSoup(f'<div>{k}</div>', "html.parser").div)
            text = URL_RE.sub(r'\1', self.md_gen.generate_markdown(input_html=str(soup), base_url=url).raw_markdown)
            for k, v in ph.items():
                text = text.replace(k, f"\n[BANG]\n{v}\n")
        return re.sub(r'\n{3,}', '\n\n', self._process(text, self._min_len))

    async def extract(self, results: list[HtmlResult], include_pdf: bool, include_image: bool) -> list[WebSource]:
        ssl = os.getenv("WEB_SEARCH_SSL", "True").lower() in ("true", "1")
        jobs = [self._job(ssl, r, include_pdf, include_image) for r in results]
        return [r for r in await asyncio.gather(*jobs) if r]

    async def _job(self, ssl: bool, hr: HtmlResult, pdf: bool, include_image: bool) -> WebSource | None:
        content = self._extract(hr["html"], hr["url"])
        files: list[FileSource] = (await self.pdf_finder.find_pdfs(hr["html"], hr["url"], ssl))[0] if pdf else []
        image_parts: list[str] = []

        if include_image and self.image_ocr.enabled:
            try:
                ocr_start = time.time()
                print(f"[ImageOCR] Running for URL: {hr['url']}")
                images = self.image_finder.find_images(hr["html"], hr["url"])
                print(f"[ImageOCR] Found {len(images)} candidate images")
                
                max_images_ocr = min(len(images), 5)
                img_urls = [(img["title"], img["url"]) for img in images[:max_images_ocr]]
                if len(images) > max_images_ocr:
                    print(f"[ImageOCR] Limiting to {max_images_ocr} images (from {len(images)} total)")

                async def _fetch_and_ocr(name: str, url: str) -> tuple[str, str] | None:
                    img_start = time.time()
                    async with self._image_semaphore:
                        try:
                            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                            async with self.session.get(url=url, timeout=self.timeout, ssl=ssl, headers=headers) as r:
                                if not r.ok:
                                    return None
                                data = await r.read()
                                if not data:
                                    return None
                                text = await self.image_ocr.aocr_image_bytes(name, data, timeout=10.0)
                                if not text.strip():
                                    return None
                                print(f"[ImageOCR] OCR ok for image: {name} ({time.time() - img_start:.2f}s)")
                                return name, text
                        except Exception as e:
                            print(f"[ImageOCR] Error processing {name}: {str(e)[:100]}")
                            return None

                ocr_results = await asyncio.gather(*[_fetch_and_ocr(n, u) for n, u in img_urls], return_exceptions=True)
                print(f"[ImageOCR] Total OCR time: {time.time() - ocr_start:.2f}s for {len(img_urls)} images")
                
                for res in ocr_results:
                    if not res:
                        continue
                    name, text = res
                    image_parts.append(f"## [IMAGE: {name}]\n\n{text.strip()}")
                    files.append({
                        "file_title": name,
                        "file_url": urljoin(hr["url"], name),
                        "file_type": "image",
                        "text": text,
                    })
            except Exception:
                pass

        parts = [content.strip()] if content.strip() else []
        for f in files:
            if f["text"].strip():
                parts.append(f"## [PDF: {f['file_url'].split('/')[-1] or 'doc.pdf'}]\n\n{f['text'].strip()}")
        parts.extend(image_parts)
        final = re.sub(r'\n{3,}', '\n\n', "\n\n".join(parts)).strip()
        if not (final or files):
            return None
        return {
            "query": hr["query"],
            "title": hr["title"],
            "url": hr["url"],
            "description": hr["description"],
            "text": final,
            "files": files,
            "score": hr["score"]
        }
