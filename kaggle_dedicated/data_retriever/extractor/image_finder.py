"""
Utility to detect and filter "useful" images from HTML.

Hiện tại file chỉ cung cấp skeleton để bạn tự implement logic
lọc logo / banner / quảng cáo / ảnh rác, giữ lại ảnh chính.
"""

from __future__ import annotations

from typing import List, Dict

from bs4 import BeautifulSoup

from .pdf_finder import normalize_link


class ImageFinder:
    """
    Simple HTML image collector.

    - Đầu vào: raw HTML + base URL
    - Đầu ra: list[dict] với các field cơ bản:
        {
            "title": str,   # alt/title hoặc tên file
            "url": str,     # absolute URL
        }

    Bạn có thể sửa/extend class này:
    - Thêm heuristic để bỏ logo, icon, banner quảng cáo,...
    - Thêm trường metadata (width/height, position, ...).
    """

    def __init__(self, max_per_page: int = 10) -> None:
        self.max_per_page = max_per_page

        # Từ khoá nhận diện ảnh rác (logo, banner, icon,...)
        self._trash_keywords = [
            "logo",
            "icon",
            "avatar",
            "banner",
            "ads",
            "advert",
            "tracking",
            "pixel",
            "sprite",
            "placeholder",
            "thumb",
            "favicon",
            "social",
        ]

        # Phần mở rộng ưu tiên bỏ qua (thường là icon/vector)
        self._ignore_ext = {".svg", ".ico", ".gif"}

    def _is_trash_image(self, img) -> bool:
        """Heuristic đơn giản để bỏ ảnh rác."""
        # Check size attributes (nếu có)
        try:
            w = int(img.get("width") or 0)
            h = int(img.get("height") or 0)
            if (w and w < 64) or (h and h < 64):
                return True
        except Exception:
            pass

        # Check class/id/alt/title theo từ khoá
        text_meta = " ".join(
            [
                " ".join(img.get("class") or []),
                img.get("id") or "",
                img.get("alt") or "",
                img.get("title") or "",
            ]
        ).lower()
        if any(kw in text_meta for kw in self._trash_keywords):
            return True

        return False

    def find_images(self, html: str, base_url: str) -> List[Dict]:
        """
        Parse HTML, trả về danh sách ảnh (chưa lọc rác).

        TODO (tự implement):
        - Lọc theo class/id (logo, icon, banner, avatar, ads,...)
        - Lọc theo kích thước (too small -> bỏ)
        - Lọc theo domain (CDN ads vs content chính)
        """
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        seen: set[str] = set()
        results: List[Dict] = []

        for img in soup.find_all("img"):
            src = (img.get("src") or "").strip()
            if not src:
                continue

            url = normalize_link(src, base_url)
            if not url or url in seen:
                continue

            # Bỏ qua theo extension
            lower_url = url.lower()
            for ext in self._ignore_ext:
                if lower_url.endswith(ext):
                    seen.add(url)
                    break
            else:
                # Lọc theo heuristic rác
                if self._is_trash_image(img):
                    seen.add(url)
                    continue

                seen.add(url)

                # Text mô tả ảnh: ưu tiên alt -> title -> filename
                alt = (img.get("alt") or "").strip()
                title = (img.get("title") or "").strip()
                name = url.split("/")[-1] or "image"
                display_title = alt or title or name

                results.append(
                    {
                        "title": display_title,
                        "url": url,
                    }
                )

                if len(results) >= self.max_per_page:
                    break

        return results



