import asyncio
import os
import re
from pathlib import Path
from typing import Optional

try:
    from landingai_ade import LandingAIADE
except Exception:  # landingai_ade might not be installed in all environments
    LandingAIADE = None  # type: ignore


def _clean_markdown(md_text: str) -> str:
    """Clean markdown output from ADE, same logic as eval/ade.ipynb."""
    md_text = re.sub(r"<::.*?::>", "", md_text, flags=re.DOTALL)
    md_text = re.sub(r"<a id=['\"].*?['\"]></a>", "", md_text)

    def html_table_to_text(match: re.Match) -> str:
        table_html = match.group(0)
        rows = re.findall(r"<tr>(.*?)</tr>", table_html, flags=re.DOTALL)
        table_text = []
        for r in rows:
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.DOTALL)
            row_text = " | ".join(cell.strip() for cell in cells)
            table_text.append(row_text)
        return "\n".join(table_text)

    md_text = re.sub(
        r"<table.*?>.*?</table>", html_table_to_text, md_text, flags=re.DOTALL
    )

    html_unescape = {
        "&lt;": "<",
        "&gt;": ">",
        "&amp;": "&",
        "&quot;": '"',
        "&#39;": "'",
    }
    for k, v in html_unescape.items():
        md_text = md_text.replace(k, v)

    md_text = re.sub(r"\n{3,}", "\n\n", md_text).strip()
    return md_text


class ImageOCR:
    """Minimal ADE-based OCR wrapper for images (used when include_image=True)."""

    def __init__(self) -> None:
        api_key_env = None
        if os.getenv("VISION_AGENT_API_KEY"):
            api_key_env = "VISION_AGENT_API_KEY"
        elif os.getenv("LANDINGAI_API_KEY"):
            api_key_env = "LANDINGAI_API_KEY"

        api_key = os.getenv(api_key_env) if api_key_env else None
        self._enabled = bool(api_key and LandingAIADE is not None)
        self._client: Optional[LandingAIADE] = None
        if self._enabled:
            try:
                self._client = LandingAIADE(apikey=api_key)  # type: ignore[arg-type]
                print(f"[ImageOCR] Enabled with {api_key_env}")
            except Exception:
                # If client init fails, silently disable OCR to avoid breaking pipeline
                self._client = None
                self._enabled = False
                print("[ImageOCR] Failed to init LandingAIADE, OCR disabled")
        else:
            if LandingAIADE is None:
                print("[ImageOCR] Disabled: landingai_ade not installed")
            elif not api_key_env:
                print("[ImageOCR] Disabled: missing VISION_AGENT_API_KEY/LANDINGAI_API_KEY")
            else:
                print(f"[ImageOCR] Disabled: {api_key_env} is empty")

    @property
    def enabled(self) -> bool:
        return self._enabled and self._client is not None

    def ocr_image_bytes(self, name: str, data: bytes) -> str:
        """Run OCR on raw image bytes. Returns cleaned markdown text or empty string."""
        if not self.enabled or not data:
            return ""
        assert self._client is not None  # for type checkers

        # LandingAIADE expects a file-like object or path; use a temporary file
        import tempfile

        suffix = Path(name).suffix or ".png"
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                tmp.write(data)
                tmp.flush()
                resp = self._client.parse(document=Path(tmp.name))  # type: ignore[arg-type]
                md_text = getattr(resp, "markdown", "") or ""
                return _clean_markdown(md_text)
        except Exception:
            return ""

    async def aocr_image_bytes(self, name: str, data: bytes, timeout: float = 10.0) -> str:
        """Async wrapper using a thread to avoid blocking event loop. Has timeout to prevent hanging."""
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, self.ocr_image_bytes, name, data),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            print(f"[ImageOCR] Timeout ({timeout}s) for image: {name}")
            return ""
        except Exception as e:
            print(f"[ImageOCR] Error OCR image {name}: {str(e)[:100]}")
            return ""


