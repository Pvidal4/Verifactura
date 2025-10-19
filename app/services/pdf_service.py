from __future__ import annotations

import io
from typing import List, Tuple

from PyPDF2 import PdfReader

class PDFTextExtractor:

    def __init__(self, max_chars_per_chunk: int = 50_000) -> None:
        self.max_chars_per_chunk = max_chars_per_chunk

    def read_text(self, file_bytes: bytes) -> str:

        reader = PdfReader(io.BytesIO(file_bytes))
        parts: List[str] = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            parts.append(text)
        joined = "\n".join(parts).strip()
        return " ".join(joined.split())

    @staticmethod
    def _guess_image_content_type(data: bytes, image_format: str) -> str:
        fmt = (image_format or "").lower()
        if fmt in {"jpeg", "jpg", "jpe"}:
            return "image/jpeg"
        if fmt in {"png"}:
            return "image/png"
        if fmt in {"tif", "tiff"}:
            return "image/tiff"
        if fmt in {"bmp"}:
            return "image/bmp"
        if fmt in {"gif"}:
            return "image/gif"
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if data[:4] in (b"II*\x00", b"MM\x00*"):
            return "image/tiff"
        return "image/png"

    def render_page_images(self, file_bytes: bytes) -> List[Tuple[bytes, str]]:
        reader = PdfReader(io.BytesIO(file_bytes))
        rendered: List[Tuple[bytes, str]] = []
        for page in reader.pages:
            try:
                images = list(getattr(page, "images", []))
            except Exception:
                images = []
            if not images:
                continue
            chosen: Tuple[bytes, str] | None = None
            max_area = -1
            for image in images:
                data = getattr(image, "data", b"")
                if not data:
                    continue
                width = getattr(image, "width", 0) or 0
                height = getattr(image, "height", 0) or 0
                area = int(width) * int(height)
                if area <= max_area:
                    continue
                content_type = self._guess_image_content_type(
                    data, getattr(image, "image_format", "")
                )
                chosen = (data, content_type)
                max_area = area
            if chosen is not None:
                rendered.append(chosen)
        return rendered

    def chunk_text(self, text: str) -> list[str]:
        if len(text) <= self.max_chars_per_chunk:
            return [text]
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chars_per_chunk, len(text))
            split_at = text.rfind("\n\n", start, end)
            if split_at == -1:
                split_at = text.rfind(". ", start, end)
            if split_at == -1:
                split_at = end
            chunk = text[start:split_at].strip()
            if chunk:
                chunks.append(chunk)
            start = split_at
        return chunks
