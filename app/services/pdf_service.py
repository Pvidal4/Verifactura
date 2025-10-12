from __future__ import annotations

import io
from typing import List

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
