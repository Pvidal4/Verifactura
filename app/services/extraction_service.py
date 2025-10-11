"""High-level orchestration for document ingestion and extraction."""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Dict, Optional

from app.config import Config
from app.services.llm_service import OpenAILLMService
from app.services.ocr_service import AzureOCRConfig, AzureOCRService
from app.services.pdf_service import PDFTextExtractor

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
TEXT_EXTENSIONS = {".txt", ".json", ".csv"}
XML_EXTENSIONS = {".xml"}
PDF_EXTENSIONS = {".pdf"}


class ExtractionService:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._pdf = PDFTextExtractor(config.MAX_CHARS_PER_CHUNK)
        self._llm = OpenAILLMService(config)
        self._ocr = None
        if config.azure_configured:
            self._ocr = AzureOCRService(
                AzureOCRConfig(endpoint=config.AZURE_ENDPOINT, key=config.AZURE_KEY)
            )

    def _needs_ocr(self, extension: str, text: str) -> bool:
        if extension in IMAGE_EXTENSIONS:
            return True
        if extension in PDF_EXTENSIONS and not text:
            return True
        return False

    def _extract_text_from_file(
        self,
        filename: str,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix in PDF_EXTENSIONS:
            text = self._pdf.read_text(data)
            if text:
                return text
        elif suffix in TEXT_EXTENSIONS:
            return data.decode("utf-8", errors="replace")
        elif suffix in XML_EXTENSIONS:
            return data.decode("utf-8", errors="replace")
        if self._ocr is None:
            raise RuntimeError(
                "Azure OCR is not configured but is required for this file type."
            )
        if content_type is None:
            content_type = mimetypes.guess_type(filename)[0]
        return self._ocr.extract_text(data, content_type=content_type)

    def extract_from_text(self, text: str) -> Dict[str, object]:
        return self._llm.extract(text)

    def extract_from_file(
        self,
        filename: str,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> Dict[str, object]:
        text = ""
        suffix = Path(filename).suffix.lower()
        if suffix in PDF_EXTENSIONS:
            text = self._pdf.read_text(data)
        if self._needs_ocr(suffix, text):
            text = self._extract_text_from_file(filename, data, content_type)
        elif not text:
            text = self._extract_text_from_file(filename, data, content_type)
        return self.extract_from_text(text)
