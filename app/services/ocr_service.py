"""Azure OCR integration for extracting text from images and image-based PDFs."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover - optional dependency
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential
except Exception:  # pragma: no cover - guard when azure libraries are missing
    DocumentAnalysisClient = None  # type: ignore
    AzureKeyCredential = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class AzureOCRConfig:
    endpoint: str
    key: str


class AzureOCRService:
    """Wraps Azure Form Recognizer's `prebuilt-read` model."""

    def __init__(self, config: AzureOCRConfig) -> None:
        if DocumentAnalysisClient is None or AzureKeyCredential is None:
            raise RuntimeError(
                "azure-ai-formrecognizer is required for OCR functionality."
            )
        self._client = DocumentAnalysisClient(
            endpoint=config.endpoint,
            credential=AzureKeyCredential(config.key),
        )

    def extract_text(self, data: bytes, content_type: Optional[str] = None) -> str:
        """Call Azure Form Recognizer to extract plain text."""
        poller = self._client.begin_analyze_document(
            model_id="prebuilt-read",
            document=data,
            content_type=content_type,
        )
        result = poller.result()
        lines = []
        for page in result.pages:
            for line in page.lines:
                lines.append(line.content)
        text = "\n".join(lines).strip()
        return text
