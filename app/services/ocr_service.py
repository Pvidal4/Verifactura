"""Abstracciones para interactuar con el servicio Azure Form Recognizer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

LOGGER = logging.getLogger(__name__)


@dataclass
class AzureOCRConfig:
    """Representa la configuración mínima necesaria para conectarse a Azure OCR."""

    endpoint: str
    key: str


class AzureOCRService:
    """Pequeño envoltorio del cliente oficial de Azure Form Recognizer."""

    def __init__(self, config: AzureOCRConfig) -> None:

        self._client = DocumentAnalysisClient(
            endpoint=config.endpoint,
            credential=AzureKeyCredential(config.key),
        )

    def extract_text(self, data: bytes, content_type: Optional[str] = None) -> str:
        """Ejecuta el modelo `prebuilt-read` y concatena las líneas detectadas."""

        if content_type:
            try:
                poller = self._client.begin_analyze_document(
                    model_id="prebuilt-read",
                    document=data,
                    content_type=content_type,
                )
            except TypeError as exc:
                if "content_type" not in str(exc):
                    raise
                LOGGER.warning(
                    "Azure Form Recognizer rechazó content_type '%s'; "
                    "reintentando sin especificarlo.",
                    content_type,
                )
                poller = self._client.begin_analyze_document(
                    model_id="prebuilt-read",
                    document=data,
                )
        else:
            poller = self._client.begin_analyze_document(
                model_id="prebuilt-read",
                document=data,
            )
        result = poller.result()
        lines = []
        for page in result.pages:
            for line in page.lines:
                lines.append(line.content)
        text = "\n".join(lines).strip()
        return text
