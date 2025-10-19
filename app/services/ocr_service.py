from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

LOGGER = logging.getLogger(__name__)

@dataclass
class AzureOCRConfig:
    endpoint: str
    key: str

class AzureOCRService:
    def __init__(self, config: AzureOCRConfig) -> None:

        self._client = DocumentAnalysisClient(
            endpoint=config.endpoint,
            credential=AzureKeyCredential(config.key),
        )
    def extract_text(self, data: bytes, content_type: Optional[str] = None) -> str:

        options = {}
        if content_type:
            options["content_type"] = content_type
        poller = self._client.begin_analyze_document(
            model_id="prebuilt-read",
            document=data,
            **options,
        )
        result = poller.result()
        lines = []
        for page in result.pages:
            for line in page.lines:
                lines.append(line.content)
        text = "\n".join(lines).strip()
        return text
