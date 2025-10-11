"""Application configuration for the Verifactura extraction API."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Runtime configuration sourced from environment variables."""

    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

    AZURE_ENDPOINT: str | None = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
    AZURE_KEY: str | None = os.getenv("AZURE_FORM_RECOGNIZER_KEY")

    MAX_CHARS_PER_CHUNK: int = int(os.getenv("MAX_CHARS_PER_CHUNK", "50000"))
    JSON_MODE_SCHEMA_NAME: str = os.getenv("JSON_MODE_SCHEMA_NAME", "vehicular_invoice")

    @property
    def azure_configured(self) -> bool:
        return bool(self.AZURE_ENDPOINT and self.AZURE_KEY)

    @property
    def openai_configured(self) -> bool:
        return bool(self.OPENAI_API_KEY)
