from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

@dataclass(frozen=True)
class Config:

    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

    LOCAL_LLM_MODEL_ID: str = os.getenv("LOCAL_LLM_MODEL_ID", "openai/gpt-oss-20b")
    LOCAL_LLM_MODEL_PATH: str = os.getenv("LOCAL_LLM_MODEL_PATH", "models/gpt-oss-20b")

    AZURE_ENDPOINT: str | None = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
    AZURE_KEY: str | None = os.getenv("AZURE_FORM_RECOGNIZER_KEY")

    MAX_CHARS_PER_CHUNK: int = int(os.getenv("MAX_CHARS_PER_CHUNK", "50000"))
    JSON_MODE_SCHEMA_NAME: str = os.getenv("JSON_MODE_SCHEMA_NAME", "factura_vehicular")

    @property
    def azure_configured(self) -> bool:
        return bool(self.AZURE_ENDPOINT and self.AZURE_KEY)

    @property
    def openai_configured(self) -> bool:
        return bool(self.OPENAI_API_KEY)
