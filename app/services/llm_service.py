"""Interface with OpenAI for structured invoice extraction."""
from __future__ import annotations

import json
from typing import Any, Dict

try:  # pragma: no cover - dependency guard
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - allow runtime error later
    OpenAI = None  # type: ignore
    _OPENAI_IMPORT_ERROR = exc
else:
    _OPENAI_IMPORT_ERROR = None

from app.config import Config

INVOICE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "FECHA_DOCUMENTO": {"type": ["string", "null"]},
        "DIRECCION": {"type": ["string", "null"]},
        "MODELO_HOMOLOGADO_ANT": {"type": ["string", "null"]},
        "SUBSIDIO": {"type": ["number", "null"]},
        "AÑO": {"type": ["number", "null"]},
        "SUBTOTAL": {"type": ["number", "null"]},
        "CLASE": {"type": ["string", "null"]},
        "TOTAL": {"type": ["number", "null"]},
        "CILINDRAJE": {"type": ["string", "null"]},
        "MODELO": {"type": ["string", "null"]},
        "MODELO_REGISTRADO_SRI": {"type": ["string", "null"]},
        "RAMV_CPN": {"type": ["string", "null"]},
        "RUEDAS": {"type": ["number", "null"]},
        "DESCUENTO": {"type": ["number", "null"]},
        "NUMERO_FACTURA": {"type": ["string", "null"]},
        "COLOR": {"type": ["string", "null"]},
        "MOTOR": {"type": ["string", "null"]},
        "NOMBRE_CLIENTE": {"type": ["string", "null"]},
        "CAPACIDAD": {"type": ["number", "null"]},
        "MARCA": {"type": ["string", "null"]},
        "RUC": {"type": ["string", "null"]},
        "COMBUSTIBLE": {"type": ["string", "null"]},
        "EJES": {"type": ["number", "null"]},
        "TIPO": {"type": ["string", "null"]},
        "IVA": {"type": ["number", "null"]},
        "CONCESIONARIA": {"type": ["string", "null"]},
        "TONELAJE": {"type": ["number", "null"]},
        "VIN_CHASIS": {"type": ["string", "null"]},
        "PAIS_ORIGEN": {"type": ["string", "null"]},
        "ETIQUETA": {"type": ["string", "null"]},
    },
    "required": list(),
}

SYSTEM_PROMPT = (
    "Eres un asistente que extrae datos estructurados de documentos vehiculares. "
    "Responde únicamente con JSON válido que coincida con el esquema dado. "
    "Utiliza null cuando la información no esté presente."
)


class OpenAILLMService:
    def __init__(self, config: Config) -> None:
        if _OPENAI_IMPORT_ERROR is not None or OpenAI is None:
            raise RuntimeError(
                "The 'openai' package is required. Install it with 'pip install openai'."
            ) from _OPENAI_IMPORT_ERROR
        if not config.openai_configured:
            raise RuntimeError("OPENAI_API_KEY must be configured to use the LLM service.")
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        self._model = config.OPENAI_MODEL
        self._schema_name = config.JSON_MODE_SCHEMA_NAME

    def extract(self, text: str) -> Dict[str, Any]:
        response = self._client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": self._schema_name,
                    "schema": INVOICE_SCHEMA,
                    "strict": True,
                },
            },
            temperature=0,
        )
        content = response.output[0].content[0].text
        data = json.loads(content)
        return data
