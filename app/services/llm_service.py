from __future__ import annotations

import json
from typing import Any, Dict, Optional

from openai import OpenAI

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

INVOICE_SCHEMA["required"] = list(INVOICE_SCHEMA["properties"].keys())


SYSTEM_PROMPT = (
    "Eres un asistente que extrae datos estructurados de documentos vehiculares. "
    "Responde únicamente con JSON válido que coincida con el esquema dado. "
    "Utiliza null cuando la información no esté presente."
)

class OpenAILLMService:
    def __init__(self, config: Config) -> None:

        if not config.openai_configured:
            raise RuntimeError("No existe: OPENAI_API_KEY")
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        self._model = config.OPENAI_MODEL
        self._schema_name = config.JSON_MODE_SCHEMA_NAME

    def extract(self, text: str) -> Dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
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
            temperature=1,
            reasoning_effort="minimal"
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return data


class LocalLLMService:
    """Servicio de inferencia para modelos alojados localmente."""

    def __init__(
        self,
        model_id: str,
        *,
        task: str = "text-generation",
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._model_id = model_id
        self._task = task
        self._pipeline_kwargs = dict(pipeline_kwargs or {})
        self._pipeline: Optional[Any] = None

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is None:
            from transformers import pipeline as hf_pipeline

            kwargs = dict(self._pipeline_kwargs)
            model_kwargs = dict(kwargs.pop('model_kwargs', {}))
            model_kwargs.setdefault('trust_remote_code', True)
            if model_kwargs:
                kwargs['model_kwargs'] = model_kwargs

            trust_remote_code = kwargs.pop('trust_remote_code', True)

            self._pipeline = hf_pipeline(
                self._task,
                model=self._model_id,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

        return self._pipeline
