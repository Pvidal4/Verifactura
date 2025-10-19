from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from openai import OpenAI
from transformers import pipeline

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


def _parse_model_response(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive programming
        raise RuntimeError(
            "El modelo no devolvió un JSON válido conforme al esquema solicitado."
        ) from exc
    return data

class OpenAILLMService:
    def __init__(self, config: Config) -> None:

        if not config.openai_configured:
            raise RuntimeError("No existe: OPENAI_API_KEY")
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        self._model = config.OPENAI_MODEL
        self._schema_name = config.JSON_MODE_SCHEMA_NAME
        self._default_temperature = 1.0
        self._default_top_p = 1.0

    def extract(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        chosen_model = (model or self._model).strip()
        response = self._client.chat.completions.create(
            model=chosen_model,
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
            temperature=self._default_temperature if temperature is None else temperature,
            top_p=self._default_top_p if top_p is None else top_p,
            reasoning_effort="minimal",
        )
        content = response.choices[0].message.content
        return _parse_model_response(content)


class LocalLLMService:
    def __init__(self, config: Config) -> None:
        configured_path = config.LOCAL_LLM_MODEL_PATH
        configured_id = config.LOCAL_LLM_MODEL_ID
        candidate = configured_path or configured_id or "models/gpt-oss-20b"
        if configured_path:
            resolved_path = Path(configured_path)
            if resolved_path.exists():
                candidate = str(resolved_path)
            elif configured_id:
                candidate = configured_id

        self._default_model = candidate
        self._device = 0 if torch.cuda.is_available() else -1
        print(
            f"Using device: {'GPU (CUDA)' if self._device == 0 else 'CPU'}"
        )
        self._pipelines: Dict[str, Any] = {}
        self._default_temperature = 1.0
        self._default_top_p = 1.0

    def _get_pipeline(self, model: Optional[str] = None):
        source = (model or self._default_model).strip()
        if source not in self._pipelines:
            resolved = Path(source)
            model_source = str(resolved) if resolved.exists() else source
            self._pipelines[source] = pipeline(
                "text-generation",
                model=model_source,
                dtype=torch.bfloat16,
                device_map=None,
                device=self._device,
                trust_remote_code=True,
            )
        return self._pipelines[source]

    def extract(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = [
            {
                "role": "user",
                "content": f"{SYSTEM_PROMPT}\n\n{text}",
            },
        ]

        pipeline_instance = self._get_pipeline(model)

        outputs = pipeline_instance(
            messages,
            max_new_tokens=256,
            temperature=self._default_temperature if temperature is None else temperature,
            top_p=self._default_top_p if top_p is None else top_p,
        )

        final_message = outputs[0]["generated_text"][-1]
        if isinstance(final_message, dict):
            content = str(final_message.get("content", ""))
        else:
            content = str(final_message)

        if not content.strip():
            raise RuntimeError("El modelo local devolvió una respuesta vacía.")

        return _parse_model_response(content)
