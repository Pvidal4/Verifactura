from __future__ import annotations

"""Servicios concretos para interactuar con modelos de lenguaje (API y local)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from openai import OpenAI
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

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
        "PAIS_ORIGEN": {"type": ["string", "null"]}
    },
    "required": list(),
}

INVOICE_SCHEMA["required"] = list(INVOICE_SCHEMA["properties"].keys())


SYSTEM_PROMPT = (
    "Eres un asistente que extrae datos estructurados de documentos vehiculares. "
    "Cuando tengas disponible una imagen del documento (png) y el contenido OCR de la imagen para facilitar la lectura, puede ocurrir que algunos detalles importantes se pierdan debido a saltos de línea inesperados."
    "\n\nINICIO DE EJEMPLO\n\nTexto original:\n\nMODELO: V3 VAN AC 1.5 4P 4X2 TM\n\nTexto OCR:\n\nMODELO: V3 VAN\nCOLOR: PLATEADO\nAC 1.5 4P 4X2 TM\n\nFIN DE EJEMPLO\n\n"
    "En el texto OCR, la información de 'AC 1.5 4P 4X2 TM' se ha separado de la línea 'MODELO: V3 VAN' debido a un salto de línea no intencionado."
    "La respuesta correcta seria: 'MODELO: V3 VAN AC 1.5 4P 4X2 TM'"
    "Es importante tener en cuenta esta posibilidad y realizar un análisis adicional para recombinar las partes separadas y garantizar que se capture toda la información correctamente."
    "Se te ha proporcionado una estructura que indica los campos que debes extraer del contenido OCR de la imagen. Asegúrate de leer bien todo el contenido antes de identificar el campo, es posible que el campo completo se encuentre luego en el documento."
    "Responde únicamente con JSON válido que coincida con el esquema dado. "
    "Algunos de los campos son:"
    "MARCA: El fabricante del vehículo."
    "MODELO: Siglas designadas por el fabricante del vehículo."
    "COLOR: El color del vehículo."
    "MOTOR: Código individual (letras y números) que identifica el motor del vehículo, la capacidad y el tipo de combustible. (sin espacio)"
    "VIN_CHASIS: También se puede encontrar como “Serie” o “VIN” (de 17 caracteres). Identifica el chasis o carrocería único de cada vehículo. (sin espacio)"
    "AÑO: Año de fabricación del vehículo."
    "CLASE: La clase del vehículo."
    "TIPO: El tipo del vehículo."
    "RAMV_CPN: Se puede encontrar como RAMV o CPN. Ejemplo: T00123456. (sin espacio)"
    "Los que dicen 'sin espacio' significa que debes quitarle espacios si estan presentes. Ej. T00 123 456 -> T00123456"
    "Utiliza null cuando la información no esté presente."
)


def _parse_model_response(raw: str) -> Dict[str, Any]:
    """Convierte la respuesta textual del modelo en un diccionario Python."""

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive programming
        raise RuntimeError(
            "El modelo no devolvió un JSON válido conforme al esquema solicitado."
        ) from exc
    return data

class OpenAILLMService:
    """Cliente especializado para llamar a la API de OpenAI con esquema JSON."""

    def __init__(self, config: Config) -> None:
        """Inicializa el cliente recordando valores por defecto y credenciales."""

        self._configured_api_key = (config.OPENAI_API_KEY or "").strip()
        self._client = (
            OpenAI(api_key=self._configured_api_key)
            if self._configured_api_key
            else None
        )
        self._runtime_api_key: Optional[str] = None
        self._model = config.OPENAI_MODEL
        self._schema_name = config.JSON_MODE_SCHEMA_NAME
        self._default_temperature = 1.0
        self._default_top_p = 1.0
        self._default_reasoning_effort = "minimal"
        self._default_frequency_penalty = 0.0
        self._default_presence_penalty = 0.0

    def extract(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Invoca el endpoint de chat completions utilizando modo JSON Schema."""

        chosen_model = (model or self._model).strip()
        selected_reasoning_effort = (
            self._default_reasoning_effort
            if reasoning_effort is None
            else reasoning_effort
        )
        openai_reasoning_effort = (
            None if selected_reasoning_effort == "none" else selected_reasoning_effort
        )
        resolved_api_key = (api_key or self._configured_api_key or "").strip()
        if not resolved_api_key:
            raise RuntimeError(
                "Proporciona una clave de API de OpenAI válida para completar la solicitud."
            )
        client = self._client
        if client is None or (
            resolved_api_key != self._configured_api_key
            and resolved_api_key != self._runtime_api_key
        ):
            client = OpenAI(api_key=resolved_api_key)
            if not self._configured_api_key:
                self._client = client
                self._runtime_api_key = resolved_api_key
        selected_frequency_penalty = (
            self._default_frequency_penalty
            if frequency_penalty is None
            else frequency_penalty
        )
        selected_presence_penalty = (
            self._default_presence_penalty
            if presence_penalty is None
            else presence_penalty
        )
        response = client.chat.completions.create(
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
            reasoning_effort=openai_reasoning_effort,
            frequency_penalty=selected_frequency_penalty,
            presence_penalty=selected_presence_penalty,
        )
        content = response.choices[0].message.content
        return _parse_model_response(content)


class LocalLLMService:
    """Implementación basada en HuggingFace para ejecutar un modelo local."""

    def __init__(self, config: Config) -> None:
        """Localiza el modelo local y prepara el dispositivo de inferencia."""
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
        """Carga o reutiliza el pipeline de inferencia configurado."""
        source = (model or self._default_model).strip()
        if source not in self._pipelines:
            resolved = Path(source)
            model_source = str(resolved) if resolved.exists() else source
            load_kwargs: Dict[str, Any] = {"trust_remote_code": True}

            config = AutoConfig.from_pretrained(model_source, **load_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_source, **load_kwargs)

            model_kwargs: Dict[str, Any] = {
                "config": config,
                "trust_remote_code": True,
            }
            if torch.cuda.is_available():
                # Si hay GPU disponible, se usa bfloat16 para optimizar memoria
                model_kwargs["torch_dtype"] = torch.bfloat16

            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                **model_kwargs,
            )

            self._pipelines[source] = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map=None,
                device=self._device,
            )
        return self._pipelines[source]

    def extract(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Genera texto con el modelo local y lo interpreta como JSON."""

        _ = api_key  # Compatibilidad con la interfaz API
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
