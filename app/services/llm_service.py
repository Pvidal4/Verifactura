from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from openai import OpenAI
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module

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
        return _parse_model_response(content)


class LocalLLMService:
    def __init__(self, config: Config) -> None:
        self._model_id = config.LOCAL_LLM_MODEL_ID
        self._model_path = config.LOCAL_LLM_MODEL_PATH
        self._pipeline = None

    def _resolve_model_source(self) -> str:
        if self._model_path:
            local_path = Path(self._model_path)
            if local_path.exists():
                return str(local_path)
        return self._model_id

    def _load_tokenizer_config(self, model_source: str) -> Tuple[Dict[str, Any], List[str]]:
        errors: List[str] = []
        config_data: Dict[str, Any] = {}

        local_config = Path(model_source) / "tokenizer_config.json"
        if local_config.exists():
            try:
                config_data = json.loads(local_config.read_text())
                return config_data, errors
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"tokenizer_config.json lectura local: {exc}")

        try:
            from huggingface_hub import hf_hub_download

            downloaded_path = hf_hub_download(
                model_source,
                filename="tokenizer_config.json",
            )
            config_data = json.loads(Path(downloaded_path).read_text())
        except Exception as exc:  # pragma: no cover - defensivo
            errors.append(f"tokenizer_config.json descarga: {exc}")
        return config_data, errors

    def _ensure_pipeline(self):
        if self._pipeline is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model_source = self._resolve_model_source()
            try:
                config = AutoConfig.from_pretrained(
                    model_source,
                    trust_remote_code=True,
                )
            except KeyError:
                config = None
            tokenizer = None
            tokenizer_errors: List[str] = []
            tokenizer_config: Dict[str, Any] = {}
            tokenizer_config_errors: List[str] = []
            tokenizer_attempts = [
                {"use_fast": True},
                {"use_fast": False},
            ]
            local_tokenizer_file = Path(model_source) / "tokenizer.model"
            if local_tokenizer_file.exists():
                tokenizer_attempts.append(
                    {
                        "use_fast": False,
                        "tokenizer_file": str(local_tokenizer_file),
                    }
                )

            for attempt in tokenizer_attempts:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_source,
                        trust_remote_code=True,
                        **attempt,
                    )
                    break
                except Exception as exc:
                    tokenizer_errors.append(f"{attempt}: {exc}")

            if tokenizer is None and config is not None:
                auto_map = getattr(config, "auto_map", None)
                mapping = None
                if isinstance(auto_map, dict):
                    mapping = auto_map.get("AutoTokenizer")
                if mapping:
                    dynamic_spec: str
                    if isinstance(mapping, (list, tuple)) and mapping:
                        dynamic_spec = mapping[0]
                    else:
                        dynamic_spec = str(mapping)
                    try:
                        tokenizer_cls = get_class_from_dynamic_module(
                            dynamic_spec,
                            model_source,
                            trust_remote_code=True,
                        )
                        tokenizer = tokenizer_cls.from_pretrained(
                            model_source,
                            trust_remote_code=True,
                        )
                    except Exception as exc:
                        tokenizer_errors.append(
                            f"dynamic AutoTokenizer {dynamic_spec}: {exc}"
                        )

            if tokenizer is None:
                tokenizer_config, tokenizer_config_errors = self._load_tokenizer_config(
                    model_source
                )

            if tokenizer is None and tokenizer_config:
                tokenizer_class = tokenizer_config.get("tokenizer_class")
                dynamic_candidates: List[str] = []
                if isinstance(tokenizer_class, str) and tokenizer_class:
                    if "." in tokenizer_class:
                        dynamic_candidates.append(tokenizer_class)
                    elif config is not None and getattr(config, "model_type", None):
                        model_type = getattr(config, "model_type")
                        dynamic_candidates.append(
                            f"tokenization_{model_type}.{tokenizer_class}"
                        )
                        dynamic_candidates.append(tokenizer_class)

                for candidate in dynamic_candidates:
                    try:
                        tokenizer_cls = get_class_from_dynamic_module(
                            candidate,
                            model_source,
                            trust_remote_code=True,
                        )
                        tokenizer = tokenizer_cls.from_pretrained(
                            model_source,
                            trust_remote_code=True,
                        )
                        break
                    except Exception as exc:
                        tokenizer_errors.append(
                            f"dynamic tokenizer {candidate}: {exc}"
                        )

            if tokenizer is None:
                error_sources = tokenizer_errors + tokenizer_config_errors
                error_details = " | ".join(error_sources)
                raise RuntimeError(
                    "No se pudo cargar el tokenizador del modelo local. "
                    "Verifica la descarga de los pesos o vuelve a intentarlo con "
                    "un paquete actualizado de transformers. "
                    f"Detalles: {error_details}"
                )
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                config=config,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto",
            )
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=dtype,
                device_map="auto",
            )
        return self._pipeline

    def _extract_content(self, outputs: Sequence[Dict[str, Any]]) -> str:
        if not outputs:
            raise RuntimeError("El modelo local no generó ninguna respuesta.")
        generated = outputs[0].get("generated_text")
        if isinstance(generated, list):
            # Los modelos con formato de chat retornan una lista de mensajes
            final_message = generated[-1]
            if isinstance(final_message, dict):
                return str(final_message.get("content", ""))
            return str(final_message)
        return str(generated or "")

    def extract(self, text: str) -> Dict[str, Any]:
        pipe = self._ensure_pipeline()
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        outputs = pipe(
            messages,
            max_new_tokens=256,
        )
        content = self._extract_content(outputs)
        if not content.strip():
            raise RuntimeError("El modelo local devolvió una respuesta vacía.")
        return _parse_model_response(content)
