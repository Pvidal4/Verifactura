from __future__ import annotations

import inspect
import json
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from openai import OpenAI
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
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


_DYNAMIC_MODULE_SIGNATURE: Optional[inspect.Signature] = None


def _get_dynamic_module_signature() -> inspect.Signature:
    global _DYNAMIC_MODULE_SIGNATURE
    if _DYNAMIC_MODULE_SIGNATURE is None:
        _DYNAMIC_MODULE_SIGNATURE = inspect.signature(get_class_from_dynamic_module)
    return _DYNAMIC_MODULE_SIGNATURE


def _load_dynamic_class(
    module_name: str,
    class_name: str,
    model_source: str,
) -> type:
    signature = _get_dynamic_module_signature()
    params = signature.parameters
    supports_named = "class_name" in params and "module_file" in params
    module_candidates = [module_name]
    if (
        supports_named
        and module_name
        and not module_name.endswith(".py")
    ):
        module_candidates.append(f"{module_name}.py")
    last_error: Optional[Exception] = None
    for module_candidate in module_candidates:
        if supports_named:
            kwargs: Dict[str, Any] = {
                "class_name": class_name,
                "module_file": module_candidate,
            }
            if "pretrained_model_name_or_path" in params:
                kwargs["pretrained_model_name_or_path"] = model_source
            if "model_id" in params:
                kwargs["model_id"] = model_source
            if "trust_remote_code" in params:
                kwargs["trust_remote_code"] = True
            try:
                return get_class_from_dynamic_module(**kwargs)
            except Exception as exc:  # pragma: no cover - defensive
                last_error = exc
                continue
        else:
            reference = (
                f"{module_candidate}.{class_name}" if module_candidate else class_name
            )
            args: List[Any] = [reference]
            param_names = list(params)
            if len(param_names) >= 2 and param_names[1] != "module_file":
                args.append(model_source)
            kwargs: Dict[str, Any] = {}
            if "trust_remote_code" in params:
                kwargs["trust_remote_code"] = True
            try:
                return get_class_from_dynamic_module(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive
                last_error = exc
                continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Dynamic module resolution failed without error detail")


def _parse_dynamic_spec(spec: Any) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    if isinstance(spec, str) and spec:
        if "." in spec:
            module_name, class_name = spec.rsplit(".", 1)
            candidates.append((module_name, class_name))
    elif isinstance(spec, (list, tuple)):
        if len(spec) >= 2:
            module_name, class_name = spec[0], spec[1]
            if isinstance(module_name, str) and isinstance(class_name, str):
                candidates.append((module_name, class_name))
    return candidates


class LocalLLMService:
    def __init__(self, config: Config) -> None:
        self._model_id = config.LOCAL_LLM_MODEL_ID
        self._model_path = config.LOCAL_LLM_MODEL_PATH
        self._pipeline = None

    def _resolve_model_source(self) -> Tuple[str, Optional[Path]]:
        if self._model_path:
            local_path = Path(self._model_path)
            if local_path.exists():
                try:
                    resolved_path = local_path.resolve()
                except (OSError, RuntimeError, ValueError):
                    resolved_path = local_path
                return str(resolved_path), resolved_path
        return self._model_id, None

    def _load_tokenizer_config(
        self, model_source: str, local_model_path: Optional[Path]
    ) -> Tuple[Dict[str, Any], List[str]]:
        errors: List[str] = []
        config_data: Dict[str, Any] = {}

        search_paths: List[Path] = []
        if local_model_path is not None:
            search_paths.append(local_model_path / "tokenizer_config.json")
        else:
            potential_path = Path(model_source) / "tokenizer_config.json"
            if potential_path.exists():
                search_paths.append(potential_path)

        for local_config in search_paths:
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

    def _load_special_tokens(
        self,
        tokenizer: PreTrainedTokenizerFast,
        tokenizer_config: Dict[str, Any],
        model_source: str,
        tokenizer_errors: List[str],
    ) -> None:
        special_tokens_map_file = tokenizer_config.get("special_tokens_map_file")
        if isinstance(special_tokens_map_file, str) and special_tokens_map_file:
            special_tokens_path = Path(special_tokens_map_file)
            if not special_tokens_path.is_absolute():
                special_tokens_path = Path(model_source) / special_tokens_path
            if special_tokens_path.exists():
                try:
                    special_tokens_map = json.loads(special_tokens_path.read_text())
                    tokenizer.add_special_tokens(special_tokens_map)
                except Exception as exc:  # pragma: no cover - defensive
                    tokenizer_errors.append(
                        f"special_tokens_map_file {special_tokens_map_file}: {exc}"
                    )
        token_attributes = [
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
        ]
        for attr in token_attributes:
            value = tokenizer_config.get(attr)
            if isinstance(value, dict):
                value = value.get("content") or value.get("token")
            if isinstance(value, str) and value:
                try:
                    setattr(tokenizer, attr, value)
                except Exception as exc:  # pragma: no cover - defensive
                    tokenizer_errors.append(f"{attr} assignment: {exc}")

        for key in [
            "model_max_length",
            "padding_side",
            "truncation_side",
            "clean_up_tokenization_spaces",
        ]:
            if key in tokenizer_config:
                try:
                    setattr(tokenizer, key, tokenizer_config[key])
                except Exception as exc:  # pragma: no cover - defensive
                    tokenizer_errors.append(f"{key} assignment: {exc}")

    def _ensure_pipeline(self):
        if self._pipeline is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model_source, local_model_path = self._resolve_model_source()
            effective_model_source: Any = model_source
            if local_model_path is not None:
                effective_model_source = local_model_path
            try:
                config = AutoConfig.from_pretrained(
                    effective_model_source,
                    trust_remote_code=True,
                    local_files_only=local_model_path is not None,
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

            search_root = local_model_path or Path(model_source)

            for filename in ("tokenizer.json", "tokenizer.model"):
                local_tokenizer_file = search_root / filename
                if local_tokenizer_file.exists():
                    tokenizer_file_str = local_tokenizer_file.as_posix()
                    tokenizer_attempts.append(
                        {
                            "use_fast": filename.endswith(".json"),
                            "tokenizer_file": tokenizer_file_str,
                        }
                    )

            for attempt in tokenizer_attempts:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        effective_model_source,
                        trust_remote_code=True,
                        local_files_only=local_model_path is not None,
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
                dynamic_specs = _parse_dynamic_spec(mapping)
                for module_name, class_name in dynamic_specs:
                    try:
                        tokenizer_cls = _load_dynamic_class(
                            module_name,
                            class_name,
                            model_source,
                        )
                        tokenizer = tokenizer_cls.from_pretrained(
                            effective_model_source,
                            trust_remote_code=True,
                            local_files_only=local_model_path is not None,
                        )
                        break
                    except Exception as exc:
                        tokenizer_errors.append(
                            f"dynamic AutoTokenizer {module_name}.{class_name}: {exc}"
                        )

            if tokenizer is None:
                tokenizer_config, tokenizer_config_errors = self._load_tokenizer_config(
                    model_source,
                    local_model_path,
                )

            if tokenizer is None and tokenizer_config:
                tokenizer_file = tokenizer_config.get("tokenizer_file")
                if isinstance(tokenizer_file, str) and tokenizer_file:
                    tokenizer_path = Path(tokenizer_file)
                    if not tokenizer_path.is_absolute():
                        tokenizer_path = search_root / tokenizer_path
                    if tokenizer_path.exists():
                        tokenizer_file_str = tokenizer_path.as_posix()
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(
                                effective_model_source,
                                trust_remote_code=True,
                                tokenizer_file=tokenizer_file_str,
                                local_files_only=local_model_path is not None,
                            )
                        except Exception as exc:
                            tokenizer_errors.append(
                                f"tokenizer_config tokenizer_file {tokenizer_file}: {exc}"
                            )

                if tokenizer is None:
                    tokenizer_class = tokenizer_config.get("tokenizer_class")
                tokenizer_module: Optional[str] = None
                dynamic_candidates: List[Tuple[str, str]] = []
                base_class_name: Optional[str] = None
                if tokenizer is None and isinstance(tokenizer_class, str) and tokenizer_class:
                    if "." in tokenizer_class:
                        dynamic_candidates.extend(_parse_dynamic_spec(tokenizer_class))
                    else:
                        base_class_name = tokenizer_class.split(".")[-1]
                        tokenizer_module = tokenizer_config.get("tokenizer_module")
                        if isinstance(tokenizer_module, str) and tokenizer_module:
                            dynamic_candidates.append((tokenizer_module, base_class_name))
                        if config is not None and getattr(config, "model_type", None):
                            model_type = str(getattr(config, "model_type")).replace(
                                "-",
                                "_",
                            )
                            dynamic_candidates.append(
                                (f"tokenization_{model_type}", base_class_name)
                            )
                        snake_class = base_class_name.replace("Tokenizer", "")
                        if snake_class:
                            snake_class = snake_class.replace("-", "_").lower()
                            dynamic_candidates.append(
                                (f"tokenization_{snake_class}", base_class_name)
                            )

                for module_name, class_name in dynamic_candidates:
                    try:
                        tokenizer_cls = _load_dynamic_class(
                            module_name,
                            class_name,
                            model_source,
                        )
                        tokenizer = tokenizer_cls.from_pretrained(
                            effective_model_source,
                            trust_remote_code=True,
                            local_files_only=local_model_path is not None,
                        )
                        break
                    except Exception as exc:
                        tokenizer_errors.append(
                            f"dynamic tokenizer {module_name}.{class_name}: {exc}"
                        )

                if (
                    tokenizer is None
                    and isinstance(tokenizer_class, str)
                    and base_class_name is None
                ):
                    base_class_name = tokenizer_class.split(".")[-1]

                if tokenizer is None and base_class_name:
                    module_name = "transformers"
                    if isinstance(tokenizer_module, str) and tokenizer_module:
                        module_name = tokenizer_module
                    try:
                        module = importlib.import_module(module_name)
                        tokenizer_cls = getattr(module, base_class_name)
                        tokenizer = tokenizer_cls.from_pretrained(
                            effective_model_source,
                            trust_remote_code=True,
                            local_files_only=local_model_path is not None,
                        )
                    except Exception as exc:
                        tokenizer_errors.append(
                            f"direct import {module_name}.{base_class_name}: {exc}"
                        )

            if tokenizer is None:
                manual_candidates: List[Tuple[Path, str]] = []
                tokenizer_file_hint = tokenizer_config.get("tokenizer_file") if tokenizer_config else None
                if isinstance(tokenizer_file_hint, str) and tokenizer_file_hint:
                    candidate_path = Path(tokenizer_file_hint)
                    if not candidate_path.is_absolute():
                        candidate_path = search_root / candidate_path
                    manual_candidates.append((candidate_path, tokenizer_file_hint))
                manual_candidates.append((search_root / "tokenizer.json", "tokenizer.json"))

                for candidate_path, description in manual_candidates:
                    if candidate_path.exists() and candidate_path.suffix == ".json":
                        try:
                            tokenizer = PreTrainedTokenizerFast(
                                tokenizer_file=candidate_path.as_posix(),
                            )
                            if tokenizer_config:
                                self._load_special_tokens(
                                    tokenizer,
                                    tokenizer_config,
                                    model_source,
                                    tokenizer_errors,
                                )
                            break
                        except Exception as exc:
                            tokenizer_errors.append(
                                f"PreTrainedTokenizerFast {description}: {exc}"
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
                effective_model_source,
                config=config,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto",
                local_files_only=local_model_path is not None,
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
