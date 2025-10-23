"""Rutas HTTP relacionadas con la extracción de información estructurada."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from app.config import Config
from app.services.extraction_service import (
    IMAGE_EXTENSIONS,
    PDF_EXTENSIONS,
    ExtractionService,
)

router = APIRouter(tags=["Extracción"])


class TextExtractionRequest(BaseModel):
    """Cuerpo esperado cuando el cliente envía texto plano para procesar."""

    text: str = Field(
        ..., description="Texto completo del comprobante o documento a procesar."
    )
    llm_provider: Optional[Literal["api", "local"]] = Field(
        None,
        description=(
            "Proveedor del modelo de lenguaje a utilizar. "
            "Usa 'api' para OpenAI o 'local' para el modelo con pesos abiertos."
        ),
    )
    llm_model: Optional[str] = Field(
        None,
        description=(
            "Modelo específico a utilizar para el proveedor seleccionado."
        ),
    )
    temperature: Optional[float] = Field(
        None,
        ge=0,
        le=2,
        description="Temperatura para el muestreo del modelo (0-2).",
    )
    top_p: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Top-p (nucleus sampling) del modelo (0-1).",
    )
    reasoning_effort: Optional[
        Literal["none", "minimal", "low", "medium", "high"]
    ] = Field(
        None,
        description=(
            "Nivel de esfuerzo de razonamiento a solicitar al modelo. "
            "Selecciona 'none' para omitir este parámetro."
        ),
    )
    frequency_penalty: Optional[float] = Field(
        None,
        ge=-2,
        le=2,
        description="Penalización por frecuencia (-2 a 2).",
    )
    presence_penalty: Optional[float] = Field(
        None,
        ge=-2,
        le=2,
        description="Penalización por presencia (-2 a 2).",
    )
    openai_api_key: Optional[str] = Field(
        None,
        description="Clave de API de OpenAI a utilizar para la solicitud actual.",
    )


def _normalize_reasoning_effort(
    value: Optional[Literal["none", "minimal", "low", "medium", "high"]]
) -> Optional[str]:
    """Convierte el valor recibido en un modo válido para OpenAI."""

    if value == "none":
        return None
    return value


def _normalize_optional_string(value: Optional[str]) -> Optional[str]:
    """Limpia cadenas opcionales evitando valores vacíos o con espacios."""

    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


def _normalize_api_key(value: Optional[str]) -> Optional[str]:
    """Reutiliza la normalización genérica para claves de API."""

    return _normalize_optional_string(value)


def _normalize_ocr_provider(value: Optional[str]) -> Optional[str]:
    """Valida el proveedor OCR admitido y estandariza la cadena proporcionada."""

    normalized = _normalize_optional_string(value)
    if not normalized:
        return None
    lowered = normalized.lower()
    if lowered in {"azure-vision", "azure_vision", "azurevision", "azure"}:
        return "azure-vision"
    raise HTTPException(
        status_code=400,
        detail=f"Proveedor OCR '{value}' no es válido.",
    )


def _get_service(request: Request) -> ExtractionService:
    """Obtiene o inicializa el servicio de extracción y lo cachea en la app."""

    service: Optional[ExtractionService] = getattr(
        request.app.state, "extraction_service", None
    )
    if service is None:
        config: Config = getattr(request.app.state, "config", Config())
        service = ExtractionService(config)
        request.app.state.extraction_service = service
    return service


def _validate_not_image(upload: UploadFile) -> None:
    """Evita que se procese una imagen en el endpoint de archivos generales."""

    filename = (upload.filename or "").lower()
    suffix = Path(filename).suffix.lower()
    if suffix in IMAGE_EXTENSIONS or (upload.content_type or "").startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Utiliza el endpoint de imágenes para procesar archivos gráficos.",
        )


@router.post(
    "/extract/text",
    summary="Extraer información estructurada desde texto plano",
    description=(
        "Envía un texto plano con el contenido completo del comprobante para obtener "
        "la extracción estructurada generada por el modelo de lenguaje."
    ),
    response_description="Resultado JSON con los campos extraídos.",
)
async def extract_from_text_endpoint(
    payload: TextExtractionRequest, service: ExtractionService = Depends(_get_service)
) -> Dict[str, Any]:
    """Procesa texto plano y devuelve la extracción estructurada generada."""

    text = payload.text.strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail="El texto proporcionado está vacío.",
        )
    result = service.extract_from_text(
        text,
        provider=payload.llm_provider,
        model=payload.llm_model,
        temperature=payload.temperature,
        top_p=payload.top_p,
        reasoning_effort=_normalize_reasoning_effort(payload.reasoning_effort),
        frequency_penalty=payload.frequency_penalty,
        presence_penalty=payload.presence_penalty,
        openai_api_key=_normalize_api_key(payload.openai_api_key),
    )
    return result.to_payload()


@router.post(
    "/extract/file",
    summary="Subir un archivo (PDF, XML o JSON) para su extracción",
    description=(
        "Adjunta un archivo soportado (PDF, XML, JSON) para procesarlo. "
        "Las imágenes deben enviarse mediante el endpoint dedicado a OCR."
    ),
    response_description="Resultado JSON con los campos extraídos.",
)
async def extract_from_file_endpoint(
    file: UploadFile = File(...),
    force_ocr: bool = Query(
        False,
        description=(
            "Forzar el uso de OCR incluso si se puede leer texto directamente (PDF). "
            "Convierte cada página del PDF en imagen antes de aplicar Azure OCR."
        ),
    ),
    llm_provider: Optional[Literal["api", "local"]] = Query(
        None,
        description=(
            "Proveedor del modelo de lenguaje a utilizar. "
            "Usa 'api' para OpenAI o 'local' para el modelo con pesos abiertos."
        ),
    ),
    llm_model: Optional[str] = Query(
        None,
        description=(
            "Modelo específico a utilizar para el proveedor seleccionado."
        ),
    ),
    temperature: Optional[float] = Query(
        None,
        ge=0,
        le=2,
        description="Temperatura para el muestreo del modelo (0-2).",
    ),
    top_p: Optional[float] = Query(
        None,
        ge=0,
        le=1,
        description="Top-p (nucleus sampling) del modelo (0-1).",
    ),
    reasoning_effort: Optional[
        Literal["none", "minimal", "low", "medium", "high"]
    ] = Query(
        None,
        description=(
            "Nivel de esfuerzo de razonamiento a solicitar al modelo. "
            "Selecciona 'none' para omitir este parámetro."
        ),
    ),
    frequency_penalty: Optional[float] = Query(
        None,
        ge=-2,
        le=2,
        description="Penalización por frecuencia (-2 a 2).",
    ),
    presence_penalty: Optional[float] = Query(
        None,
        ge=-2,
        le=2,
        description="Penalización por presencia (-2 a 2).",
    ),
    openai_api_key: Optional[str] = Query(
        None,
        description="Clave de API de OpenAI a utilizar para la solicitud actual.",
    ),
    ocr_provider: Optional[str] = Query(
        None,
        description="Proveedor OCR a utilizar (por ejemplo, 'azure-vision').",
    ),
    azure_form_recognizer_endpoint: Optional[str] = Query(
        None,
        description="Endpoint de Azure Form Recognizer para esta solicitud.",
    ),
    azure_form_recognizer_key: Optional[str] = Query(
        None,
        description="Clave de Azure Form Recognizer para esta solicitud.",
    ),
    service: ExtractionService = Depends(_get_service),
) -> Dict[str, Any]:
    """Gestiona la carga de archivos y orquesta la lógica de OCR y extracción."""

    _validate_not_image(file)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="El archivo subido está vacío.")
    try:
        result = service.extract_from_file(
            file.filename or "archivo",
            data,
            file.content_type,
            force_ocr=force_ocr,
            provider=llm_provider,
            model=llm_model,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=_normalize_reasoning_effort(reasoning_effort),
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            openai_api_key=_normalize_api_key(openai_api_key),
            ocr_provider=_normalize_ocr_provider(ocr_provider),
            ocr_endpoint=_normalize_optional_string(azure_form_recognizer_endpoint),
            ocr_key=_normalize_optional_string(azure_form_recognizer_key),
        )
        return result.to_payload()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/extract/image",
    summary="Extraer texto a partir de una imagen",
    description=(
        "Acepta imágenes (PNG, JPG, TIFF) y aplica OCR de Azure antes de "
        "enviar el contenido al modelo de lenguaje."
    ),
    response_description="Resultado JSON con los campos extraídos tras el OCR.",
)
async def extract_from_image_endpoint(
    image: UploadFile = File(...),
    llm_provider: Optional[Literal["api", "local"]] = Query(
        None,
        description=(
            "Proveedor del modelo de lenguaje a utilizar. "
            "Usa 'api' para OpenAI o 'local' para el modelo con pesos abiertos."
        ),
    ),
    llm_model: Optional[str] = Query(
        None,
        description=(
            "Modelo específico a utilizar para el proveedor seleccionado."
        ),
    ),
    temperature: Optional[float] = Query(
        None,
        ge=0,
        le=2,
        description="Temperatura para el muestreo del modelo (0-2).",
    ),
    top_p: Optional[float] = Query(
        None,
        ge=0,
        le=1,
        description="Top-p (nucleus sampling) del modelo (0-1).",
    ),
    reasoning_effort: Optional[
        Literal["none", "minimal", "low", "medium", "high"]
    ] = Query(
        None,
        description=(
            "Nivel de esfuerzo de razonamiento a solicitar al modelo. "
            "Selecciona 'none' para omitir este parámetro."
        ),
    ),
    frequency_penalty: Optional[float] = Query(
        None,
        ge=-2,
        le=2,
        description="Penalización por frecuencia (-2 a 2).",
    ),
    presence_penalty: Optional[float] = Query(
        None,
        ge=-2,
        le=2,
        description="Penalización por presencia (-2 a 2).",
    ),
    openai_api_key: Optional[str] = Query(
        None,
        description="Clave de API de OpenAI a utilizar para la solicitud actual.",
    ),
    ocr_provider: Optional[str] = Query(
        None,
        description="Proveedor OCR a utilizar (por ejemplo, 'azure-vision').",
    ),
    azure_form_recognizer_endpoint: Optional[str] = Query(
        None,
        description="Endpoint de Azure Form Recognizer para esta solicitud.",
    ),
    azure_form_recognizer_key: Optional[str] = Query(
        None,
        description="Clave de Azure Form Recognizer para esta solicitud.",
    ),
    service: ExtractionService = Depends(_get_service),
) -> Dict[str, Any]:
    """Extrae texto mediante OCR y delega la estructuración en el servicio LLM."""

    content_type = (image.content_type or "").lower()
    suffix = Path((image.filename or "").lower()).suffix
    if not (
        content_type.startswith("image/")
        or suffix in IMAGE_EXTENSIONS
        or suffix in PDF_EXTENSIONS
        or content_type == "application/pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="El archivo proporcionado no es una imagen soportada.",
        )
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="La imagen subida está vacía.")
    try:
        result = service.extract_from_image(
            image.filename or "imagen",
            data,
            image.content_type,
            provider=llm_provider,
            model=llm_model,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=_normalize_reasoning_effort(reasoning_effort),
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            openai_api_key=_normalize_api_key(openai_api_key),
            ocr_provider=_normalize_ocr_provider(ocr_provider),
            ocr_endpoint=_normalize_optional_string(azure_form_recognizer_endpoint),
            ocr_key=_normalize_optional_string(azure_form_recognizer_key),
        )
        return result.to_payload()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
