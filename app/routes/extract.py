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
    text: str = Field(
        ..., description="Texto completo del comprobante o documento a procesar."
    )
    llm_provider: Literal["api", "local"] | None = Field(
        None,
        description=(
            "Proveedor del modelo de lenguaje a utilizar. "
            "Usa 'api' para OpenAI o 'local' para el modelo con pesos abiertos."
        ),
    )


def _get_service(request: Request) -> ExtractionService:
    service: Optional[ExtractionService] = getattr(
        request.app.state, "extraction_service", None
    )
    if service is None:
        config: Config = getattr(request.app.state, "config", Config())
        service = ExtractionService(config)
        request.app.state.extraction_service = service
    return service


def _validate_not_image(upload: UploadFile) -> None:
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
    
    text = payload.text.strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail="El texto proporcionado está vacío.",
        )
    return service.extract_from_text(text, provider=payload.llm_provider)


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
    llm_provider: Literal["api", "local"] | None = Query(
        None,
        description=(
            "Proveedor del modelo de lenguaje a utilizar. "
            "Usa 'api' para OpenAI o 'local' para el modelo con pesos abiertos."
        ),
    ),
    service: ExtractionService = Depends(_get_service),
) -> Dict[str, Any]:

    _validate_not_image(file)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="El archivo subido está vacío.")
    try:
        return service.extract_from_file(
            file.filename or "archivo",
            data,
            file.content_type,
            force_ocr=force_ocr,
            provider=llm_provider,
        )
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
    llm_provider: Literal["api", "local"] | None = Query(
        None,
        description=(
            "Proveedor del modelo de lenguaje a utilizar. "
            "Usa 'api' para OpenAI o 'local' para el modelo con pesos abiertos."
        ),
    ),
    service: ExtractionService = Depends(_get_service),
) -> Dict[str, Any]:

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
        return service.extract_from_image(
            image.filename or "imagen",
            data,
            image.content_type,
            provider=llm_provider,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
