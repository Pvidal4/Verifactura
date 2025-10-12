from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.config import Config
from app.services.extraction_service import IMAGE_EXTENSIONS, ExtractionService

router = APIRouter(tags=["Extracción"])

class TextExtractionRequest(BaseModel):
    text: str = Field(
        ..., description="Texto completo del comprobante o documento a procesar."
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
    return service.extract_from_text(text)


@router.post(
    "/extract/file",
    summary="Subir un archivo (PDF, XML o texto) para su extracción",
    description=(
        "Adjunta un archivo soportado (PDF, XML, TXT, CSV) para procesarlo. "
        "Las imágenes deben enviarse mediante el endpoint dedicado a OCR."
    ),
    response_description="Resultado JSON con los campos extraídos.",
)
async def extract_from_file_endpoint(
    file: UploadFile = File(...),
    service: ExtractionService = Depends(_get_service),
) -> Dict[str, Any]:

    _validate_not_image(file)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="El archivo subido está vacío.")
    try:
        return service.extract_from_file(file.filename or "archivo", data, file.content_type)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/extract/image",
    summary="Extraer texto a partir de una imagen",
    description=(
        "Acepta imágenes (PNG, JPG, TIFF, BMP, GIF) y aplica OCR de Azure antes de "
        "enviar el contenido al modelo de lenguaje."
    ),
    response_description="Resultado JSON con los campos extraídos tras el OCR.",
)
async def extract_from_image_endpoint(
    image: UploadFile = File(...),
    service: ExtractionService = Depends(_get_service),
) -> Dict[str, Any]:

    if not (image.content_type or "").startswith("image/"):
        suffix = Path((image.filename or "").lower()).suffix
        if suffix not in IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="El archivo proporcionado no es una imagen soportada.",
            )
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="La imagen subida está vacía.")
    try:
        return service.extract_from_image(
            image.filename or "imagen", data, image.content_type
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
