#!/usr/bin/env python3
"""FastAPI service for extracting structured vehicular invoice data.

This service wraps the original OpenAI-powered extraction workflow in a
fully documented FastAPI application with endpoints for text, file and image
inputs.  The API exposes detailed Swagger documentation (OpenAPI) describing
the supported payloads and responses so that clients can easily integrate the
service into automation pipelines.
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field, constr
import pytesseract


# ---------------------------------------------------------------------------
# PDF reader selection without try/except around imports (project convention)
# ---------------------------------------------------------------------------
_PDF_LIB = None
PdfReader = None
for candidate in ("pypdf", "PyPDF2"):
    module_spec = importlib.util.find_spec(candidate)
    if module_spec is not None:
        module = importlib.import_module(candidate)
        PdfReader = getattr(module, "PdfReader", None)
        if PdfReader is not None:
            _PDF_LIB = candidate
            break
if PdfReader is None:
    raise ImportError(
        "No PDF reader available. Install either `pypdf` or `PyPDF2` to "
        "enable PDF processing."
    )


# ---------------------------------------------------------------------------
# Data models used for the FastAPI documentation
# ---------------------------------------------------------------------------
class InvoiceExtraction(BaseModel):
    """Structured invoice payload returned by the extraction endpoints."""

    FECHA_DOCUMENTO: Optional[str] = Field(
        None, description="Fecha del documento (dd/mm/aaaa o formato claro)."
    )
    DIRECCION: Optional[str] = Field(None, description="Dirección del emisor.")
    MODELO_HOMOLOGADO_ANT: Optional[str] = Field(
        None, description="Modelo homologado anterior si consta."
    )
    SUBSIDIO: Optional[float] = Field(
        None, description="Monto del subsidio reportado en la factura."
    )
    AÑO: Optional[float] = Field(
        None, description="Año del modelo o registro (numérico)."
    )
    SUBTOTAL: Optional[float] = Field(
        None, description="Subtotal monetario de la factura."
    )
    CLASE: Optional[str] = Field(None, description="Clase del vehículo.")
    TOTAL: Optional[float] = Field(
        None, description="Total monetario de la factura sin símbolo de moneda."
    )
    CILINDRAJE: Optional[str] = Field(
        None, description="Cilindraje registrado (formato libre)."
    )
    MODELO: Optional[str] = Field(None, description="Modelo del vehículo.")
    MODELO_REGISTRADO_SRI: Optional[str] = Field(
        None, description="Modelo registrado en el SRI si está disponible."
    )
    RAMV_CPN: Optional[str] = Field(None, description="RAMV/CPN indicado.")
    RUEDAS: Optional[float] = Field(None, description="Número de ruedas.")
    DESCUENTO: Optional[float] = Field(None, description="Descuentos aplicados.")
    NUMERO_FACTURA: Optional[str] = Field(
        None, description="Número de factura normalizado con guiones."
    )
    COLOR: Optional[str] = Field(None, description="Color del vehículo.")
    MOTOR: Optional[str] = Field(None, description="Número de motor.")
    NOMBRE_CLIENTE: Optional[str] = Field(
        None, description="Nombre o razón social del cliente."
    )
    CAPACIDAD: Optional[float] = Field(
        None, description="Capacidad declarada (personas o carga)."
    )
    MARCA: Optional[str] = Field(None, description="Marca del vehículo.")
    RUC: Optional[str] = Field(None, description="RUC o cédula del emisor.")
    COMBUSTIBLE: Optional[str] = Field(
        None, description="Tipo de combustible indicado en la factura."
    )
    EJES: Optional[float] = Field(None, description="Número de ejes del vehículo.")
    TIPO: Optional[str] = Field(None, description="Tipo de vehículo según el documento.")
    IVA: Optional[float] = Field(None, description="Valor del IVA.")
    CONCESIONARIA: Optional[str] = Field(
        None, description="Nombre de la concesionaria si aplica."
    )
    TONELAJE: Optional[float] = Field(
        None, description="Tonelaje reportado en la documentación."
    )
    VIN_CHASIS: Optional[str] = Field(None, description="VIN o número de chasis.")
    PAIS_ORIGEN: Optional[str] = Field(None, description="País de origen del vehículo.")
    ETIQUETA: Optional[str] = Field(
        None, description="Etiquetas de completitud o estado del documento."
    )
    SUBTOTAL_12: Optional[float] = Field(
        None, description="Subtotal gravado al 12% si está disponible."
    )
    SUBTOTAL_0: Optional[float] = Field(
        None, description="Subtotal gravado al 0% si está disponible."
    )
    MONEDA: Optional[str] = Field(
        None, description="Moneda de referencia, por ejemplo USD."
    )
    RAZON_SOCIAL: Optional[str] = Field(None, description="Razón social del emisor.")
    ANIO_MODELO: Optional[str] = Field(
        None, description="Año del modelo según la factura (texto libre)."
    )
    observaciones: Optional[str] = Field(
        None, description="Notas de extracción o advertencias detectadas."
    )
    confidence: Optional[Dict[str, float]] = Field(
        None, description="Mapa campo→confianza normalizada (0.0 – 1.0)."
    )
    _file: Optional[str] = Field(
        None,
        alias="_file",
        description="Nombre de archivo asociado cuando el origen es un documento.",
    )
    error: Optional[str] = Field(
        None, description="Mensaje de error si la extracción falló parcialmente."
    )

    class Config:
        allow_population_by_field_name = True
        extra = "allow"


class TextExtractionRequest(BaseModel):
    """Body payload for raw text extraction."""

    text: constr(strip_whitespace=True, min_length=1) = Field(
        ..., description="Texto plano sobre el cual se realizará la extracción."
    )
    model: str = Field(
        "gpt-4.1-mini",
        description="Modelo de OpenAI a utilizar para la extracción estructurada.",
    )


class ExtractionSummary(BaseModel):
    """Informative response for metadata endpoints."""

    message: str
    pdf_backend: str = Field(..., description="Librería utilizada para procesar PDFs.")


# ---------------------------------------------------------------------------
# Utility helpers reused across the endpoints
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Eres un analista experto en extracción de datos de facturas vehiculares ecuatorianas.
Devuelve **un único objeto JSON** con **exactamente** las siguientes claves y tipos (no agregues ni quites claves):
{
  "FECHA_DOCUMENTO": string|null,          # dd/mm/aaaa si es posible
  "DIRECCION": string|null,
  "MODELO_HOMOLOGADO_ANT": string|null,
  "SUBSIDIO": number|null,                  # si no consta, usa null (no NaN)
  "AÑO": number|null,                       # año modelo o registro; número (p.ej. 2023)
  "SUBTOTAL": number|null,
  "CLASE": string|null,
  "TOTAL": number|null,
  "CILINDRAJE": string|null,                # conservar formato (p.ej. "1.451 C.C.")
  "MODELO": string|null,
  "MODELO_REGISTRADO_SRI": string|null,
  "RAMV_CPN": string|null,
  "RUEDAS": number|null,
  "DESCUENTO": number|null,
  "NUMERO_FACTURA": string|null,            # normaliza con guiones completos (ej.: 002-101-000004850)
  "COLOR": string|null,
  "MOTOR": string|null,
  "NOMBRE_CLIENTE": string|null,
  "CAPACIDAD": number|null,
  "MARCA": string|null,
  "RUC": string|null,
  "COMBUSTIBLE": string|null,
  "EJES": number|null,
  "TIPO": string|null,
  "IVA": number|null,
  "CONCESIONARIA": string|null,
  "TONELAJE": number|null,
  "VIN_CHASIS": string|null,
  "PAIS_ORIGEN": string|null,
  "ETIQUETA": string|null                   # p.ej. "COMPLETA" si el documento parece completo
}

REGLAS:
- NO inventes datos. Si un valor no está, usa null. **Nunca uses NaN** (JSON no lo permite).
- Devuelve montos y cantidades como números (no como texto). No incluyas símbolos de moneda.
- FECHA_DOCUMENTO en formato dd/mm/aaaa si el texto lo permite; de lo contrario, conserva un formato claro.
- NUMERO_FACTURA normalizado con guiones si aplica.
- Limpia espacios, saltos de línea, y texto de sellos/footers.
- NO devuelvas ningún texto adicional fuera del JSON, ni explicaciones, ni claves extra.
"""


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Attempt to recover a top-level JSON object from a model response."""

    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
    except Exception:
        pass

    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
        start = text.find("{", start + 1)

    try:
        sanitized = re.sub(r",\s*}", "}", text)
        obj = json.loads(sanitized)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    raise RuntimeError("No pude extraer un objeto JSON válido del texto de salida del modelo.")


def read_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF using the selected backend."""

    reader = PdfReader(str(pdf_path))
    parts: List[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        parts.append(page_text)
    text = "\n".join(parts).strip()
    return " ".join(text.split())


def chunk_text(text: str, max_chars: int = 50_000) -> List[str]:
    """Naively split large texts to stay within model limits."""

    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        split_at = text.rfind("\n\n", start, end)
        if split_at == -1:
            split_at = text.rfind(". ", start, end)
        if split_at == -1:
            split_at = end
        chunks.append(text[start:split_at].strip())
        start = split_at
    return [c for c in chunks if c]


def _initialise_openai_client() -> OpenAI:
    """Create an OpenAI client using environment configuration."""

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Configure it as an environment variable or in a .env file."
        )
    return OpenAI(api_key=api_key)


_OPENAI_CLIENT: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    """Lazy-load and cache the OpenAI client for reuse across requests."""

    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = _initialise_openai_client()
    return _OPENAI_CLIENT


def call_openai_structured(client: OpenAI, model: str, pdf_text: str) -> Dict[str, Any]:
    """Invoke OpenAI to obtain a structured response from the provided text."""

    chunks = chunk_text(pdf_text, max_chars=45_000)
    partial_results: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Extrae los campos del JSON Schema a partir del siguiente texto "
                        f"OCR de la factura (parte {idx + 1}/{len(chunks)}):{chunk}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )

        data_json: Optional[Dict[str, Any]] = None
        if getattr(response, "choices", None):
            message = getattr(response.choices[0], "message", None)
            content = getattr(message, "content", None) if message is not None else None
            if content:
                data_json = _extract_json_object(content)
        if not isinstance(data_json, dict):
            raise RuntimeError("La respuesta del modelo no es un JSON de objeto válido. Revisa el SDK/versión.")
        partial_results.append(data_json)

    merged: Dict[str, Any] = {}
    confidence: Dict[str, float] = {}
    for result in partial_results:
        for key, value in result.items():
            if key == "confidence" and isinstance(value, dict):
                for c_key, c_value in value.items():
                    try:
                        confidence[c_key] = max(confidence.get(c_key, 0.0), float(c_value))
                    except Exception:
                        continue
                continue
            if key not in merged or merged.get(key) in (None, "", []):
                merged[key] = value
    if confidence:
        merged["confidence"] = confidence
    return merged


def extract_invoice_from_text(text: str, model: str, client: OpenAI) -> InvoiceExtraction:
    """Shared orchestration method used by all endpoints."""

    if not text or not text.strip():
        raise ValueError("No se proporcionó texto válido para procesar.")
    data = call_openai_structured(client, model, text.strip())
    return InvoiceExtraction.parse_obj(data)


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """Run OCR over the provided image payload using Tesseract."""

    image = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# FastAPI application with rich documentation
# ---------------------------------------------------------------------------
tags_metadata = [
    {
        "name": "Extraction",
        "description": (
            "Operaciones para transformar facturas vehiculares en JSON estructurado "
            "utilizando modelos de OpenAI. Las entradas pueden ser texto plano, "
            "archivos (PDF, XML, TXT, JSON) o imágenes donde se aplicará OCR."
        ),
    }
]

app = FastAPI(
    title="Verifactura Invoice Extraction API",
    description=(
        "API para convertir facturas vehiculares ecuatorianas en un JSON estandarizado. "
        "Incluye extracción directa de texto, lectura de archivos y OCR para imágenes."
    ),
    version="2.0.0",
    openapi_tags=tags_metadata,
    contact={
        "name": "Verifactura",
        "url": "https://example.com",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)


def _get_client_dependency() -> OpenAI:
    try:
        return get_openai_client()
    except RuntimeError as exc:  # pragma: no cover - configuration error path
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/",
    response_model=ExtractionSummary,
    summary="Información del servicio",
    tags=["Extraction"],
)
def service_info() -> ExtractionSummary:
    """Return high-level metadata about the running service."""

    return ExtractionSummary(
        message="Servicio de extracción de facturas listo.",
        pdf_backend=_PDF_LIB or "desconocido",
    )


@app.post(
    "/extract/text",
    response_model=InvoiceExtraction,
    summary="Extraer datos desde texto plano",
    response_description="JSON estructurado con los campos de la factura.",
    tags=["Extraction"],
)
async def extract_from_text(
    payload: TextExtractionRequest,
    client: OpenAI = Depends(_get_client_dependency),
) -> InvoiceExtraction:
    """Procesa texto plano enviado en el cuerpo de la petición."""

    try:
        return await run_in_threadpool(
            extract_invoice_from_text, payload.text, payload.model, client
        )
    except Exception as exc:  # pragma: no cover - depends on OpenAI responses
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/extract/file",
    response_model=InvoiceExtraction,
    summary="Extraer datos desde un archivo",
    response_description="JSON estructurado con los campos de la factura.",
    tags=["Extraction"],
)
async def extract_from_file(
    file: UploadFile = File(
        ..., description="Archivo PDF, XML, JSON o TXT que contiene una factura."),
    model: str = Query(
        "gpt-4.1-mini",
        description="Modelo de OpenAI a utilizar para la extracción.",
    ),
    client: OpenAI = Depends(_get_client_dependency),
) -> InvoiceExtraction:
    """Procesa un archivo cargado para obtener el JSON de la factura."""

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="El archivo cargado está vacío.")

    suffix = (Path(file.filename or "").suffix or "").lower()
    text: Optional[str] = None

    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            text = await run_in_threadpool(read_pdf_text, tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    elif suffix in {".txt", ".xml", ".json", ".csv"}:
        text = data.decode("utf-8", errors="ignore")
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=415,
                detail=(
                    "Tipo de archivo no soportado. Proporcione PDF, XML, JSON, TXT o un "
                    "archivo de texto compatible."
                ),
            ) from exc

    cleaned_text = " ".join((text or "").split())
    if not cleaned_text:
        raise HTTPException(
            status_code=422, detail="No se pudo extraer texto legible del archivo subido."
        )

    try:
        extraction = await run_in_threadpool(
            extract_invoice_from_text, cleaned_text, model, client
        )
        extraction._file = file.filename
        return extraction
    except Exception as exc:  # pragma: no cover - depende del backend externo
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/extract/image",
    response_model=InvoiceExtraction,
    summary="Extraer datos desde una imagen",
    response_description="JSON estructurado derivado del texto OCR de la imagen.",
    tags=["Extraction"],
)
async def extract_from_image(
    image: UploadFile = File(
        ..., description="Imagen (PNG, JPG, TIFF, BMP) que contenga la factura."),
    model: str = Query(
        "gpt-4.1-mini",
        description="Modelo de OpenAI a utilizar para la extracción.",
    ),
    client: OpenAI = Depends(_get_client_dependency),
) -> InvoiceExtraction:
    """Realiza OCR sobre una imagen y extrae el JSON de la factura resultante."""

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="La imagen cargada está vacía.")

    try:
        text = await run_in_threadpool(extract_text_from_image_bytes, data)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="No se pudo procesar la imagen u obtener texto mediante OCR.",
        ) from exc

    if not text:
        raise HTTPException(
            status_code=422,
            detail="El OCR no detectó texto en la imagen proporcionada.",
        )

    try:
        extraction = await run_in_threadpool(
            extract_invoice_from_text, text, model, client
        )
        extraction._file = image.filename
        return extraction
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.exception_handler(RuntimeError)
async def runtime_error_handler(_, exc: RuntimeError) -> JSONResponse:
    """Return runtime errors as JSON API responses."""

    return JSONResponse(status_code=500, content={"detail": str(exc)})


__all__ = [
    "app",
    "InvoiceExtraction",
    "extract_from_text",
    "extract_from_file",
    "extract_from_image",
]
