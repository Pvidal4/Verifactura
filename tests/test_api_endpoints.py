"""Pruebas de alto nivel para los endpoints de predicción y extracción."""
from __future__ import annotations

import asyncio
from collections import OrderedDict

import pytest
from fastapi import HTTPException

from app.config import Config
from app.routes.extract import TextExtractionRequest, extract_from_file_endpoint, extract_from_text_endpoint
from app.routes.predictions import PredictionRequest, create_prediction
from app.services.extraction_service import ExtractionResult, ExtractionService
from app.services.prediction_service import PredictionResult


class _StubPredictionService:
    """Servicio simulado que devuelve una predicción determinista."""

    def __init__(self) -> None:
        self.last_features: dict[str, object] | None = None

    def predict(self, features: dict[str, object]) -> PredictionResult:
        self.last_features = dict(features)
        return PredictionResult(
            predicted_class="COMERCIAL",
            probabilities=OrderedDict(
                [
                    ("COMERCIAL", 0.72),
                    ("FAMILIAR", 0.20),
                    ("DEPORTIVO", 0.08),
                ]
            ),
        )


class _FailingPredictionService:
    """Simula una falla inesperada al calcular la predicción."""

    def predict(self, features: dict[str, object]) -> PredictionResult:  # pragma: no cover - camino negativo
        raise RuntimeError("modelo no disponible")


class _StubExtractionService:
    """Servicio de extracción configurable para verificar argumentos."""

    def __init__(self) -> None:
        self.text_calls: list[dict[str, object]] = []
        self.file_calls: list[dict[str, object]] = []

    def extract_from_text(self, text: str, **kwargs) -> ExtractionResult:
        self.text_calls.append({"text": text, **kwargs})
        return ExtractionResult(
            fields={"nit": "123456789", "total": 15000.50},
            raw_text=text,
            text_origin=kwargs.get("text_origin", "input"),
        )

    def extract_from_file(self, filename: str, data: bytes, content_type: str | None, **kwargs) -> ExtractionResult:
        self.file_calls.append(
            {
                "filename": filename,
                "content_type": content_type,
                "size": len(data),
                **kwargs,
            }
        )
        return ExtractionResult(
            fields={"nit": "987654321"},
            raw_text="contenido procesado",
            text_origin="file",
        )


class _DummyUploadFile:
    """Equivalente mínimo de :class:`fastapi.UploadFile` para las pruebas."""

    def __init__(self, filename: str, content_type: str, data: bytes) -> None:
        self.filename = filename
        self.content_type = content_type
        self._data = data

    async def read(self) -> bytes:
        return self._data


class _StubPdfExtractor:
    """Simula un lector de PDF permitiendo controlar su salida en las pruebas."""

    def __init__(self) -> None:
        self.text = "texto pdf"
        self.images = [(b"page-1", "image/png")]
        self.render_calls = 0

    def read_text(self, data: bytes) -> str:
        return self.text

    def render_page_images(self, data: bytes):
        self.render_calls += 1
        return list(self.images)


class _StubAzureOCRService:
    """OCR mínimo que devuelve siempre el mismo texto."""

    def __init__(self) -> None:
        self.text = "texto ocr"
        self.calls: list[tuple[bytes, str | None]] = []

    def extract_text(self, data: bytes, content_type: str | None = None) -> str:
        self.calls.append((data, content_type))
        return self.text


class _InstrumentedExtractionService(ExtractionService):
    """Extensión del servicio real que captura invocaciones para validarlas."""

    def __init__(self) -> None:
        super().__init__(Config())
        self._pdf = _StubPdfExtractor()
        self._ocr_stub = _StubAzureOCRService()
        self.text_invocations: list[dict[str, object]] = []
        self.vision_invocations: list[list[tuple[bytes, str | None]]] = []
        self.ocr_invocations = 0

    def extract_from_text(self, text: str, **kwargs) -> ExtractionResult:  # type: ignore[override]
        payload = {"text": text, **kwargs}
        self.text_invocations.append(payload)
        return ExtractionResult(
            fields={"ok": True},
            raw_text=text,
            text_origin=kwargs.get("text_origin", "file"),
        )

    def _resolve_ocr_service(self, *args, **kwargs):  # type: ignore[override]
        return self._ocr_stub

    def _encode_vision_images(self, images, limit=3):  # type: ignore[override]
        collected = list(images)
        self.vision_invocations.append(collected)
        if not collected:
            return None
        encoded = []
        for index, (_, media_type) in enumerate(collected):
            normalized = (media_type or "image/png").lower()
            encoded.append({"media_type": normalized, "data": f"encoded-{index}"})
        return encoded

    def _extract_text_from_file(self, *args, **kwargs):  # type: ignore[override]
        self.ocr_invocations += 1
        return self._ocr_stub.text


def test_create_prediction_endpoint_returns_payload():
    """Debe entregar la clase predicha y las probabilidades calculadas."""

    service = _StubPredictionService()
    payload = PredictionRequest(
        marca=" Ford ",
        tipo="SUV",
        clase="Camioneta",
        capacidad=5,
        combustible="gasolina",
        ruedas=4,
        total="23.500,80",
    )

    response = asyncio.run(create_prediction(payload, service=service))

    assert response.categoria_predicha == "COMERCIAL"
    assert [p.clase for p in response.probabilidades] == ["COMERCIAL", "FAMILIAR", "DEPORTIVO"]
    assert service.last_features == payload.to_features()


def test_create_prediction_endpoint_handles_service_error():
    """Convierte errores inesperados del servicio en respuestas HTTP 500."""

    payload = PredictionRequest(
        marca="Ford",
        tipo="Sedan",
        clase="Automovil",
        capacidad=4,
        combustible="Gasolina",
        ruedas=4,
        total=18000,
    )

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(create_prediction(payload, service=_FailingPredictionService()))

    assert excinfo.value.status_code == 500
    assert "No se pudo" in str(excinfo.value)


def test_extract_from_text_endpoint_trims_and_returns_payload():
    """Normaliza el texto de entrada antes de delegar en el servicio."""

    service = _StubExtractionService()
    payload = TextExtractionRequest(text="  Total: 10.000  ", llm_provider="api")

    result = asyncio.run(extract_from_text_endpoint(payload, service=service))

    assert result["fields"]["total"] == 15000.50
    assert service.text_calls[0]["text"] == "Total: 10.000"


def test_extract_from_file_endpoint_rejects_images():
    """El endpoint principal de archivos no debe aceptar imágenes directas."""

    upload = _DummyUploadFile("comprobante.png", "image/png", b"data")

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            extract_from_file_endpoint(upload, service=_StubExtractionService())
        )

    assert excinfo.value.status_code == 400


def test_extract_from_file_endpoint_returns_payload():
    """Procesa archivos válidos y retorna el resultado estructurado."""

    service = _StubExtractionService()
    upload = _DummyUploadFile("factura.pdf", "application/pdf", b"pdf-bytes")

    result = asyncio.run(extract_from_file_endpoint(upload, service=service))

    assert result["fields"]["nit"] == "987654321"


def test_extract_from_file_endpoint_forwards_use_vision():
    """Debe propagar el indicador de visión cuando se solicite."""

    service = _StubExtractionService()
    upload = _DummyUploadFile("factura.pdf", "application/pdf", b"pdf-bytes")

    asyncio.run(
        extract_from_file_endpoint(upload, use_vision=True, service=service)
    )

    assert service.file_calls[0]["use_vision"] is True
    assert service.file_calls[0]["filename"] == "factura.pdf"
    assert service.file_calls[0]["size"] == len(b"pdf-bytes")


def test_extraction_service_uses_direct_text_without_vision():
    """Cuando Visión está apagado, solo debe enviarse el texto disponible."""

    service = _InstrumentedExtractionService()
    service._pdf.text = "contenido directo"

    result = service.extract_from_file(
        "factura.pdf", b"%PDF", "application/pdf", use_vision=False
    )

    invocation = service.text_invocations[-1]
    assert invocation["text"] == "contenido directo"
    assert invocation["vision_images"] is None
    assert invocation["text_origin"] == "file"
    assert service._pdf.render_calls == 0
    assert result.raw_text == "contenido directo"
    assert result.text_origin == "file"


def test_extraction_service_adds_images_when_vision_enabled():
    """Si Visión está activo, debe adjuntar capturas además del texto."""

    service = _InstrumentedExtractionService()
    service._pdf.text = "contenido digital"
    service._pdf.images = [(b"page-1", "image/png"), (b"page-2", "image/jpeg")]

    result = service.extract_from_file(
        "factura.pdf", b"%PDF", "application/pdf", use_vision=True
    )

    invocation = service.text_invocations[-1]
    assert invocation["text"] == "contenido digital"
    assert invocation["vision_images"] == [
        {"media_type": "image/png", "data": "encoded-0"},
        {"media_type": "image/jpeg", "data": "encoded-1"},
    ]
    assert service._pdf.render_calls == 1
    assert service.vision_invocations[-1] == [
        (b"page-1", "image/png"),
        (b"page-2", "image/jpeg"),
    ]
    assert result.raw_text == "contenido digital"
    assert result.text_origin == "file"


def test_extraction_service_never_enables_vision_for_xml():
    """Archivos XML solo deben enviar el contenido plano al modelo."""

    service = _InstrumentedExtractionService()

    result = service.extract_from_file(
        "factura.xml",
        b"<factura>contenido</factura>",
        "application/xml",
        force_ocr=True,
        use_vision=True,
    )

    invocation = service.text_invocations[-1]
    assert invocation["text"] == "<factura>contenido</factura>"
    assert invocation["vision_images"] is None
    assert invocation["text_origin"] == "file"
    assert service._pdf.render_calls == 0
    assert service.ocr_invocations == 0
    assert result.text_origin == "file"


def test_extraction_service_never_enables_vision_for_json():
    """Los JSON deben ignorar indicadores de OCR o Visión forzados."""

    service = _InstrumentedExtractionService()

    result = service.extract_from_file(
        "factura.json",
        b'{"monto": 100}',
        "application/json",
        use_vision=True,
        force_ocr=True,
    )

    invocation = service.text_invocations[-1]
    assert invocation["text"] == '{"monto": 100}'
    assert invocation["vision_images"] is None
    assert invocation["text_origin"] == "file"
    assert service._pdf.render_calls == 0
    assert service.ocr_invocations == 0
    assert result.text_origin == "file"


def test_extraction_service_uses_ocr_when_forced():
    """Forzar OCR debe reemplazar el texto plano y marcar el origen correcto."""

    service = _InstrumentedExtractionService()
    service._pdf.text = "texto directo"
    service._ocr_stub.text = "texto via ocr"

    result = service.extract_from_file(
        "factura.pdf", b"%PDF", "application/pdf", force_ocr=True
    )

    invocation = service.text_invocations[-1]
    assert invocation["text"] == "texto via ocr"
    assert invocation["vision_images"] is None
    assert invocation["text_origin"] == "ocr"
    assert service.ocr_invocations == 1
    assert result.text_origin == "ocr"


def test_extraction_service_falls_back_to_ocr_when_no_text():
    """Si el PDF no tiene texto legible, debe activarse OCR automáticamente."""

    service = _InstrumentedExtractionService()
    service._pdf.text = ""
    service._ocr_stub.text = "texto recuperado"

    result = service.extract_from_file("factura.pdf", b"%PDF", "application/pdf")

    invocation = service.text_invocations[-1]
    assert invocation["text"] == "texto recuperado"
    assert invocation["text_origin"] == "ocr"
    assert invocation["vision_images"] is None
    assert service.ocr_invocations == 1
    assert result.text_origin == "ocr"


def test_extraction_service_always_sends_images_with_pixels():
    """Las imágenes deben adjuntar su captura base64 aunque Visión se desactive."""

    service = _InstrumentedExtractionService()
    service._ocr_stub.text = "texto imagen"

    result = service.extract_from_file(
        "foto.png",
        b"\x89PNGdatos",
        "image/png",
        use_vision=False,
    )

    invocation = service.text_invocations[-1]
    assert invocation["text"] == "texto imagen"
    assert invocation["vision_images"] == [
        {"media_type": "image/png", "data": "encoded-0"},
    ]
    assert service.vision_invocations[-1] == [(b"\x89PNGdatos", "image/png")]
    assert invocation["text_origin"] == "ocr"
    assert result.text_origin == "ocr"
