"""Pruebas de alto nivel para los endpoints de predicción y extracción."""
from __future__ import annotations

import asyncio
from collections import OrderedDict

import pytest
from fastapi import HTTPException

from app.routes.extract import TextExtractionRequest, extract_from_file_endpoint, extract_from_text_endpoint
from app.routes.predictions import PredictionRequest, create_prediction
from app.services.extraction_service import ExtractionResult
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
