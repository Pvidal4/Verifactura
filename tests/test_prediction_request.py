"""Tests covering the prediction request parsing helpers."""
from __future__ import annotations

import math

import pytest

from app.routes.predictions import PredictionRequest, RetrainResponse


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({
            "marca": " chevrolet ",
            "tipo": "sedan",
            "clase": " automovil",
            "capacidad": "4",
            "combustible": "gasolina",
            "ruedas": "4",
            "total": "17.900,50",
        }, {
            "marca": "CHEVROLET",
            "tipo": "SEDAN",
            "clase": "AUTOMOVIL",
            "capacidad": 4,
            "combustible": "GASOLINA",
            "ruedas": 4,
            "total": 17900.50,
        }),
        ({
            "marca": "TOYOTA",
            "tipo": "SUV",
            "clase": "CAMIONETA",
            "capacidad": 5.2,
            "combustible": "Diesel",
            "ruedas": 4.49,
            "total": 25000,
        }, {
            "marca": "TOYOTA",
            "tipo": "SUV",
            "clase": "CAMIONETA",
            "capacidad": 5,
            "combustible": "DIESEL",
            "ruedas": 4,
            "total": 25000.0,
        }),
    ],
)
def test_prediction_request_normalization(raw, expected):
    """PredictionRequest should normalise casing and numeric values."""
    request = PredictionRequest(**raw)
    features = request.to_features()
    assert features == expected


@pytest.mark.parametrize(
    "payload",
    [
        {
            "marca": "",
            "tipo": "SEDAN",
            "clase": "AUTOMOVIL",
            "capacidad": 4,
            "combustible": "GASOLINA",
            "ruedas": 4,
            "total": 17900,
        },
        {
            "marca": "FORD",
            "tipo": "",
            "clase": "AUTOMOVIL",
            "capacidad": 4,
            "combustible": "GASOLINA",
            "ruedas": 4,
            "total": 17900,
        },
        {
            "marca": "FORD",
            "tipo": "SEDAN",
            "clase": "AUTOMOVIL",
            "capacidad": -1,
            "combustible": "GASOLINA",
            "ruedas": 4,
            "total": 17900,
        },
        {
            "marca": "FORD",
            "tipo": "SEDAN",
            "clase": "AUTOMOVIL",
            "capacidad": 4,
            "combustible": "GASOLINA",
            "ruedas": -1,
            "total": 17900,
        },
        {
            "marca": "FORD",
            "tipo": "SEDAN",
            "clase": "AUTOMOVIL",
            "capacidad": 4,
            "combustible": "GASOLINA",
            "ruedas": 4,
            "total": -1,
        },
    ],
)
def test_prediction_request_rejects_invalid_payload(payload):
    with pytest.raises(ValueError):
        PredictionRequest(**payload)


def test_prediction_request_rejects_non_numeric_values():
    payload = {
        "marca": "FORD",
        "tipo": "SEDAN",
        "clase": "AUTOMOVIL",
        "capacidad": "dos",
        "combustible": "GASOLINA",
        "ruedas": 4,
        "total": 17900,
    }
    with pytest.raises(ValueError):
        PredictionRequest(**payload)


def test_prediction_request_handles_thousand_separators():
    request = PredictionRequest(
        marca="FORD",
        tipo="SEDAN",
        clase="AUTOMOVIL",
        capacidad="5",
        combustible="gasolina",
        ruedas="4",
        total="17 900,75",
    )
    assert math.isclose(request.total, 17900.75)


def test_retrain_response_serialization():
    payload = RetrainResponse(
        mensaje="Modelo actualizado.",
        modelo="verifactura_rf_model.pkl",
        clases=["Familiar", "Comercial"],
        muestras_entrenamiento=400,
        muestras_validacion=100,
        matriz_confusion=[[80, 5], [3, 12]],
        reporte={
            "Familiar": {"precision": 0.9, "recall": 0.85, "f1-score": 0.87},
            "Comercial": {"precision": 0.75, "recall": 0.8, "f1-score": 0.77},
            "accuracy": {"score": 0.83},
        },
        servicio_recargado=True,
    )
    assert payload.servicio_recargado is True
    assert payload.matriz_confusion[0][0] == 80
