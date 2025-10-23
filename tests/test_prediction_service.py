"""Pruebas enfocadas en la lógica auxiliar del servicio de predicciones."""
from __future__ import annotations

from types import SimpleNamespace

from app.services.prediction_service import PredictionService


class _AmbiguousSequence(list):
    """Secuencia que emula estructuras con verdad ambigua como arrays de NumPy."""

    def __bool__(self) -> bool:  # pragma: no cover - defensive branch
        raise ValueError("ambiguous truth value")


def test_resolve_feature_columns_handles_ambiguous_sequences():
    """Debe manejar `feature_names_in_` que no permiten evaluación booleana directa."""

    service = object.__new__(PredictionService)
    service._model = SimpleNamespace(
        feature_names_in_=_AmbiguousSequence([  # type: ignore[attr-defined]
            "marca",
            "tipo",
            "clase",
            "capacidad",
            "combustible",
            "ruedas",
            "total",
        ])
    )

    columns = PredictionService._resolve_feature_columns(service)

    assert columns == [
        "marca",
        "tipo",
        "clase",
        "capacidad",
        "combustible",
        "ruedas",
        "total",
    ]
