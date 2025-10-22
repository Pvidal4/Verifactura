"""Tests for the prediction service helpers."""
from __future__ import annotations

from types import SimpleNamespace

from app.services.prediction_service import PredictionService


class _AmbiguousSequence(list):
    """List-like sequence whose truthiness raises ValueError."""

    def __bool__(self) -> bool:  # pragma: no cover - defensive branch
        raise ValueError("ambiguous truth value")


def test_resolve_feature_columns_handles_ambiguous_sequences():
    """feature_names_in_ may be numpy arrays with ambiguous truthiness."""

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
