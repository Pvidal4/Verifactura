from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import joblib
import pandas as pd


@dataclass(frozen=True)
class PredictionResult:
    """Structured result returned by the random forest classifier."""

    predicted_class: str
    probabilities: Dict[str, float]


class PredictionService:
    """Thin wrapper around the trained random forest model used for scoring."""

    def __init__(self, model_path: Path | str) -> None:
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo del modelo en {self._model_path!s}."
            )
        self._model = self._load_model()
        self._feature_columns = self._resolve_feature_columns()

    def _load_model(self):  # type: ignore[no-untyped-def]
        try:
            return joblib.load(self._model_path)
        except Exception as exc:  # pragma: no cover - defensive path
            raise RuntimeError("No se pudo cargar el modelo de predicción.") from exc

    def _resolve_feature_columns(self) -> Sequence[str]:
        candidates: Iterable[str] | None = getattr(
            self._model, "feature_names_in_", None
        )
        if candidates:
            return list(candidates)
        # Default to the expected training order if metadata is not present.
        return ["marca", "tipo", "clase", "capacidad", "combustible", "ruedas", "total"]

    def predict(self, features: Mapping[str, object]) -> PredictionResult:
        row = {column: features.get(column) for column in self._feature_columns}
        frame = pd.DataFrame([row], columns=list(self._feature_columns))
        predicted = self._model.predict(frame)[0]
        probabilities: List[float]
        if hasattr(self._model, "predict_proba"):
            probabilities = list(self._model.predict_proba(frame)[0])
        else:  # pragma: no cover - the shipped model implements predict_proba
            probabilities = [0.0 for _ in getattr(self._model, "classes_", [])]
        classes: Sequence[str] = getattr(self._model, "classes_", [])
        probability_map = {
            str(label): float(prob)
            for label, prob in zip(classes, probabilities)
        }
        return PredictionResult(predicted_class=str(predicted), probabilities=probability_map)

