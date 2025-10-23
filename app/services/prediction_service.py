from __future__ import annotations
"""Servicios auxiliares para interactuar con el modelo de clasificación."""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:  # pragma: no cover - dependencias opcionales en tiempo de importación
    import joblib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - dependencias opcionales
    joblib = None  # type: ignore[assignment]

try:  # pragma: no cover - dependencias opcionales en tiempo de importación
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - dependencias opcionales
    pd = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PredictionResult:
    """Resultado estructurado producido por el clasificador Random Forest."""

    predicted_class: str
    probabilities: OrderedDict[str, float]


class PredictionService:
    """Envoltorio ligero alrededor del modelo entrenado para realizar inferencias."""

    def __init__(self, model_path: Path | str) -> None:
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo del modelo en {self._model_path!s}."
            )
        self._model = self._load_model()
        self._feature_columns = self._resolve_feature_columns()

    def _load_model(self):  # type: ignore[no-untyped-def]
        """Carga el modelo desde disco validando la presencia de joblib."""

        if joblib is None:
            raise RuntimeError(
                "joblib no está instalado. Instálalo para utilizar el servicio de predicciones."
            )
        try:
            return joblib.load(self._model_path)
        except Exception as exc:  # pragma: no cover - defensive path
            raise RuntimeError("No se pudo cargar el modelo de predicción.") from exc

    def _resolve_feature_columns(self) -> Sequence[str]:
        """Obtiene el orden esperado de columnas a partir del modelo entrenado."""

        candidates: Iterable[str] | None = getattr(
            self._model, "feature_names_in_", None
        )
        if candidates is not None:
            return list(candidates)
        # Se vuelve al orden utilizado durante el entrenamiento cuando no hay metadatos
        return ["marca", "tipo", "clase", "capacidad", "combustible", "ruedas", "total"]

    def predict(self, features: Mapping[str, object]) -> PredictionResult:
        """Recibe un diccionario de atributos y devuelve la clase estimada."""

        try:
            if pd is None:
                raise RuntimeError(
                    "pandas no está instalado. Instálalo para utilizar el servicio de predicciones."
                )
            ordered_row = {
                column: features[column] for column in self._feature_columns
            }
        except KeyError as exc:  # pragma: no cover - defensive path
            missing = exc.args[0]
            raise ValueError(
                f"Falta el atributo requerido '{missing}' en la solicitud de predicción."
            ) from exc
        frame = pd.DataFrame([ordered_row], columns=list(self._feature_columns))
        predicted = self._model.predict(frame)[0]
        try:
            probability_vector = self._model.predict_proba(frame)[0]
        except AttributeError as exc:  # pragma: no cover - modelos sin predict_proba
            raise RuntimeError(
                "El modelo configurado no expone probabilidades de clase."
            ) from exc
        classes: Sequence[str] = getattr(self._model, "classes_", [])
        probability_map: OrderedDict[str, float] = OrderedDict(
            (str(label), float(prob)) for label, prob in zip(classes, probability_vector)
        )
        return PredictionResult(
            predicted_class=str(predicted), probabilities=probability_map
        )

    def reload(self) -> None:
        """Recarga el modelo desde disco tras un reentrenamiento."""

        self._model = self._load_model()
        self._feature_columns = self._resolve_feature_columns()

