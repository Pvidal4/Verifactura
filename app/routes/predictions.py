from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, conint, confloat, validator

from app.config import Config
from app.services.prediction_service import PredictionService
from app.services.training_service import TrainingService

router = APIRouter(tags=["Predicciones"])


class PredictionRequest(BaseModel):
    marca: str = Field(..., description="Marca del vehículo en mayúsculas.")
    tipo: str = Field(..., description="Tipo del vehículo (por ejemplo, SEDAN).")
    clase: str = Field(..., description="Clase del vehículo.")
    capacidad: conint(ge=0) = Field(..., description="Capacidad de pasajeros.")
    combustible: str = Field(..., description="Tipo de combustible.")
    ruedas: conint(ge=0) = Field(..., description="Número total de ruedas.")
    total: confloat(ge=0) = Field(..., description="Monto total del comprobante.")

    @validator("marca", "tipo", "clase", "combustible", pre=True)
    def _normalize_text(cls, value: Any) -> str:
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                raise ValueError("El valor no puede estar vacío.")
            return cleaned.upper()
        raise ValueError("El valor debe ser una cadena de texto.")

    @validator("capacidad", "ruedas", pre=True)
    def _coerce_int(cls, value: Any) -> int:
        number = PredictionRequest._parse_number(value)
        if number < 0:
            raise ValueError("El valor debe ser mayor o igual a cero.")
        return int(round(number))

    @validator("total", pre=True)
    def _coerce_float(cls, value: Any) -> float:
        number = PredictionRequest._parse_number(value)
        if number < 0:
            raise ValueError("El valor debe ser mayor o igual a cero.")
        return float(number)

    @staticmethod
    def _parse_number(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                raise ValueError("El valor no puede estar vacío.")
            normalized = cleaned.replace(" ", "")
            filtered = "".join(ch for ch in normalized if ch.isdigit() or ch in {"-", ",", "."})
            if not filtered:
                raise ValueError("El valor no contiene dígitos.")
            last_comma = filtered.rfind(",")
            last_dot = filtered.rfind(".")
            if last_comma > last_dot:
                candidate = filtered.replace(".", "").replace(",", ".")
            else:
                candidate = filtered.replace(",", "")
            try:
                return float(candidate)
            except ValueError as exc:  # pragma: no cover - validaciones defensivas
                raise ValueError("No se pudo interpretar el número proporcionado.") from exc
        raise ValueError("El valor debe ser numérico.")

    def to_features(self) -> Dict[str, object]:
        return {
            "marca": self.marca,
            "tipo": self.tipo,
            "clase": self.clase,
            "capacidad": int(self.capacidad),
            "combustible": self.combustible,
            "ruedas": int(self.ruedas),
            "total": float(self.total),
        }


class PredictionProbability(BaseModel):
    clase: str = Field(..., description="Nombre de la categoría evaluada.")
    probabilidad: float = Field(..., ge=0, le=1, description="Probabilidad asociada a la categoría.")


class PredictionResponse(BaseModel):
    categoria_predicha: str = Field(..., description="Categoría predicha por el modelo.")
    probabilidades: list[PredictionProbability] = Field(
        ..., description="Listado de probabilidades por categoría."
    )
    valores_entrada: Dict[str, object] = Field(
        ..., description="Valores utilizados para generar la predicción."
    )


class RetrainResponse(BaseModel):
    mensaje: str = Field(
        ..., description="Descripción del resultado del proceso de reentrenamiento."
    )
    modelo: str = Field(..., description="Ruta en disco del modelo entrenado.")
    clases: list[str] = Field(
        ..., description="Listado de clases conocidas por el modelo entrenado."
    )
    muestras_entrenamiento: conint(ge=0) = Field(
        ..., description="Cantidad de muestras utilizadas para el entrenamiento."
    )
    muestras_validacion: conint(ge=0) = Field(
        ..., description="Cantidad de muestras utilizadas para la validación."
    )
    matriz_confusion: list[list[int]] = Field(
        ..., description="Matriz de confusión calculada en el conjunto de validación."
    )
    reporte: Dict[str, Dict[str, float]] = Field(
        ..., description="Reporte de métricas por clase."
    )
    servicio_recargado: bool = Field(
        ..., description="Indica si el servicio de predicciones se recargó correctamente."
    )


def _get_prediction_service(request: Request) -> PredictionService:
    service: Optional[PredictionService] = getattr(
        request.app.state, "prediction_service", None
    )
    if service is None:
        config: Config = getattr(request.app.state, "config", Config())
        try:
            service = PredictionService(config.RF_MODEL_PATH)
        except FileNotFoundError as exc:  # pragma: no cover - dependiente del entorno
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No se encontró el modelo de predicción configurado.",
            ) from exc
        except RuntimeError as exc:  # pragma: no cover - dependiente del entorno
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        request.app.state.prediction_service = service
    return service


@router.post(
    "/predictions",
    response_model=PredictionResponse,
    summary="Obtener la categoría estimada por el modelo de clasificación",
)
async def create_prediction(
    payload: PredictionRequest, service: PredictionService = Depends(_get_prediction_service)
) -> PredictionResponse:
    features = payload.to_features()
    try:
        result = service.predict(features)
    except Exception as exc:  # pragma: no cover - errores en la capa del modelo
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo calcular la predicción solicitada.",
        ) from exc
    ordered = list(result.probabilities.items())
    probabilities = [
        PredictionProbability(clase=label, probabilidad=prob) for label, prob in ordered
    ]
    return PredictionResponse(
        categoria_predicha=result.predicted_class,
        probabilidades=probabilities,
        valores_entrada=features,
    )


@router.post(
    "/predictions/retrain",
    response_model=RetrainResponse,
    summary="Reentrenar el modelo Random Forest con el dataset configurado",
)
async def retrain_prediction_model(request: Request) -> RetrainResponse:
    config: Config = getattr(request.app.state, "config", Config())
    dataset_path = Path(config.RF_TRAINING_DATA_PATH).expanduser()
    model_path = Path(config.RF_MODEL_PATH).expanduser()
    training_service = TrainingService(dataset_path=dataset_path, model_path=model_path)
    try:
        result = training_service.retrain_random_forest()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "No se encontró el archivo CSV configurado para el entrenamiento "
                f"({dataset_path!s})."
            ),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    servicio_recargado = False
    try:
        service: Optional[PredictionService] = getattr(
            request.app.state, "prediction_service", None
        )
        if service is None:
            service = PredictionService(model_path)
            request.app.state.prediction_service = service
        else:
            service.reload()
        servicio_recargado = True
    except Exception:
        request.app.state.prediction_service = None

    mensaje = "Modelo Random Forest reentrenado correctamente."
    if not servicio_recargado:
        mensaje += " Vuelve a ejecutar una predicción para cargar el nuevo modelo."

    return RetrainResponse(
        mensaje=mensaje,
        modelo=str(result.model_path),
        clases=result.classes,
        muestras_entrenamiento=result.training_samples,
        muestras_validacion=result.validation_samples,
        matriz_confusion=result.confusion_matrix,
        reporte={key: dict(value) for key, value in result.classification_report.items()},
        servicio_recargado=servicio_recargado,
    )
