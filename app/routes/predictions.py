from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, conint, confloat, validator

from app.config import Config
from app.services.prediction_service import PredictionService

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
    ordered = sorted(
        result.probabilities.items(), key=lambda item: item[1], reverse=True
    )
    probabilities = [
        PredictionProbability(clase=label, probabilidad=prob) for label, prob in ordered
    ]
    return PredictionResponse(
        categoria_predicha=result.predicted_class,
        probabilidades=probabilities,
        valores_entrada=features,
    )
