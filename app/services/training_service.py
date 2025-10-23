"""Servicios de conveniencia para reentrenar el modelo de clasificación."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from train.random_forest import RandomForestTrainingResult, train_random_forest


@dataclass
class TrainingService:
    """Gestiona el ciclo de reentrenamiento del modelo Random Forest."""

    dataset_path: Path
    model_path: Path

    def retrain_random_forest(self) -> RandomForestTrainingResult:
        """Ejecuta el pipeline de entrenamiento y devuelve sus métricas."""

        return train_random_forest(self.dataset_path, self.model_path)
