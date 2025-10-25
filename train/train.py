"""Rutinas para entrenar el modelo Random Forest utilizado por Verifactura.

Este módulo replica la lógica del script de referencia proporcionado por el
usuario y la empaqueta como una función reutilizable. Coloca el archivo CSV
utilizado para el entrenamiento en la ruta indicada por
``DEFAULT_DATASET_PATH``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence

DEFAULT_DATASET_PATH = Path(
    "train/data/verifactura_dataset.csv"
)


@dataclass(frozen=True)
class RandomForestTrainingResult:
    """Información relevante producida tras reentrenar el modelo."""

    model_path: Path
    classes: List[str]
    training_samples: int
    validation_samples: int
    classification_report: Mapping[str, Mapping[str, float]]
    confusion_matrix: List[List[int]]


class _TrainingDependencies(dict):
    """Mapa de dependencias necesarias para el entrenamiento."""

    pd: object  # pandas
    joblib: object
    ColumnTransformer: object
    OneHotEncoder: object
    StandardScaler: object
    Pipeline: object
    RandomForestClassifier: object
    train_test_split: object
    classification_report: object
    confusion_matrix: object


def _ensure_dependencies() -> _TrainingDependencies:
    """Importa las dependencias de entrenamiento con mensajes amigables."""

    missing: List[str] = []
    modules: MutableMapping[str, object] = {}

    try:
        import pandas as pd  # type: ignore

        modules["pd"] = pd
    except ModuleNotFoundError:
        missing.append("pandas")

    try:
        import joblib  # type: ignore

        modules["joblib"] = joblib
    except ModuleNotFoundError:
        missing.append("joblib")

    try:
        from sklearn.compose import ColumnTransformer  # type: ignore
        from sklearn.ensemble import RandomForestClassifier  # type: ignore
        from sklearn.metrics import classification_report, confusion_matrix  # type: ignore
        from sklearn.model_selection import train_test_split  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore

        modules.update(
            {
                "ColumnTransformer": ColumnTransformer,
                "RandomForestClassifier": RandomForestClassifier,
                "classification_report": classification_report,
                "confusion_matrix": confusion_matrix,
                "train_test_split": train_test_split,
                "Pipeline": Pipeline,
                "OneHotEncoder": OneHotEncoder,
                "StandardScaler": StandardScaler,
            }
        )
    except ModuleNotFoundError:
        missing.append("scikit-learn")

    if missing:
        raise RuntimeError(
            "Faltan dependencias para entrenar el modelo: " + ", ".join(sorted(missing))
        )

    return _TrainingDependencies(modules)


def _coerce_float_map(data: Mapping[str, object]) -> Dict[str, float]:
    """Convierte recursivamente los valores numéricos a float nativo."""

    result: Dict[str, float] = {}
    for key, value in data.items():
        try:
            result[key] = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            # Se ignoran las claves no numéricas
            continue
    return result


def _normalize_report(report: Mapping[str, object]) -> Dict[str, Mapping[str, float]]:
    """Normaliza el `classification_report` para hacerlo serializable."""

    normalized: Dict[str, Mapping[str, float]] = {}
    for key, value in report.items():
        if isinstance(value, Mapping):
            normalized[key] = _coerce_float_map(value)
        else:
            try:
                normalized[key] = {"score": float(value)}  # type: ignore[arg-type]
            except (TypeError, ValueError):
                # En último caso se omite la entrada no numérica
                continue
    return normalized


def train_random_forest(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    model_path: Path | str = "verifactura_rf_model.pkl",
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> RandomForestTrainingResult:
    """Entrena y guarda el modelo Random Forest con el dataset indicado."""

    deps = _ensure_dependencies()
    pd = deps["pd"]
    joblib = deps["joblib"]
    ColumnTransformer = deps["ColumnTransformer"]
    OneHotEncoder = deps["OneHotEncoder"]
    StandardScaler = deps["StandardScaler"]
    Pipeline = deps["Pipeline"]
    RandomForestClassifier = deps["RandomForestClassifier"]
    train_test_split = deps["train_test_split"]
    classification_report = deps["classification_report"]
    confusion_matrix = deps["confusion_matrix"]

    dataset = Path(dataset_path)
    if not dataset.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de entrenamiento en {dataset!s}."
        )

    model_destination = Path(model_path)
    if not model_destination.parent.exists():
        model_destination.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(dataset)

    required_columns = [
        "marca",
        "tipo",
        "clase",
        "capacidad",
        "combustible",
        "ruedas",
        "total",
        "categoria",
    ]
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "El dataset de entrenamiento no contiene las columnas necesarias: "
            + ", ".join(missing_columns)
        )

    feature_columns = [
        "marca",
        "tipo",
        "clase",
        "capacidad",
        "combustible",
        "ruedas",
        "total",
    ]
    target_column = "categoria"

    X = frame[feature_columns]
    y = frame[target_column]

    categorical_features = ["marca", "tipo", "clase", "combustible"]
    numerical_features = ["capacidad", "ruedas", "total"]

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numerical_features),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=15,
        max_features=0.6,
        min_samples_split=5,
        min_samples_leaf=3,
        max_samples=0.8,
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", rf),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    report_raw = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    report = _normalize_report(report_raw)
    confusion = confusion_matrix(y_test, y_pred)

    joblib.dump(pipeline, model_destination)

    classes: Sequence[str] = getattr(pipeline, "classes_", [])
    class_list = [str(label) for label in classes]

    return RandomForestTrainingResult(
        model_path=model_destination,
        classes=class_list,
        training_samples=int(len(X_train)),
        validation_samples=int(len(X_test)),
        classification_report=report,
        confusion_matrix=[list(map(int, row)) for row in confusion.tolist()],
    )
