"""Ejemplo simple de cómo consumir el modelo Random Forest desde un script."""


def main() -> None:
    """Carga el modelo entrenado y muestra una predicción de ejemplo."""

    import joblib
    import pandas as pd

    model_path = "verifactura_rf_model.pkl"
    model = joblib.load(model_path)

    nueva_factura = pd.DataFrame(
        [
            {
                "marca": "CHEVROLET",
                "tipo": "SEDAN",
                "clase": "AUTOMOVIL",
                "capacidad": 5,
                "combustible": "GASOLINA",
                "ruedas": 4,
                "total": 17900,
            }
        ]
    )

    prediccion = model.predict(nueva_factura)[0]
    probabilidades = model.predict_proba(nueva_factura)[0]

    print(f"Categoría predicha: {prediccion}")
    print("\nProbabilidades por clase:")
    for clase, prob in zip(model.classes_, probabilidades):
        print(f"{clase}: {prob:.2f}")


if __name__ == "__main__":  # pragma: no cover - ejecución manual
    main()
