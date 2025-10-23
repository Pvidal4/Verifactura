"""Ejemplo simple de cómo consumir el modelo Random Forest desde un script."""

import joblib
import pandas as pd

# Se carga el modelo previamente entrenado desde disco
model_path = "verifactura_rf_model.pkl"
model = joblib.load(model_path)

# Factura de ejemplo con los campos necesarios para realizar la predicción
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

# Ejecución de la predicción y obtención del vector de probabilidades
prediccion = model.predict(nueva_factura)[0]
probabilidades = model.predict_proba(nueva_factura)[0]

print(f"Categoría predicha: {prediccion}")
print("\nProbabilidades por clase:")
for clase, prob in zip(model.classes_, probabilidades):
    print(f"{clase}: {prob:.2f}")
