# Guía para ejecutar las pruebas automatizadas

Esta guía resume los pasos necesarios para preparar el entorno y ejecutar la batería
de pruebas incluidas en el proyecto. Sigue estas indicaciones si necesitas validar
un cambio o reproducir los escenarios que cubren los endpoints de extracción y
predicción.

## 1. Preparar el entorno
1. Asegúrate de contar con **Python 3.10 o 3.11** y `pip` actualizado.
2. Crea y activa un entorno virtual para aislar las dependencias del proyecto:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\\Scripts\\activate
   ```
3. Instala las dependencias declaradas en `requirements.txt`:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Este conjunto incluye bibliotecas clave como `fastapi`, `pytest`, `openai`,
   `PyPDF2` y `transformers`. Si vas a ejecutar modelos locales con aceleración
   GPU, instala la variante CUDA de PyTorch tal como se indica en la guía de
   instalación.

> Nota: Las pruebas incorporan stubs ligeros para servicios externos (FastAPI,
> Azure Form Recognizer, OpenAI, entre otros), por lo que no necesitas contar con
> esas dependencias reales ni con claves de API para ejecutar la suite.

## 2. Ejecutar la suite completa
Con el entorno virtual activo en la raíz del repositorio, lanza todas las pruebas con:
```bash
pytest
```
El archivo `tests/conftest.py` se ocupa de preparar los stubs y de añadir la raíz
`app/` al `PYTHONPATH`, por lo que basta con ejecutar el comando anterior desde la
carpeta del proyecto o desde cualquier subdirectorio.

## 3. Ejecutar subconjuntos o pruebas individuales
Puedes filtrar por archivo, clase o nombre de función para iterar más rápido. Algunos
ejemplos útiles:

- Ejecutar solo las pruebas de los endpoints:
  ```bash
  pytest tests/test_api_endpoints.py
  ```
- Ejecutar una prueba concreta del flujo de predicción:
  ```bash
  pytest tests/test_api_endpoints.py::test_create_prediction_endpoint_returns_payload
  ```
- Mostrar los nombres de las pruebas a medida que se ejecutan:
  ```bash
  pytest -v
  ```

## 4. Comandos opcionales
Si necesitas medir cobertura o depurar fallos específicos, puedes usar:
```bash
pytest --maxfail=1 --disable-warnings -q
pytest --cov=app --cov-report=term-missing
```
Recuerda instalar previamente los extras necesarios (`pytest-cov`, por ejemplo)
si requieres estos comandos adicionales.

## 5. Interpretar resultados
- **Éxito:** verás un resumen `collected N items` seguido de `N passed`.
- **Fallo:** `pytest` mostrará la traza y el motivo del error. Utiliza el parámetro
  `-x` para detenerte en la primera falla cuando estés depurando.

Con estos pasos podrás verificar la funcionalidad principal del proyecto antes de
publicar cambios o preparar despliegues.
