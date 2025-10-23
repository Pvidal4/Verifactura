# Guía de instalación de Verifactura

## Requisitos del sistema
- **Sistema operativo:** Windows 10/11, macOS 13+ o cualquier distribución Linux actual.
- **Python:** 3.10 o 3.11 con `pip` actualizado.
- **CPU:** Procesador de 4 núcleos (se recomienda soporte para instrucciones AVX).
- **Memoria RAM:** 8 GB mínimo (16 GB recomendados para OCR y modelos locales).
- **GPU (opcional):** NVIDIA con soporte CUDA 12.1 para acelerar modelos locales con PyTorch.
- **Credenciales externas:**
  - Clave de API de OpenAI para el flujo de extracción vía API.
  - Endpoint y clave de Azure Form Recognizer para OCR.

## Instrucciones paso a paso para instalar
1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/<tu-organizacion>/Verifactura.git
   cd Verifactura
   ```
2. **Crear y activar un entorno virtual**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows usa: .venv\\Scripts\\activate
   ```
3. **Instalar dependencias principales**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Las bibliotecas críticas como `fastapi` 0.110.0, `openai` 1.17.0, `PyPDF2` 3.0.1, `azure-ai-formrecognizer` 3.3.0 y `transformers` 4.39.3 están listadas en `requirements.txt`.
4. **(Opcional) Acelerar modelos locales con GPU**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   Este comando instala las variantes de PyTorch compatibles con CUDA 12.1. Si no cuentas con GPU NVIDIA, conserva la instalación CPU incluida en `requirements.txt`.
5. **Configurar variables de entorno**
   - Copia `.env.example` a `.env` (si existe) o crea un archivo `.env` con las claves siguientes:
     ```env
     OPENAI_API_KEY=tu_clave
     OPENAI_MODEL=gpt-5-mini
     AZURE_FORM_RECOGNIZER_ENDPOINT=https://<tu-endpoint>.cognitiveservices.azure.com/
     AZURE_FORM_RECOGNIZER_KEY=tu_clave_azure
     ```
   - Ajusta `LOCAL_LLM_MODEL_PATH` o `LOCAL_LLM_MODEL_ID` si cuentas con pesos locales.

## Comandos para ejecutar el proyecto
- **API en modo desarrollo**
  ```bash
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
  ```
- **Pruebas automatizadas**
  ```bash
  pytest
  ```
- **Reentrenar el modelo Random Forest**
  ```bash
  python -m train.random_forest
  ```
  (Asegúrate de que `train/data/verifactura_dataset.csv` contenga tu dataset actualizado.)

## Ejemplos de uso
- **Extracción desde texto plano**
  ```bash
  curl -X POST http://localhost:8000/api/v1/extract/text \
       -H "Content-Type: application/json" \
       -d '{
             "text": "Factura Nº 123 emitida el 15/03/2024...",
             "llm_provider": "api",
             "llm_model": "gpt-5-mini"
           }'
  ```
- **Extracción desde archivo PDF**
  ```bash
  curl -X POST http://localhost:8000/api/v1/extract/file \
       -F "file=@factura.pdf" \
       -F "force_ocr=false"
  ```
- **Clasificación de categoría vehicular**
  ```bash
  curl -X POST http://localhost:8000/api/v1/predictions \
       -H "Content-Type: application/json" \
       -d '{
             "marca": "TOYOTA",
             "tipo": "SUV",
             "clase": "CAMIONETA",
             "capacidad": 5,
             "combustible": "GASOLINA",
             "ruedas": 4,
             "total": 24990.00
           }'
  ```
