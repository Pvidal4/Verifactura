"""Punto de entrada de FastAPI para la API de extracción de Verifactura."""
from __future__ import annotations

import logging

from fastapi import FastAPI
import uvicorn

from app import create_app
from app.config import Config


logging.basicConfig(level=logging.INFO)


def _initialise_app() -> FastAPI:
    """Crea y configura la aplicación principal con los servicios necesarios."""
    config = Config()
    app = create_app(config)
    # Se registran logs informativos sobre el modelo y los servicios disponibles
    logging.info("FastAPI application initialised with model %s", config.OPENAI_MODEL)
    if config.azure_configured:
        logging.info("Azure OCR endpoint configured: %s", config.AZURE_ENDPOINT)
    else:
        logging.warning("Azure OCR is not configured. Image-only documents will be rejected.")
    return app


app = _initialise_app()


if __name__ == "__main__":  # pragma: no cover - manual execution
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
