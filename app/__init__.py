"""Inicialización del paquete principal de la aplicación FastAPI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import Config
from app.routes.extract import router as extract_router
from app.routes.predictions import router as predictions_router
from app.routes.ui import router as ui_router


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Construye la instancia de :class:`FastAPI` con rutas y configuración."""

    app = FastAPI(title="Verifactura Extraction API")
    # Almacenar la configuración en el estado permite accederla desde los routers
    app.state.config = config or Config()
    static_dir = Path(__file__).resolve().parent / "static"
    # Se publica el directorio de archivos estáticos para la interfaz web
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    # Registro de rutas de la interfaz gráfica y de los endpoints API
    app.include_router(ui_router)
    app.include_router(extract_router, prefix="/api/v1")
    app.include_router(predictions_router, prefix="/api/v1")
    return app
