"""FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI

from app.config import Config
from app.routes.extract import router as extract_router


def create_app(config: Config | None = None) -> FastAPI:
    app = FastAPI(title="Verifactura Extraction API")
    app.state.config = config or Config()
    app.include_router(extract_router, prefix="/api/v1")
    return app
