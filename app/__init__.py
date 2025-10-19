from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import Config
from app.routes.extract import router as extract_router
from app.routes.ui import router as ui_router


def create_app(config: Optional[Config] = None) -> FastAPI:
    app = FastAPI(title="Verifactura Extraction API")
    app.state.config = config or Config()
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.include_router(ui_router)
    app.include_router(extract_router, prefix="/api/v1")
    return app
