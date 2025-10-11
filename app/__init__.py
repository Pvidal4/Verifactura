"""Flask application factory."""
from __future__ import annotations

from flask import Flask

from app.config import Config
from app.routes.extract import extract_bp


def create_app(config: Config | None = None) -> Flask:
    app = Flask(__name__)
    config = config or Config()
    app.config["APP_CONFIG"] = config
    app.register_blueprint(extract_bp, url_prefix="/api/v1")
    return app
