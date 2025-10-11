"""Flask entrypoint for the Verifactura extraction API."""
from __future__ import annotations

import logging

from flask import Flask

from app import create_app
from app.config import Config


logging.basicConfig(level=logging.INFO)


def _initialise_app() -> Flask:
    config = Config()
    app = create_app(config)
    logging.info("Flask application initialised with model %s", config.OPENAI_MODEL)
    if config.azure_configured:
        logging.info("Azure OCR endpoint configured: %s", config.AZURE_ENDPOINT)
    else:
        logging.warning("Azure OCR is not configured. Image-only documents will be rejected.")
    return app


app = _initialise_app()


if __name__ == "__main__":  # pragma: no cover - manual execution
    app.run(host="0.0.0.0", port=8000, debug=False)
