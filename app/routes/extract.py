"""HTTP endpoint definitions for the extraction API."""
from __future__ import annotations

from http import HTTPStatus

from flask import Blueprint, Response, current_app, jsonify, request

from app.config import Config
from app.services.extraction_service import ExtractionService

extract_bp = Blueprint("extract", __name__)


def _get_service() -> ExtractionService:
    if "EXTRACTION_SERVICE" not in current_app.config:
        config: Config = current_app.config["APP_CONFIG"]
        current_app.config["EXTRACTION_SERVICE"] = ExtractionService(config)
    return current_app.config["EXTRACTION_SERVICE"]


@extract_bp.route("/extract", methods=["POST"])
def extract_endpoint() -> Response:
    service = _get_service()
    if request.content_type and request.content_type.startswith("application/json"):
        payload = request.get_json(silent=True) or {}
        text = payload.get("text")
        if not text:
            return jsonify({"error": "Missing 'text' in request body"}), HTTPStatus.BAD_REQUEST
        result = service.extract_from_text(text)
        return jsonify(result)

    if "file" not in request.files and not request.form.get("text"):
        return (
            jsonify({"error": "Provide a 'text' field or upload a file via multipart/form-data."}),
            HTTPStatus.BAD_REQUEST,
        )

    if "file" in request.files:
        uploaded = request.files["file"]
        filename = uploaded.filename or "uploaded"
        data = uploaded.read()
        try:
            result = service.extract_from_file(filename, data, uploaded.mimetype)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), HTTPStatus.BAD_REQUEST
        return jsonify(result)

    text = request.form.get("text", "")
    if not text:
        return jsonify({"error": "Empty 'text' value."}), HTTPStatus.BAD_REQUEST
    result = service.extract_from_text(text)
    return jsonify(result)
