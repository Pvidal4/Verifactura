"""HTTP endpoint definitions for the extraction API."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from app.config import Config
from app.services.extraction_service import ExtractionService

router = APIRouter()


def _get_service(request: Request) -> ExtractionService:
    service: Optional[ExtractionService] = getattr(
        request.app.state, "extraction_service", None
    )
    if service is None:
        config: Config = getattr(request.app.state, "config", Config())
        service = ExtractionService(config)
        request.app.state.extraction_service = service
    return service


def _normalise_content_type(raw: str | None) -> str:
    if not raw:
        return ""
    return raw.split(";", 1)[0].strip().lower()


async def _handle_file_upload(
    service: ExtractionService, uploaded: UploadFile
) -> Dict[str, Any]:
    filename = uploaded.filename or "uploaded"
    data = await uploaded.read()
    try:
        return service.extract_from_file(filename, data, uploaded.content_type)
    except RuntimeError as exc:  # pragma: no cover - defensive branch
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/extract")
async def extract_endpoint(
    request: Request, service: ExtractionService = Depends(_get_service)
) -> Dict[str, Any]:
    content_type = _normalise_content_type(request.headers.get("content-type"))

    if content_type == "application/json":
        try:
            payload = await request.json()
        except ValueError as exc:  # pragma: no cover - FastAPI validates JSON
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
        text = (payload or {}).get("text") if isinstance(payload, dict) else None
        if not text:
            raise HTTPException(status_code=400, detail="Missing 'text' in request body")
        return service.extract_from_text(str(text))

    if content_type in {"multipart/form-data", "application/x-www-form-urlencoded"}:
        form = await request.form()
        uploaded = form.get("file")
        if isinstance(uploaded, list):
            uploaded = uploaded[0]
        if isinstance(uploaded, UploadFile):
            return await _handle_file_upload(service, uploaded)

        text = form.get("text")
        if not text:
            raise HTTPException(
                status_code=400,
                detail="Provide a 'text' field or upload a file via multipart/form-data.",
            )
        if isinstance(text, UploadFile):
            return await _handle_file_upload(service, text)
        text_value = str(text).strip()
        if not text_value:
            raise HTTPException(status_code=400, detail="Empty 'text' value.")
        return service.extract_from_text(text_value)

    if content_type == "text/plain":
        text = (await request.body()).decode("utf-8", errors="replace").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty 'text' value.")
        return service.extract_from_text(text)

    raise HTTPException(
        status_code=400,
        detail="Provide JSON with 'text', form data with 'text', or upload a file.",
    )
