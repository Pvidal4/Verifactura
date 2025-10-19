from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import Config


templates = Jinja2Templates(
    directory=str(Path(__file__).resolve().parent.parent / "templates")
)

router = APIRouter(include_in_schema=False)


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    config: Config = getattr(request.app.state, "config", Config())
    api_model = config.OPENAI_MODEL or "gpt-5-mini"
    local_candidates: list[str] = []
    for candidate in (
        config.LOCAL_LLM_MODEL_PATH,
        config.LOCAL_LLM_MODEL_ID,
        "models/gpt-oss-20b",
    ):
        if candidate and candidate not in local_candidates:
            local_candidates.append(candidate)
    local_model = local_candidates[0] if local_candidates else "gpt-oss-20b"
    llm_defaults = {
        "api": {
            "model": api_model,
            "options": [api_model],
        },
        "local": {
            "model": local_model,
            "options": local_candidates,
        },
    }
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "llm_defaults": llm_defaults,
        },
    )
