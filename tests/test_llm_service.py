"""Pruebas unitarias específicas para la integración con OpenAI."""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

python_multipart = ModuleType("python_multipart")
python_multipart.__version__ = "0.0.1"
sys.modules.setdefault("python_multipart", python_multipart)

multipart_module = ModuleType("multipart")
multipart_module.__version__ = "0.0.1"
sys.modules.setdefault("multipart", multipart_module)

multipart_submodule = ModuleType("multipart.multipart")

def _parse_options_header(value: str):  # pragma: no cover - utilitario para stub
    return value, {}

multipart_submodule.parse_options_header = _parse_options_header
sys.modules.setdefault("multipart.multipart", multipart_submodule)

from app.services.llm_service import OpenAILLMService


class _RecordingClient:
    """Cliente simulado que captura los argumentos de cada invocación."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._record_call)
        )

    def _record_call(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="{}",
                    )
                )
            ]
        )


def test_openai_service_omits_reasoning_effort_when_disabled(monkeypatch):
    """No debe enviar `reasoning_effort` cuando el usuario lo desactiva."""

    created_clients: list[_RecordingClient] = []

    class _DummyConfig:
        OPENAI_API_KEY = None
        OPENAI_MODEL = "gpt-4o-mini"
        JSON_MODE_SCHEMA_NAME = "factura_vehicular"

    def _fake_openai_client(*args, **kwargs):
        client = _RecordingClient()
        created_clients.append(client)
        return client

    monkeypatch.setattr("app.services.llm_service.OpenAI", _fake_openai_client)

    service = OpenAILLMService(_DummyConfig())

    service.extract(
        "texto de prueba",
        reasoning_effort="none",
        api_key="test-key",
    )

    assert created_clients, "Se debe haber creado un cliente de OpenAI"
    assert created_clients[-1].calls, "La llamada al cliente debe registrarse"
    recorded_call = created_clients[-1].calls[-1]

    assert "reasoning_effort" not in recorded_call
