"""Fixtures y stubs para ejecutar las pruebas sin dependencias externas."""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace


def _ensure_fastapi_stub() -> None:
    try:  # pragma: no cover - solo se ejecuta cuando la dependencia real existe
        import fastapi  # type: ignore
        return
    except ModuleNotFoundError:
        pass

    fastapi_module = types.ModuleType("fastapi")

    class HTTPException(Exception):
        """Excepción ligera que imita a :class:`fastapi.HTTPException`."""

        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class Request:  # pragma: no cover - solo usado por tipado
        """Objeto mínimo con estado mutable para tests."""

        def __init__(self) -> None:
            self.app = SimpleNamespace(state=SimpleNamespace())

    class UploadFile:  # pragma: no cover - no se utiliza directamente en los tests
        def __init__(
            self,
            filename: str | None = None,
            content_type: str | None = None,
            data: bytes = b"",
        ) -> None:
            self.filename = filename
            self.content_type = content_type
            self._data = data

        async def read(self) -> bytes:
            return self._data

    class APIRouter:  # pragma: no cover - decoración sin efectos
        def __init__(self, *args, **kwargs) -> None:
            self.routes = []

        def post(self, *args, **kwargs):
            def decorator(func):
                self.routes.append(("POST", args, kwargs, func))
                return func

            return decorator

        def get(self, *args, **kwargs):
            def decorator(func):
                self.routes.append(("GET", args, kwargs, func))
                return func

            return decorator

    class FastAPI:  # pragma: no cover - utilizado solo para tipado
        def __init__(self, *args, **kwargs) -> None:
            self.state = SimpleNamespace()

        def include_router(self, *args, **kwargs) -> None:
            return None

        def mount(self, *args, **kwargs) -> None:
            return None

    def Depends(dependency):  # pragma: no cover - no utilizado en ejecución
        return dependency

    def File(default=..., **kwargs):  # pragma: no cover - valores por defecto
        return default

    def Query(default=..., **kwargs):  # pragma: no cover - valores por defecto
        return default

    fastapi_module.FastAPI = FastAPI
    fastapi_module.APIRouter = APIRouter
    fastapi_module.Depends = Depends
    fastapi_module.File = File
    fastapi_module.Query = Query
    fastapi_module.HTTPException = HTTPException
    fastapi_module.Request = Request
    fastapi_module.UploadFile = UploadFile

    status_module = types.ModuleType("fastapi.status")
    status_module.HTTP_500_INTERNAL_SERVER_ERROR = 500
    status_module.HTTP_400_BAD_REQUEST = 400
    fastapi_module.status = status_module

    responses_module = types.ModuleType("fastapi.responses")

    class HTMLResponse:  # pragma: no cover - solo para tipado
        def __init__(self, content: str, status_code: int = 200) -> None:
            self.body = content
            self.status_code = status_code

    responses_module.HTMLResponse = HTMLResponse

    templating_module = types.ModuleType("fastapi.templating")

    class Jinja2Templates:  # pragma: no cover - utilizado solo por tipado
        def __init__(self, directory: str) -> None:
            self.directory = directory

        def TemplateResponse(self, name: str, context: dict) -> dict:
            return {"template": name, "context": context}

    templating_module.Jinja2Templates = Jinja2Templates

    staticfiles_module = types.ModuleType("fastapi.staticfiles")

    class StaticFiles:  # pragma: no cover - solo informativo
        def __init__(self, directory: str) -> None:
            self.directory = directory

    staticfiles_module.StaticFiles = StaticFiles

    sys.modules.setdefault("fastapi", fastapi_module)
    sys.modules.setdefault("fastapi.status", status_module)
    sys.modules.setdefault("fastapi.responses", responses_module)
    sys.modules.setdefault("fastapi.templating", templating_module)
    sys.modules.setdefault("fastapi.staticfiles", staticfiles_module)


def _ensure_pydantic_stub() -> None:
    try:  # pragma: no cover - solo se ejecuta cuando la dependencia real existe
        import pydantic  # type: ignore
        return
    except ModuleNotFoundError:
        pass

    pydantic_module = types.ModuleType("pydantic")

    class FieldInfo:
        """Contenedor mínimo de metadatos para un campo."""

        def __init__(self, default=..., **metadata) -> None:
            self.default = default
            self.metadata = metadata

    def Field(default=..., **metadata):
        return FieldInfo(default, **metadata)

    def conint(*, ge=None, le=None):  # pragma: no cover - validación en los validadores
        constraint = FieldInfo(None, ge=ge, le=le)
        constraint.type = int
        return constraint

    def confloat(*, ge=None, le=None):  # pragma: no cover - validación en los validadores
        constraint = FieldInfo(None, ge=ge, le=le)
        constraint.type = float
        return constraint

    def validator(*fields, pre: bool = False):
        def decorator(func):
            func.__validator_fields__ = fields
            func.__validator_pre__ = pre
            return func

        return decorator

    class BaseModelMeta(type):
        def __new__(mcls, name, bases, namespace):
            validators = {}  # field -> list[(pre, func)]
            for base in bases:
                for field, funcs in getattr(base, "__validators__", {}).items():
                    validators.setdefault(field, []).extend(funcs)
            for attr_name, value in list(namespace.items()):
                fields = getattr(value, "__validator_fields__", None)
                if fields is None:
                    continue
                pre = getattr(value, "__validator_pre__", False)
                for field in fields:
                    validators.setdefault(field, []).append((pre, value))
            namespace["__validators__"] = validators
            return super().__new__(mcls, name, bases, namespace)

    class BaseModel(metaclass=BaseModelMeta):
        __validators__: dict[str, list[tuple[bool, callable]]] = {}

        def __init__(self, **data):
            annotations = getattr(self, "__annotations__", {})
            values = dict(data)
            for field, annotation in annotations.items():
                default = None
                has_default = False
                field_info = getattr(self.__class__, field, FieldInfo(...))
                if isinstance(field_info, FieldInfo):
                    default = field_info.default
                    has_default = default is not ...
                else:
                    default = field_info
                    has_default = True
                if field in values:
                    value = values.pop(field)
                elif has_default:
                    value = default
                else:
                    raise ValueError(f"El campo '{field}' es obligatorio")
                for pre, func in self.__validators__.get(field, []):
                    if pre:
                        value = func(self.__class__, value)
                for pre, func in self.__validators__.get(field, []):
                    if not pre:
                        value = func(self.__class__, value)
                setattr(self, field, value)
            # Ignorar campos extra, imitando el comportamiento por defecto de Pydantic

        def dict(self) -> dict:
            return {
                field: getattr(self, field)
                for field in getattr(self, "__annotations__", {})
            }

    pydantic_module.BaseModel = BaseModel
    pydantic_module.Field = Field
    pydantic_module.conint = conint
    pydantic_module.confloat = confloat
    pydantic_module.validator = validator

    sys.modules.setdefault("pydantic", pydantic_module)


def _ensure_azure_stub() -> None:
    try:  # pragma: no cover
        import azure  # type: ignore
        return
    except ModuleNotFoundError:
        pass

    azure_module = types.ModuleType("azure")
    ai_module = types.ModuleType("azure.ai")
    formrecognizer_module = types.ModuleType("azure.ai.formrecognizer")

    class DocumentAnalysisClient:  # pragma: no cover - no se usa directamente
        def __init__(self, *args, **kwargs) -> None:
            pass

        def begin_analyze_document(self, *args, **kwargs):
            raise RuntimeError("Azure SDK stub in use")

    formrecognizer_module.DocumentAnalysisClient = DocumentAnalysisClient

    core_module = types.ModuleType("azure.core")
    credentials_module = types.ModuleType("azure.core.credentials")

    class AzureKeyCredential:  # pragma: no cover - no se usa directamente
        def __init__(self, key: str) -> None:
            self.key = key

    credentials_module.AzureKeyCredential = AzureKeyCredential

    azure_module.ai = ai_module
    ai_module.formrecognizer = formrecognizer_module
    azure_module.core = core_module
    core_module.credentials = credentials_module

    sys.modules.setdefault("azure", azure_module)
    sys.modules.setdefault("azure.ai", ai_module)
    sys.modules.setdefault("azure.ai.formrecognizer", formrecognizer_module)
    sys.modules.setdefault("azure.core", core_module)
    sys.modules.setdefault("azure.core.credentials", credentials_module)


def _ensure_pypdf2_stub() -> None:
    try:  # pragma: no cover
        import PyPDF2  # type: ignore
        return
    except ModuleNotFoundError:
        pass

    pypdf2_module = types.ModuleType("PyPDF2")

    class _DummyPage:  # pragma: no cover - no se usa en las pruebas
        def extract_text(self) -> str:
            return ""

    class PdfReader:  # pragma: no cover - evita errores de importación
        def __init__(self, *args, **kwargs) -> None:
            self.pages = []

    pypdf2_module.PdfReader = PdfReader
    pypdf2_module.PageObject = _DummyPage

    sys.modules.setdefault("PyPDF2", pypdf2_module)


def _ensure_dotenv_stub() -> None:
    try:  # pragma: no cover
        import dotenv  # type: ignore
        return
    except ModuleNotFoundError:
        pass

    dotenv_module = types.ModuleType("dotenv")

    def load_dotenv(*args, **kwargs) -> None:  # pragma: no cover - función mínima
        return None

    dotenv_module.load_dotenv = load_dotenv
    sys.modules.setdefault("dotenv", dotenv_module)


def _ensure_torch_stub() -> None:
    try:  # pragma: no cover
        import torch  # type: ignore
        return
    except ModuleNotFoundError:
        pass

    torch_module = types.ModuleType("torch")

    class _CudaModule:  # pragma: no cover - no se utiliza directamente
        @staticmethod
        def is_available() -> bool:
            return False

    torch_module.cuda = _CudaModule()
    torch_module.bfloat16 = "bfloat16"

    sys.modules.setdefault("torch", torch_module)


def _ensure_transformers_stub() -> None:
    try:  # pragma: no cover
        import transformers  # type: ignore
        return
    except ModuleNotFoundError:
        pass

    transformers_module = types.ModuleType("transformers")

    class _DummyObject:  # pragma: no cover - utilizado solo para tipado
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _AutoBase:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _DummyObject(**kwargs)

    transformers_module.AutoConfig = _AutoBase
    transformers_module.AutoTokenizer = _AutoBase
    transformers_module.AutoModelForCausalLM = _AutoBase

    def pipeline(*args, **kwargs):  # pragma: no cover - evita dependencias reales
        def _runner(*_args, **_kwargs):
            raise RuntimeError("transformers pipeline stub invoked")

        return _runner

    transformers_module.pipeline = pipeline

    sys.modules.setdefault("transformers", transformers_module)


def _ensure_openai_stub() -> None:
    try:  # pragma: no cover
        from openai import OpenAI  # type: ignore  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    openai_module = types.ModuleType("openai")

    class _ChatCompletions:  # pragma: no cover - interfaz mínima
        def create(self, *args, **kwargs):
            raise RuntimeError("OpenAI stub invoked")

    class _Chat:
        def __init__(self) -> None:
            self.completions = _ChatCompletions()

    class OpenAI:  # pragma: no cover - sustituto liviano
        def __init__(self, api_key: str | None = None) -> None:
            self.api_key = api_key
            self.chat = _Chat()

    openai_module.OpenAI = OpenAI

    sys.modules.setdefault("openai", openai_module)


_ensure_fastapi_stub()
_ensure_pydantic_stub()
_ensure_azure_stub()
_ensure_pypdf2_stub()
_ensure_dotenv_stub()
_ensure_torch_stub()
_ensure_transformers_stub()
_ensure_openai_stub()
