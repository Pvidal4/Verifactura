"""Servicios de orquestación para extraer información desde distintos soportes."""

from __future__ import annotations

import base64
import logging
import mimetypes
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple

from app.config import Config
from app.services.llm_service import LocalLLMService, OpenAILLMService
from app.services.ocr_service import AzureOCRConfig, AzureOCRService
from app.services.pdf_service import PDFTextExtractor

# Conjuntos de extensiones soportadas que determinan la ruta de procesamiento
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TEXT_EXTENSIONS = {".json"}
XML_EXTENSIONS = {".xml"}
PDF_EXTENSIONS = {".pdf"}


LOGGER = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Estructura con el texto procesado y los campos estructurados resultantes."""

    fields: Dict[str, object]
    raw_text: str
    text_origin: Literal["input", "file", "ocr"]

    def to_payload(self) -> Dict[str, object]:
        """Convierte el resultado en un diccionario listo para serializar."""

        return {
            "fields": self.fields,
            "raw_text": self.raw_text,
            "text_origin": self.text_origin,
        }


class ExtractionService:
    """Gestiona la extracción de texto y la invocación a los modelos de lenguaje."""

    def __init__(self, config: Config) -> None:
        """Prepara los auxiliares necesarios tomando como base la configuración."""

        self._config = config
        self._pdf = PDFTextExtractor(config.MAX_CHARS_PER_CHUNK)
        # Diccionarios para crear instancias de LLM bajo demanda y cachearlas
        self._llm_factories: Dict[str, Callable[[], object]] = {}
        self._llm_cache: Dict[str, object] = {}
        self._llm_factories["api"] = partial(OpenAILLMService, config)
        self._llm_factories["local"] = partial(LocalLLMService, config)
        if not self._llm_factories:
            raise RuntimeError("No hay servicios LLM configurados para la extracción.")
        self._default_provider = "api" if "api" in self._llm_factories else next(
            iter(self._llm_factories)
        )
        # Cache de clientes OCR organizada por tupla (proveedor, endpoint, clave)
        self._ocr_cache: Dict[Tuple[str, str, str], AzureOCRService] = {}
        self._default_ocr_key: Optional[Tuple[str, str, str]] = None
        if config.azure_configured:
            endpoint = (config.AZURE_ENDPOINT or "").strip()
            key = (config.AZURE_KEY or "").strip()
            if endpoint and key:
                cache_key = ("azure", endpoint, key)
                self._ocr_cache[cache_key] = AzureOCRService(
                    AzureOCRConfig(endpoint=endpoint, key=key)
                )
                self._default_ocr_key = cache_key

    def _get_llm(self, provider: Optional[str] = None):
        """Devuelve una instancia de servicio LLM, reutilizando caches si existen."""

        key = (provider or self._default_provider).lower()
        if key not in self._llm_factories:
            available = ", ".join(sorted(self._llm_factories))
            raise RuntimeError(
                f"Proveedor LLM '{provider}' no disponible. Opciones: {available}."
            )
        if key not in self._llm_cache:
            self._llm_cache[key] = self._llm_factories[key]()
        return self._llm_cache[key]

    def _resolve_ocr_service(
        self,
        provider: Optional[str],
        *,
        endpoint: Optional[str],
        key: Optional[str],
        error_message: str,
    ) -> AzureOCRService:
        """Selecciona la instancia OCR adecuada considerando overrides en la llamada."""

        normalized_provider = (provider or "").strip().lower()
        normalized_endpoint = endpoint.strip() if isinstance(endpoint, str) else None
        normalized_key = key.strip() if isinstance(key, str) else None

        if not normalized_provider:
            if not normalized_endpoint and not normalized_key:
                if self._default_ocr_key is None:
                    raise RuntimeError(error_message)
                normalized_provider = self._default_ocr_key[0]
                normalized_endpoint = self._default_ocr_key[1]
                normalized_key = self._default_ocr_key[2]
            else:
                normalized_provider = "azure"

        if normalized_provider in {"azure", "azure-vision"}:
            final_endpoint = normalized_endpoint or (self._config.AZURE_ENDPOINT or "").strip()
            final_key = normalized_key or (self._config.AZURE_KEY or "").strip()
            if not final_endpoint or not final_key:
                raise RuntimeError(error_message)
            cache_key = ("azure", final_endpoint, final_key)
            service = self._ocr_cache.get(cache_key)
            if service is None:
                service = AzureOCRService(
                    AzureOCRConfig(endpoint=final_endpoint, key=final_key)
                )
                self._ocr_cache[cache_key] = service
            return service

        raise RuntimeError(f"Proveedor OCR '{provider}' no disponible.")

    def _needs_ocr(self, extension: str, text: str) -> bool:
        """Determina si es imprescindible recurrir a OCR para obtener texto legible."""

        if extension in IMAGE_EXTENSIONS:
            return True
        if extension in PDF_EXTENSIONS and not text:
            return True
        return False

    @staticmethod
    def _normalize_image_media_type(data: bytes, content_type: Optional[str]) -> str:
        """Intenta inferir el tipo MIME adecuado para una imagen en base a su contenido."""

        normalized = (content_type or "").lower()
        if normalized.startswith("image/"):
            return normalized
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data[:4] in (b"II*\x00", b"MM\x00*"):
            return "image/tiff"
        return "image/png"

    def _encode_vision_images(
        self,
        images: Iterable[Tuple[bytes, Optional[str]]],
        limit: int = 3,
    ) -> Optional[List[Dict[str, str]]]:
        """Convierte imágenes binarias en payloads base64 para modelos multimodales."""

        encoded: List[Dict[str, str]] = []
        for data, media_type in images:
            if not data:
                continue
            normalized = self._normalize_image_media_type(data, media_type)
            encoded.append(
                {
                    "media_type": normalized,
                    "data": base64.b64encode(data).decode("ascii"),
                }
            )
            if len(encoded) >= limit:
                break
        return encoded or None

    def _extract_text_from_pdf_with_ocr(
        self, data: bytes, ocr_service: Optional[AzureOCRService]
    ) -> str:
        """Aplica OCR a un PDF, intentando diferentes estrategias si es necesario."""

        if ocr_service is None:
            raise RuntimeError(
                "Azure OCR no está configurado pero es requirido para este tipo de archivo"
            )
        text = ocr_service.extract_text(
            data,
            content_type="application/pdf",
        )
        if text:
            return text

        images = self._pdf.render_page_images(data)
        if not images:
            LOGGER.warning(
                "No fue posible renderizar el PDF a imágenes, se reintentará el OCR directo sobre el PDF."
            )
            text = ocr_service.extract_text(
                data,
                content_type="application/pdf",
            )
            if not text:
                raise RuntimeError(
                    "No se pudo extraer texto del PDF mediante OCR."
                )
            return text
        fragments = []
        for image_data, content_type in images:
            text = ocr_service.extract_text(image_data, content_type=content_type)
            if text:
                fragments.append(text)
        joined = "\n\n".join(fragment.strip() for fragment in fragments if fragment.strip())
        if not joined:
            raise RuntimeError("No se pudo extraer texto del PDF mediante OCR.")
        return joined

    def _extract_text_from_file(
        self,
        filename: str,
        data: bytes,
        content_type: Optional[str] = None,
        *,
        force_ocr: bool = False,
        ocr_service: Optional[AzureOCRService] = None,
    ) -> str:
        """Resuelve cómo obtener texto de un archivo según su extensión y contenido."""

        suffix = Path(filename).suffix.lower()
        normalized_content_type = (content_type or "").lower()
        if suffix in PDF_EXTENSIONS or normalized_content_type == "application/pdf":
            if force_ocr:
                return self._extract_text_from_pdf_with_ocr(data, ocr_service)
            text = self._pdf.read_text(data)
            if text:
                return text
            return self._extract_text_from_pdf_with_ocr(data, ocr_service)
        if suffix in TEXT_EXTENSIONS or suffix in XML_EXTENSIONS:
            return data.decode("utf-8", errors="replace")
        if ocr_service is None:
            raise RuntimeError(
                "Azure OCR no está configurado pero es requirido para este tipo de archivo"
            )
        if suffix in PDF_EXTENSIONS:
            return self._extract_text_from_pdf_with_ocr(data, ocr_service)
        if not normalized_content_type:
            guessed = mimetypes.guess_type(filename)[0]
            normalized_content_type = guessed.lower() if guessed else None
        return ocr_service.extract_text(
            data,
            content_type=normalized_content_type,
        )

    def extract_from_text(
        self,
        text: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        openai_api_key: Optional[str] = None,
        vision_images: Optional[List[Dict[str, str]]] = None,
        text_origin: Literal["input", "file", "ocr"] = "input",
    ) -> ExtractionResult:
        """Envía texto directamente al LLM elegido para obtener los campos estructurados."""

        sanitized_model = model.strip() if isinstance(model, str) else None
        llm = self._get_llm(provider)
        extracted = llm.extract(
            text,
            model=sanitized_model,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            api_key=openai_api_key,
            vision_images=vision_images,
        )
        return ExtractionResult(fields=extracted, raw_text=text, text_origin=text_origin)

    def extract_from_image(
        self,
        filename: str,
        data: bytes,
        content_type: Optional[str] = None,
        *,
        provider: Optional[str] = None,
        use_vision: bool = False,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        openai_api_key: Optional[str] = None,
        ocr_provider: Optional[str] = None,
        ocr_endpoint: Optional[str] = None,
        ocr_key: Optional[str] = None,
    ) -> ExtractionResult:
        """Extrae texto vía OCR y luego invoca el flujo genérico de extracción."""

        ocr_service = self._resolve_ocr_service(
            ocr_provider,
            endpoint=ocr_endpoint,
            key=ocr_key,
            error_message="Azure OCR no está configurado pero es requirido para la extracción de imagen",
        )
        suffix = Path(filename).suffix.lower()
        if content_type:
            content_type = content_type.lower()
        else:
            guessed = mimetypes.guess_type(filename)[0]
            content_type = guessed.lower() if guessed else None
        if suffix in PDF_EXTENSIONS or content_type == "application/pdf":
            return self._extract_text_from_pdf_with_ocr(data, ocr_service)
        if suffix and suffix not in IMAGE_EXTENSIONS and not (
            (content_type or "").startswith("image/")
        ):
            raise RuntimeError("Formato de imagen no admitido")
        # Las imágenes siempre envían su representación visual al modelo
        use_vision = True
        text = ocr_service.extract_text(data, content_type=content_type)
        if not text:
            raise RuntimeError("No se pudo extraer texto de la imagen ingresada")
        vision_images: Optional[List[Dict[str, str]]] = None
        if use_vision:
            vision_images = self._encode_vision_images([(data, content_type)], limit=1)
        return self.extract_from_text(
            text,
            provider=provider,
            model=model,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            openai_api_key=openai_api_key,
            vision_images=vision_images,
            text_origin="ocr",
        )

    def extract_from_file(
        self,
        filename: str,
        data: bytes,
        content_type: Optional[str] = None,
        *,
        force_ocr: bool = False,
        use_vision: bool = False,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        openai_api_key: Optional[str] = None,
        ocr_provider: Optional[str] = None,
        ocr_endpoint: Optional[str] = None,
        ocr_key: Optional[str] = None,
    ) -> ExtractionResult:
        """Decide si se debe aplicar OCR o lectura directa antes de usar el LLM."""

        suffix = Path(filename).suffix.lower()
        normalized_content_type = (content_type or "").lower()
        force_ocr = bool(force_ocr)
        allow_vision = bool(
            suffix in PDF_EXTENSIONS
            or suffix in IMAGE_EXTENSIONS
            or normalized_content_type.startswith("image/")
            or normalized_content_type == "application/pdf"
        )
        if use_vision and not allow_vision:
            use_vision = False
        if suffix in IMAGE_EXTENSIONS:
            # Derivar a la ruta especializada para garantizar OCR explícito
            return self.extract_from_image(
                filename,
                data,
                content_type,
                provider=provider,
                use_vision=use_vision,
                model=model,
                temperature=temperature,
                top_p=top_p,
                reasoning_effort=reasoning_effort,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                ocr_provider=ocr_provider,
                ocr_endpoint=ocr_endpoint,
                ocr_key=ocr_key,
                openai_api_key=openai_api_key,
            )
        if force_ocr and suffix not in PDF_EXTENSIONS:
            force_ocr = False
        text = ""
        ocr_service_instance: Optional[AzureOCRService] = None
        vision_images: Optional[List[Dict[str, str]]] = None

        def require_ocr_service() -> AzureOCRService:
            """Inicializa el servicio OCR una única vez si llega a ser necesario."""

            nonlocal ocr_service_instance
            if ocr_service_instance is None:
                ocr_service_instance = self._resolve_ocr_service(
                    ocr_provider,
                    endpoint=ocr_endpoint,
                    key=ocr_key,
                    error_message="Azure OCR no está configurado pero es requirido para este tipo de archivo",
                )
            return ocr_service_instance

        text_origin: Literal["file", "ocr"] = "file"

        if not force_ocr:
            if suffix in PDF_EXTENSIONS:
                text = self._pdf.read_text(data)
            elif suffix in TEXT_EXTENSIONS or suffix in XML_EXTENSIONS:
                text = data.decode("utf-8", errors="replace")
        if force_ocr or self._needs_ocr(suffix, text) or not text:
            text = self._extract_text_from_file(
                filename,
                data,
                content_type,
                force_ocr=force_ocr,
                ocr_service=require_ocr_service(),
            )
            text_origin = "ocr"
        if use_vision and suffix in PDF_EXTENSIONS:
            try:
                vision_images = self._encode_vision_images(
                    self._pdf.render_page_images(data)
                )
            except Exception:  # pragma: no cover - fall back silently
                vision_images = None
        return self.extract_from_text(
            text,
            provider=provider,
            model=model,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            openai_api_key=openai_api_key,
            vision_images=vision_images,
            text_origin=text_origin,
        )
