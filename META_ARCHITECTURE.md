# Verifactura Extraction API Architecture

This project exposes the existing PDF/text extraction workflow as a production-ready FastAPI service.

## Layers

1. **Presentation (FastAPI Routers)**
   - `app/routes/extract.py` defines the `/api/v1/extract` endpoint.
   - Accepts JSON (`{"text": "..."}`), plain text, or `multipart/form-data` uploads with `file` and optional `text` fields.
   - Normalises responses, returning schema-compliant JSON or explicit error messages.

2. **Application Services**
   - `app/services/extraction_service.py` orchestrates file ingestion, OCR, chunking, and LLM calls.
   - Delegates specialised work to the PDF, OCR, and LLM services.

3. **Domain Services**
   - `app/services/pdf_service.py`: deterministic text extraction for digital PDFs.
   - `app/services/ocr_service.py`: Azure Form Recognizer integration for image-based content.
   - `app/services/llm_service.py`: Structured JSON extraction through OpenAI's Responses API.

4. **Configuration**
   - `app/config.py` consolidates environment configuration, including OpenAI and Azure credentials.

## Request Flow

```
Client Request ─▶ FastAPI Router ─▶ ExtractionService ─┬─▶ PDFTextExtractor
                                                        ├─▶ AzureOCRService
                                                        └─▶ OpenAILLMService ─▶ Structured JSON
```

## Environment Variables

- `OPENAI_API_KEY` *(required)*
- `OPENAI_MODEL` (default: `gpt-4.1-mini`)
- `AZURE_FORM_RECOGNIZER_ENDPOINT`, `AZURE_FORM_RECOGNIZER_KEY` *(required for OCR)*
- `MAX_CHARS_PER_CHUNK` (default: `50000`)

## Running

```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Send a POST request:

```
curl -X POST http://localhost:8000/api/v1/extract \
     -H 'Content-Type: application/json' \
     -d '{"text": "Contenido del documento"}'
```
