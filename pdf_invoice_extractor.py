#!/usr/bin/env python3
"""
PDF → Text (PyPDF2/pypdf) → OpenAI (Structured JSON) → per-file JSON + combined.json

Usage:
  python pdf_invoice_extractor.py --input ./carpeta_pdfs --out ./out --model gpt-4.1-mini

Requirements:
  pip install PyPDF2 pypdf python-dotenv openai tqdm
  # (Either PyPDF2 or pypdf will work; we'll prefer pypdf if present.)
  # Set OPENAI_API_KEY in your environment or in a .env file next to this script.

Notes:
  - Designed for facturas vehiculares (Ecuador) but the schema is easily editable.
  - Uses Structured Outputs (JSON Schema) for reliable JSON parsing.
"""

import os, re
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Prefer pypdf (actively maintained); fall back to PyPDF2 if needed.
try:
    from pypdf import PdfReader  # type: ignore
    _PDF_LIB = "pypdf"
except Exception:
    from PyPDF2 import PdfReader  # type: ignore
    _PDF_LIB = "PyPDF2"

from dotenv import load_dotenv

# OpenAI SDK (2025). If using older versions, adjust imports accordingly.
try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit(
        "The 'openai' package is required. Install with: pip install openai\n"
        f"Import error: {e}"
    )

from tqdm import tqdm



def _extract_json_object(text: str):
    """
    Try very hard to extract a top-level JSON object from a text.
    Returns a Python object (dict) or raises.
    """
    text = text.strip()
    # Fast path: direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        # If it's a list with a single dict, accept it
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
    except Exception:
        pass

    # Try to find the first {...} block (balanced) and parse it
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
        start = text.find("{", start + 1)

    # As a last resort, try to parse after removing trailing commas (very limited fix)
    try:
        sanitized = re.sub(r',\s*}', '}', text)
        obj = json.loads(sanitized)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    raise RuntimeError("No pude extraer un objeto JSON válido del texto de salida del modelo.")


def read_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF using pypdf / PyPDF2."""
    reader = PdfReader(str(pdf_path))
    parts: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            # Some PDFs may return None from extract_text; guard it.
            txt = page.extract_text() or ""
        except Exception as e:
            txt = f""  # skip problematic pages but continue
        parts.append(txt)
    text = "\n".join(parts).strip()
    # Very light cleanup: collapse super long whitespace runs
    return " ".join(text.split())


def chunk_text(text: str, max_chars: int = 50_000) -> List[str]:
    """Naive chunking by characters to avoid token limits. Splits at paragraph boundaries when possible."""
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Try to break on a nearby sentence/paragraph boundary
        split_at = text.rfind("\n\n", start, end)
        if split_at == -1:
            split_at = text.rfind(". ", start, end)
        if split_at == -1:
            split_at = end
        chunks.append(text[start:split_at].strip())
        start = split_at
    return [c for c in chunks if c]


def get_openai_client() -> OpenAI:
    """Initialize OpenAI client, loading OPENAI_API_KEY from env/.env."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Put it in your environment or a .env file.")
    return OpenAI(api_key=api_key)


INVOICE_SCHEMA: Dict[str, Any] = {
    "name": "vehicular_invoice",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            # Identificación del documento
            "NUMERO_FACTURA": {"type": "string", "description": "Número completo de la factura, ej. 002-101-000004850"},
            "FECHA_DOCUMENTO": {"type": "string", "description": "Fecha en formato dd/mm/aaaa o aaaa-mm-dd"},
            "RUC": {"type": "string", "description": "RUC o cédula del emisor"},
            "RAZON_SOCIAL": {"type": "string"},
            "DIRECCION": {"type": "string"},
            # Vehículo
            "VIN_CHASIS": {"type": "string"},
            "MOTOR": {"type": "string"},
            "MARCA": {"type": "string"},
            "MODELO": {"type": "string"},
            "ANIO_MODELO": {"type": "string"},
            "COLOR": {"type": "string"},
            # Totales
            "SUBTOTAL": {"type": "number"},
            "DESCUENTO": {"type": "number"},
            "IVA": {"type": "number"},
            "TOTAL": {"type": "number"},
            # Metadatos
            "MONEDA": {"type": "string", "description": "USD si aplica"},
            "observaciones": {"type": "string", "description": "Notas o advertencias de extracción"},
            "confidence": {
                "type": "object",
                "additionalProperties": True,
                "description": "Mapa campo→confianza (0.0–1.0) basado en señales del texto."
            }
        },
        "required": ["NUMERO_FACTURA", "FECHA_DOCUMENTO", "TOTAL"]
    }
}


SYSTEM_PROMPT = """Eres un analista experto en extracción de datos de facturas vehiculares ecuatorianas.
Devuelve **un único objeto JSON** con **exactamente** las siguientes claves y tipos (no agregues ni quites claves):
{
  "FECHA_DOCUMENTO": string|null,          # dd/mm/aaaa si es posible
  "DIRECCION": string|null,
  "MODELO_HOMOLOGADO_ANT": string|null,
  "SUBSIDIO": number|null,                  # si no consta, usa null (no NaN)
  "AÑO": number|null,                       # año modelo o registro; número (p.ej. 2023)
  "SUBTOTAL": number|null,
  "CLASE": string|null,
  "TOTAL": number|null,
  "CILINDRAJE": string|null,                # conservar formato (p.ej. "1.451 C.C.")
  "MODELO": string|null,
  "MODELO_REGISTRADO_SRI": string|null,
  "RAMV_CPN": string|null,
  "RUEDAS": number|null,
  "DESCUENTO": number|null,
  "NUMERO_FACTURA": string|null,            # normaliza con guiones completos (ej.: 002-101-000004850)
  "COLOR": string|null,
  "MOTOR": string|null,
  "NOMBRE_CLIENTE": string|null,
  "CAPACIDAD": number|null,
  "MARCA": string|null,
  "RUC": string|null,
  "COMBUSTIBLE": string|null,
  "EJES": number|null,
  "TIPO": string|null,
  "IVA": number|null,
  "CONCESIONARIA": string|null,
  "TONELAJE": number|null,
  "VIN_CHASIS": string|null,
  "PAIS_ORIGEN": string|null,
  "ETIQUETA": string|null                   # p.ej. "COMPLETA" si el documento parece completo
}

REGLAS:
- NO inventes datos. Si un valor no está, usa null. **Nunca uses NaN** (JSON no lo permite).
- Devuelve montos y cantidades como números (no como texto). No incluyas símbolos de moneda.
- FECHA_DOCUMENTO en formato dd/mm/aaaa si el texto lo permite; de lo contrario, conserva un formato claro.
- NUMERO_FACTURA normalizado con guiones si aplica.
- Limpia espacios, saltos de línea, y texto de sellos/footers.
- NO devuelvas ningún texto adicional fuera del JSON, ni explicaciones, ni claves extra.
"""


def call_openai_structured(client: OpenAI, model: str, pdf_text: str) -> Dict[str, Any]:
    """
    Call the Responses API with Structured Outputs.
    If text is too long, summarize/condense progressively but preserve relevant fields.
    """
    # Simple chunking: ask model to extract per chunk, then merge by non-null precedence.
    chunks = chunk_text(pdf_text, max_chars=45_000)
    partial_results: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        # Using the Responses API with 'response_format' as JSON schema (Structured Outputs)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extrae los campos del JSON Schema a partir del siguiente texto OCR de la factura (parte %d/%d):%s" % (idx+1, len(chunks), chunk)}
            ],
            response_format={"type": "json_object"}
        )
        # Many SDKs expose a convenience property for the primary text. Here we directly take the JSON tool output.
        # The Responses API returns output content parts; find the JSON segment.
        data_json = None
        try:
            # Chat Completions standard field
            if hasattr(resp, 'choices') and resp.choices:
                msg = getattr(resp.choices[0], 'message', None)
                if msg is not None:
                    content = getattr(msg, 'content', None)
                    if content:
                        data_json = _extract_json_object(content)
        except Exception:
            pass
        if not isinstance(data_json, dict):
            raise RuntimeError('La respuesta del modelo no es un JSON de objeto válido. Revisa el SDK/versión.')
        partial_results.append(data_json)

    # Merge: left-to-right, prefer first non-null; sum confidences if present (bounded)
    merged: Dict[str, Any] = {}
    conf_acc: Dict[str, float] = {}
    for pr in partial_results:
        for k, v in pr.items():
            if k == "confidence" and isinstance(v, dict):
                for ck, cv in v.items():
                    try:
                        conf_acc[ck] = max(conf_acc.get(ck, 0.0), float(cv))
                    except Exception:
                        pass
                continue
            if k not in merged or merged.get(k) in (None, "", []):
                merged[k] = v
    if conf_acc:
        merged["confidence"] = conf_acc
    return merged


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_folder(input_dir: Path, out_dir: Path, model: str, pause_s: float = 0.0) -> None:
    client = get_openai_client()
    pdf_paths = sorted([p for p in input_dir.glob("**/*.pdf") if p.is_file()])
    results: List[Dict[str, Any]] = []

    if not pdf_paths:
        print(f"No se encontraron PDFs en: {input_dir}")
        return

    print(f"Usando librería PDF: {_PDF_LIB}")
    print(f"Procesando {len(pdf_paths)} PDFs...")

    for pdf_path in tqdm(pdf_paths, desc="Extrayendo"):
        try:
            text = read_pdf_text(pdf_path)
            if not text or len(text) < 20:
                raise ValueError("Texto demasiado corto o vacío extraído del PDF; puede ser un PDF basado en imágenes.")

            data = call_openai_structured(client, model, text)
            # Attach filename reference
            data["_file"] = pdf_path.name
            results.append(data)

            save_json(data, out_dir / "json" / f"{pdf_path.stem}.json")
        except Exception as e:
            err = {
                "_file": pdf_path.name,
                "error": str(e)
            }
            results.append(err)
            save_json(err, out_dir / "json" / f"{pdf_path.stem}.error.json")
        if pause_s > 0:
            time.sleep(pause_s)

    # Save combined
    save_json(results, out_dir / "combined.json")
    print(f"Listo. JSONs individuales en {out_dir / 'json'} y combinado en {out_dir / 'combined.json'}.")


def main():
    parser = argparse.ArgumentParser(description="Extrae texto de PDFs, envía a OpenAI y guarda JSON estructurado.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Carpeta con PDFs (recursivo).")
    parser.add_argument("--out", "-o", type=str, default="./out", help="Carpeta de salida.")
    parser.add_argument("--model", "-m", type=str, default="gpt-4.1-mini", help="Modelo OpenAI (p. ej., gpt-4.1, gpt-4.1-mini).")
    parser.add_argument("--pause", type=float, default=0.0, help="Pausa en segundos entre archivos para evitar rate limits.")
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    process_folder(input_dir, out_dir, args.model, pause_s=args.pause)


if __name__ == "__main__":
    main()
