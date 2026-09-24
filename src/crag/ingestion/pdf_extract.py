"""
src/crag/ingestion/pdf_extract.py

Text extraction is now a thin wrapper around ocr_utils.process_document,
which is where the actual text-layer-vs-OCR decision happens. This file
just joins the per-page text and reports whether OCR was used, plus the
file-hash dedupe check.
"""
import hashlib
from crag.ingestion.ocr_utils import process_document, DocumentContent


def extract_pdf_text(pdf_path: str) -> tuple[str, bool]:
    """Convenience wrapper for standalone use and testing. If you're also
    calling chunk detection on the same file, call process_document once
    yourself and pass its result to both this function's sibling in
    chunker.py and here, rather than calling this function, which is what
    actually avoids OCR running twice per document. See
    extract_text_and_sections in chunker.py for the combined entry point
    the real ingestion pipeline should use."""
    content = process_document(pdf_path)
    return full_text_from_content(content), content.used_ocr


def full_text_from_content(content: DocumentContent) -> str:
    """Joins a DocumentContent's per-page text. Split out from
    extract_pdf_text so a caller who already has a DocumentContent (from
    calling process_document once) can get the full text without
    re-running OCR."""
    return "\n\n".join(content.pages_plain_text)


def file_content_hash(pdf_path: str) -> str:
    """Call on every file before it enters raw_pdfs/, skip any hash already
    seen. Catches a duplicate file save before it doubles that content's
    weight in the corpus. sha256 over the raw file bytes, not the extracted
    text, since a byte-identical file save is the case this is built to
    catch cheaply."""
    with open(pdf_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()