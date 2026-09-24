"""
src/crag/ingestion/ocr_utils.py

process_document() is the one place that decides, per page, whether to
read the text layer or fall back to OCR, and it does that decision only
once per document. pdf_extract.py and chunker.py both consume its output
(DocumentContent) rather than each independently opening and OCR'ing the
same PDF, which is what happened before this restructure, each file that
needed OCR was getting OCR'd twice, once for plain text, once for section
detection.
"""
import io
from dataclasses import dataclass
import pymupdf
import pytesseract
from pytesseract import Output
from PIL import Image

OCR_DPI = 200
MIN_CHARS_PER_PAGE = 20  # below this, a page has no usable text layer


@dataclass
class LineInfo:
    """One line of text with a size signal. 'size' means font size in points
    when the line came from a real text layer, or average OCR word height in
    pixels when it came from OCR. Different units, but used identically as
    the heading-detection signal, chunker.py never needs to know which one
    produced a given line."""
    text: str
    size: float
    page_num: int


@dataclass
class DocumentContent:
    pages_plain_text: list[str]  # one entry per page, for joining into full text
    lines: list[LineInfo]        # every line across the whole doc, in reading order
    used_ocr: bool               # true if any page needed the OCR fallback


def _extract_text_layer_lines(page, page_num: int) -> list[LineInfo]:
    """Reads a page's real text layer via PyMuPDF's 'dict' mode, which keeps
    font size per span, the signal used to detect headings when a text
    layer actually exists. Returns an empty list if the page has no usable
    text, the caller treats that as 'needs OCR'."""
    lines: list[LineInfo] = []
    for block in page.get_text("dict")["blocks"]:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            spans = line["spans"]
            if not spans:
                continue
            text = " ".join(s["text"].strip() for s in spans if s["text"].strip())
            if not text:
                continue
            # a line can technically mix span sizes, use the largest, the
            # same choice PyMuPDF's own heading intuition would make
            size = max(s["size"] for s in spans)
            lines.append(LineInfo(text=text, size=size, page_num=page_num))
    return lines


def _ocr_page_lines(page, page_num: int) -> tuple[str, list[LineInfo]]:
    """Rasterizes one page and runs pytesseract's word-level data extraction,
    not plain image_to_string, since image_to_data's per-word height is the
    OCR equivalent of font size, the same heading signal in different units.
    Returns (plain_text, lines)."""
    pix = page.get_pixmap(dpi=OCR_DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    data = pytesseract.image_to_data(img, output_type=Output.DICT)

    grouped: dict[tuple, list[int]] = {}
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        grouped.setdefault(key, []).append(i)

    lines: list[LineInfo] = []
    plain_text_parts: list[str] = []
    for key in sorted(grouped.keys()):
        idxs = grouped[key]
        words = [data["text"][i].strip() for i in idxs]
        heights = [data["height"][i] for i in idxs]
        line_text = " ".join(words)
        lines.append(LineInfo(text=line_text, size=sum(heights) / len(heights), page_num=page_num))
        plain_text_parts.append(line_text)

    return "\n".join(plain_text_parts), lines


def process_document(pdf_path: str) -> DocumentContent:
    """The single entry point for reading a PDF. Decides per page whether
    the text layer is usable or OCR is needed, so a document with some
    real pages and some scanned pages (not seen in this corpus yet, but
    not impossible) is handled correctly rather than assuming a whole
    file is one or the other. Every page that does need OCR gets OCR'd
    exactly once here, whatever else later needs that page's content
    reads it from the returned DocumentContent instead of re-opening the
    PDF and OCR'ing it again."""
    doc = pymupdf.open(pdf_path)
    pages_plain_text: list[str] = []
    all_lines: list[LineInfo] = []
    used_ocr = False

    for page_num, page in enumerate(doc):
        text_lines = _extract_text_layer_lines(page, page_num)
        page_text = " ".join(l.text for l in text_lines)

        if len(page_text.strip()) >= MIN_CHARS_PER_PAGE:
            pages_plain_text.append(page.get_text().strip())
            all_lines.extend(text_lines)
        else:
            print(f"  OCR'ing page {page_num + 1}/{len(doc)}...")
            ocr_text, ocr_lines = _ocr_page_lines(page, page_num)
            pages_plain_text.append(ocr_text)
            all_lines.extend(ocr_lines)
            used_ocr = True

    return DocumentContent(pages_plain_text=pages_plain_text, lines=all_lines, used_ocr=used_ocr)