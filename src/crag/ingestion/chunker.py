"""
src/crag/ingestion/chunker.py

Two-stage chunking. Stage 1 finds section boundaries from a DocumentContent
(ocr_utils.py), using each line's size signal, font size or OCR height,
whichever produced it, with an adaptive threshold rather than a fixed
number. Stage 2 is recursive character chunking within each section,
700 words with 100 words of overlap.

extract_text_and_sections() is the real entry point for the ingestion
pipeline: it calls process_document exactly once and derives both the
full text and the sections from that single result, which is what
actually avoids OCR running twice per document.
"""
import hashlib
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from crag.ingestion.ocr_utils import process_document, DocumentContent, LineInfo
from crag.ingestion.pdf_extract import full_text_from_content

# RecursiveCharacterTextSplitter: tries the largest separator first (paragraph
# breaks, then lines, then sentences, then words), only falling back to a
# harder split when a piece is still too long, this keeps chunks ending on
# a natural boundary far more often than a fixed-size sliding window would

HEADING_HEIGHT_RATIO = 1.3  # a line is a heading if its size is this many
                             # times the document's own median line size,
                             # adaptive rather than a fixed number, since
                             # font size in points and OCR height in pixels
                             # are different scales, and even within one
                             # scale, one hardcoded constant broke on the
                             # first real document it was tried against


@dataclass
class Section:
    heading: str
    text: str
    page_start: int
    page_end: int


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section_heading: str
    text: str
    page_start: int
    page_end: int


def sections_from_content(content: DocumentContent) -> list[Section]:
    """Stage 1, structure-aware, operating on lines already extracted by
    process_document, no PDF reopened, no OCR re-run. Works the same way
    whether the lines came from a real text layer or from OCR, since both
    are normalized into LineInfo(text, size, page_num) by ocr_utils.py."""
    lines = content.lines
    if not lines:
        return []

    sizes = sorted(l.size for l in lines)
    median_size = sizes[len(sizes) // 2]
    heading_threshold = median_size * HEADING_HEIGHT_RATIO

    sections: list[Section] = []
    heading, buffer, page_start = "Untitled", [], 0

    for line in lines:
        if line.size >= heading_threshold and line.text != heading:
            if buffer:
                sections.append(Section(heading, " ".join(buffer), page_start, line.page_num))
            heading, buffer, page_start = line.text, [], line.page_num
        else:
            buffer.append(line.text)

    if buffer:
        sections.append(Section(heading, " ".join(buffer), page_start, lines[-1].page_num))
    return sections


def chunk_sections(doc_id: str, sections: list[Section]) -> list[Chunk]:
    """Stage 2: recursive character chunking within each section, never
    across one. chunk_id is a content hash so it is stable across reruns,
    which is what keeps citation links from breaking if ingestion runs
    twice on the same files."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        length_function=lambda t: len(t.split()),
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks: list[Chunk] = []
    for section in sections:
        for piece in splitter.split_text(section.text):
            chunk_id = hashlib.sha256(f"{doc_id}:{section.heading}:{piece}".encode()).hexdigest()[:16]
            chunks.append(Chunk(chunk_id, doc_id, section.heading, piece, section.page_start, section.page_end))
    return chunks


def extract_text_and_sections(pdf_path: str, doc_id: str) -> tuple[str, bool, list[Chunk]]:
    """The real ingestion entry point: OCRs (or reads the text layer of)
    a document exactly once, then derives full_text, used_ocr, and chunks
    from that single pass. Use this in the ingestion notebook instead of
    calling extract_pdf_text and detect_sections separately, that pattern
    is what caused every OCR'd document to be OCR'd twice before this
    restructure."""
    content = process_document(pdf_path)
    full_text = full_text_from_content(content)
    sections = sections_from_content(content)
    chunks = chunk_sections(doc_id, sections)
    return full_text, content.used_ocr, chunks


def detect_sections(pdf_path: str) -> list[Section]:
    """Standalone convenience wrapper, kept for testing this stage in
    isolation. Calls process_document on its own, so using this alongside
    extract_pdf_text separately still double-OCRs, use
    extract_text_and_sections for the real pipeline."""
    content = process_document(pdf_path)
    return sections_from_content(content)