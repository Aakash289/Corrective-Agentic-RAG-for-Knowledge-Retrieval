"""
src/crag/ingestion/bm25_index.py

Builds and queries a BM25 index over the same Chunk objects ChromaDB
embeds. Unlike embed_index.py, this has to be built once over the full
corpus, not incrementally per document, BM25's term statistics (how rare
a word is) depend on the whole corpus, not one file's worth of chunks.

rank_bm25 has no metadata store and scores by list position, not by key,
so a parallel chunk-ordered list travels alongside the tokenized corpus,
and any returned position gets translated back to a Chunk immediately
after scoring, never passed further as a raw index.
"""
import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from crag.ingestion.chunker import Chunk

INDEX_PATH = Path("data/bm25_index.pkl")


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace. Simple on
    purpose, BM25's scoring does the real work, the tokenizer just needs
    to be consistent between indexing and querying, which is why this one
    function is used in both build and query rather than two
    implementations that could drift apart."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25_index(all_chunks: list[Chunk]) -> tuple[BM25Okapi, list[Chunk]]:
    """Takes every chunk across the whole corpus at once, collected from
    all five documents' extract_text_and_sections calls, not one
    document's chunks at a time. Returns the index alongside the exact
    chunk list it was built from, in the same order, that ordering is
    what makes a later get_scores() result's positions meaningful."""
    tokenized_corpus = [_tokenize(c.text) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, all_chunks


def save_bm25_index(bm25: BM25Okapi, chunks: list[Chunk], path: Path = INDEX_PATH) -> None:
    """Pickles both the index and its parallel chunk list together, saving
    only one without the other would leave scores with no way to resolve
    back to a chunk_id."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)


def load_bm25_index(path: Path = INDEX_PATH) -> tuple[BM25Okapi, list[Chunk]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]


def query_bm25(
    query: str,
    bm25: BM25Okapi,
    chunks: list[Chunk],
    n_results: int = 10,
) -> list[tuple[Chunk, float]]:
    """Scores every chunk against the query, then immediately zips scores
    back to their Chunk objects by position before sorting, so nothing
    downstream of this function ever sees a raw list index, only
    chunk_id-bearing Chunk objects."""
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda pair: pair[1], reverse=True)
    return scored_chunks[:n_results]


CORPUS_DOC_IDS = ["Document_1", "Document_2", "Document_3", "Document_4", "Document_5"]
CORPUS_PDF_DIR = Path("data/raw_pdfs")


def get_corpus_bm25_index() -> tuple[BM25Okapi, list[Chunk]]:
    """The loader retrieve_text_node in the LangGraph pipeline actually
    needs: the full five-document corpus's BM25 index, ready to query.
    Checks INDEX_PATH first, if build_bm25_index has already been run
    and saved once (via save_bm25_index, presumably as part of a real
    ingestion script), this is fast, a straight pickle load. Only falls
    back to OCRing and chunking all five PDFs from scratch, then saving
    the result for next time, if nothing has been persisted yet. That
    fallback path is genuinely slow (roughly 108 pages of OCR across
    the whole corpus) and is meant to run once, ever, not on a live
    request path, if this function is hitting the fallback branch
    routinely rather than the fast load branch, that's a sign ingestion
    needs to be run as its own separate step first."""
    if INDEX_PATH.exists():
        return load_bm25_index()

    from crag.ingestion.chunker import extract_text_and_sections
    all_chunks: list[Chunk] = []
    for doc_id in CORPUS_DOC_IDS:
        _, _, chunks = extract_text_and_sections(f"{CORPUS_PDF_DIR}/{doc_id}.pdf", doc_id)
        all_chunks.extend(chunks)
    bm25, chunks = build_bm25_index(all_chunks)
    save_bm25_index(bm25, chunks)
    return bm25, chunks