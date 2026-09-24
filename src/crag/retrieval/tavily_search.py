"""
src/crag/retrieval/tavily_search.py

The corrective fallback path: when the reranked, fused results from
retrieval score too low to trust, this queries Tavily's web search API
instead of forcing an answer from weak internal context. This is what
makes the system "corrective" rather than just a fixed RAG pipeline, a
grading step elsewhere decides when internal retrieval isn't good enough,
and this module is what it falls back to when that happens.

Results are normalized into the same Chunk shape the rest of retrieval
uses (chunk_id, doc_id, section_heading, text, page_start, page_end), so
downstream code (reranking, the generation node) doesn't need to know or
care whether a given chunk came from the internal corpus or a live web
search. Chunk has no field built for a URL or a page-agnostic source, so
two fields get repurposed deliberately here, doc_id holds the page's URL
instead of a "Document_N" identifier, since doc_id's real role is
identifying which source a chunk came from, a URL does that job for a
web result. page_start and page_end are set to 0, since a web page has
no PDF page range, 0 signals "not applicable" rather than a real page.
"""

from tavily import TavilyClient
from crag.ingestion.chunker import Chunk
from crag.config import TAVILY_API_KEY
import hashlib

_client = TavilyClient(api_key=TAVILY_API_KEY)

TAVILY_MAX_RESULTS = 5  # kept small deliberately, this is a fallback path,
                          # not the primary retrieval source, no need for
                          # the same breadth as internal corpus retrieval


def web_search_fallback(query: str, max_results: int = TAVILY_MAX_RESULTS) -> list[Chunk]:
    """Runs a Tavily search and wraps each result as a Chunk so it can
    flow through the same reranking and generation code paths as
    internal retrieval results. chunk_id is prefixed with "web_" so it's
    immediately distinguishable from an internal chunk_id if it shows up
    in a citation or a debug log."""
    response = _client.search(
        query=query,
        max_results=max_results,
        include_answer=False,  # we want raw source snippets to feed into
                                 # our own generation step, not Tavily's own
                                 # summarized answer, that would bypass the
                                 # citation and grounding checks the rest
                                 # of this pipeline is built around
    )

    chunks = []
    for i, result in enumerate(response.get("results", [])):
        url = result.get("url", "")
        title = result.get("title") or url or "Unknown source"
        chunks.append(
            Chunk(
                chunk_id=f"web_{hashlib.sha256(url.encode()).hexdigest()[:16]}",  # sha256, not
                # Python's built-in hash(), matching chunker.py's pattern:
                # hash() is randomized per process, so the same URL would
                # get a different chunk_id on every run, sha256 is stable
                doc_id=url,
                section_heading=title,
                text=result.get("content", ""),
                page_start=0,
                page_end=0,
            )
        )
    return chunks