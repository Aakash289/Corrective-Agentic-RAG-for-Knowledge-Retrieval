"""
src/crag/retrieval/reranker.py

Local cross-encoder reranking, the step that runs after BM25 and semantic
retrieval have been fused with RRF. A cross-encoder scores a query and a
chunk together in one forward pass, rather than embedding each separately
and comparing vectors, which is why it is far more accurate than either
BM25 or semantic similarity alone, but also why it only runs on a short
list of already-fused candidates instead of the whole corpus, scoring
every chunk in the index this way would be far too slow.
"""
from sentence_transformers import CrossEncoder
from crag.ingestion.chunker import Chunk

RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_TOP_N = 8  # matches the value settled on in config.py

_model = None  # loaded lazily, same reason as embed_index.py's _get_model,
                # don't pay for the model load if rerank_chunks never
                # actually gets called in a given process


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANKER_MODEL)
    return _model


def rerank_chunks(
    query: str,
    candidates: list[Chunk],
    top_n: int = RERANK_TOP_N,
) -> list[tuple[Chunk, float]]:
    """Scores every candidate against the query with the cross-encoder,
    then returns the top_n highest scoring, chunk paired with its score.
    candidates is expected to already be the RRF-fused, deduplicated
    shortlist coming out of retrieval, not the raw output of either
    individual retriever, and not the full corpus."""
    if not candidates:
        return []

    model = _get_model()
    # CrossEncoder.predict takes a list of (query, passage) pairs, one
    # forward pass per pair, this is the expensive part, which is exactly
    # why candidates needs to already be a short list by the time it
    # reaches this function
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_n]