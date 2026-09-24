"""
src/crag/retrieval/fusion.py

Reciprocal rank fusion (RRF), the step that combines the BM25 result list
and the semantic (ChromaDB) result list into one ranked candidate list.
RRF works on rank position, not raw score, which is what makes it usable
here at all, BM25 scores and cosine distances have no shared scale, so
averaging or summing them directly would be comparing numbers that mean
different things. Rank position, "how many results beat this one," is
comparable across any two retrievers.

Graph retrieval is deliberately excluded from this fusion, its relevance
signal is structural (does this entity connect to that one), not a
similarity ranking, mixing it into RRF alongside BM25 and semantic would
be fusing two different kinds of relevance as if they were the same kind.
"""
from crag.ingestion.chunker import Chunk

RRF_K = 60  # the constant in the 1/(k+rank) formula, damps the influence
            # of any single retriever's exact rank ordering, 60 is the
            # standard default from the original RRF paper and what this
            # project settled on rather than tuning further


def reciprocal_rank_fusion(
    ranked_lists: list[list[Chunk]],
    k: int = RRF_K,
) -> list[tuple[Chunk, float]]:
    """Takes any number of already-ranked chunk lists (here, BM25's and
    semantic's, in that order or either order, RRF doesn't care) and
    returns one fused, deduplicated list ordered by combined RRF score.
    A chunk that appears in both input lists accumulates a score
    contribution from each appearance, so a chunk both retrievers agree
    on outranks one only one retriever found, even if that one retriever
    ranked it first."""
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            # rank starts at 1, not 0, so the top result contributes
            # 1/(k+1) rather than 1/k, matching the formula as published
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_by_id[chunk.chunk_id] = chunk

    fused = [(chunk_by_id[cid], score) for cid, score in scores.items()]
    fused.sort(key=lambda pair: pair[1], reverse=True)
    return fused


def fuse_and_rerank(
    query: str,
    bm25_results: list[Chunk],
    semantic_results: list[Chunk],
    rerank_top_n: int | None = None,
) -> list[tuple[Chunk, float]]:
    """Convenience wrapper chaining fusion straight into reranking, since
    that is how these two files are actually meant to be used together in
    the real retrieval flow, RRF narrows the corpus down to a fusion-
    ranked shortlist, then the cross-encoder rescoring in reranker.py
    reorders that shortlist using the more accurate signal. Imports
    reranker locally to avoid a module-level dependency for callers who
    only want fusion on its own, fusion.py's own tests, for instance."""
    from crag.retrieval.reranker import rerank_chunks, RERANK_TOP_N

    fused = reciprocal_rank_fusion([bm25_results, semantic_results])
    fused_chunks = [chunk for chunk, _ in fused]
    top_n = rerank_top_n if rerank_top_n is not None else RERANK_TOP_N
    return rerank_chunks(query, fused_chunks, top_n=top_n)