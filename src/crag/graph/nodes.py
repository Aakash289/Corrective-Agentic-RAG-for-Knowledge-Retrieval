"""
src/crag/graph/nodes.py

LangGraph node functions. Each takes CRAGState and returns a partial
state update dict (LangGraph's standard pattern), and appends its own
name to nodes_executed for tracing.
"""
from crag.schema import CRAGState
from crag.guardrails.input_guardrail import check_input
from crag.guardrails.output_guardrail import compute_faithfulness_score
from crag.ingestion.chunker import Chunk
from crag.ingestion.bm25_index import query_bm25
from crag.ingestion.embed_index import query_similar
from crag.retrieval.fusion import reciprocal_rank_fusion
from crag.retrieval.reranker import rerank_chunks
from crag.retrieval.tavily_search import web_search_fallback
from crag.retrieval.graph_retrieve import query_graph_facts, get_graph_driver
from crag.graph.classify_query import classify_query
from crag.graph.grade_relevance import grade_chunks, decide_route
from crag.graph.grade_web_results import grade_web_chunks
from crag.graph.refine_knowledge import refine_chunks, recompose
from crag.graph.generate import generate_answer
from crag.graph.score_response import compute_context_relevance, compute_answer_relevancy


def _chunk_to_dict(chunk: Chunk, score: float | None = None) -> dict:
    d = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "section_heading": chunk.section_heading,
        "text": chunk.text,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
    }
    if score is not None:
        d["score"] = score
    return d


def _dict_to_chunk(d: dict) -> Chunk:
    return Chunk(
        chunk_id=d["chunk_id"],
        doc_id=d["doc_id"],
        section_heading=d["section_heading"],
        text=d["text"],
        page_start=d.get("page_start", 0),
        page_end=d.get("page_end", 0),
    )


def input_guardrail_node(state: CRAGState) -> dict:
    result = check_input(state["question"])
    return {
        "injection_flagged": not result.passed,
        "nodes_executed": ["input_guardrail"],
    }


def classify_query_node(state: CRAGState) -> dict:
    targets = classify_query(state["question"])
    return {"retrieve_targets": targets, "nodes_executed": ["classify_query"]}


def retrieve_text_node(state: CRAGState) -> dict:
    if "text" not in state["retrieve_targets"]:
        return {"chunks_text": [], "nodes_executed": ["retrieve_text"]}
    from crag.ingestion.bm25_index import get_corpus_bm25_index
    bm25, bm25_chunks = get_corpus_bm25_index()
    results = query_bm25(state["question"], bm25, bm25_chunks, n_results=10)
    return {
        "chunks_text": [_chunk_to_dict(c, score) for c, score in results],
        "nodes_executed": ["retrieve_text"],
    }


def retrieve_semantic_node(state: CRAGState) -> dict:
    if "semantic" not in state["retrieve_targets"]:
        return {"chunks_semantic": [], "nodes_executed": ["retrieve_semantic"]}
    raw = query_similar(state["question"], n_results=10)
    chunks = [
        {"chunk_id": cid, "text": doc, **meta}
        for cid, doc, meta in zip(raw["ids"][0], raw["documents"][0], raw["metadatas"][0])
    ]
    return {"chunks_semantic": chunks, "nodes_executed": ["retrieve_semantic"]}


def retrieve_graph_node(state: CRAGState) -> dict:
    if "graph" not in state["retrieve_targets"]:
        return {"chunks_graph": [], "nodes_executed": ["retrieve_graph"]}
    facts = query_graph_facts(state["question"], get_graph_driver())
    return {"chunks_graph": facts, "nodes_executed": ["retrieve_graph"]}


def fuse_rerank_node(state: CRAGState) -> dict:
    text_chunks = [_dict_to_chunk(d) for d in state["chunks_text"]]
    semantic_chunks = [_dict_to_chunk(d) for d in state["chunks_semantic"]]

    fused = reciprocal_rank_fusion([text_chunks, semantic_chunks])
    fused_chunks = [c for c, _ in fused]
    reranked = rerank_chunks(state["question"], fused_chunks)

    return {
        "reranked_chunks": [_chunk_to_dict(c, score) for c, score in reranked],
        "nodes_executed": ["fuse_rerank"],
    }


def grade_relevance_node(state: CRAGState) -> dict:
    reranked = [_dict_to_chunk(d) for d in state["reranked_chunks"]]
    grades = grade_chunks(state["question"], reranked)
    route = decide_route(grades)
    return {
        "relevance_grades": [g.model_dump() for g in grades],
        "route": route,
        "nodes_executed": ["grade_relevance"],
    }


def web_search_node(state: CRAGState) -> dict:
    results = web_search_fallback(state["question"])
    return {
        "web_results": [_chunk_to_dict(c) for c in results],
        "nodes_executed": ["web_search"],
    }


def grade_web_results_node(state: CRAGState) -> dict:
    web_chunks = [_dict_to_chunk(d) for d in state["web_results"]]
    grades = grade_web_chunks(state["question"], web_chunks)
    return {
        "web_grades": [g.model_dump() for g in grades],
        "nodes_executed": ["grade_web_results"],
    }


def refine_knowledge_node(state: CRAGState) -> dict:
    """REAL, now with a real efficiency fix: only refines chunks that
    were actually graded relevant, plus any web results graded
    relevant. Previously this refined EVERY reranked chunk regardless
    of grade_relevance's verdict, wasting a full refinement call on
    chunks already known to be irrelevant, confirmed directly in live
    UI testing (8 chunks graded, only 6 relevant, but all 8 got
    refined anyway). On the "irrelevant" route, relevant_internal_ids
    ends up empty (grade_relevance found nothing relevant internally),
    so this correctly skips ALL internal refinement calls in that case,
    a much bigger saving than the "relevant"/"mixed" case, and exactly
    right: there was never anything internal worth refining if
    everything was already graded irrelevant."""
    relevant_internal_ids = {g["chunk_id"] for g in state["relevance_grades"] if g["relevant"]}
    reranked = [
        _dict_to_chunk(d) for d in state["reranked_chunks"] if d["chunk_id"] in relevant_internal_ids
    ]

    relevant_web_ids = {g["chunk_id"] for g in state["web_grades"] if g["relevant"]}
    relevant_web_chunks = [
        _dict_to_chunk(d) for d in state["web_results"] if d["chunk_id"] in relevant_web_ids
    ]

    all_chunks = reranked + relevant_web_chunks
    refined = refine_chunks(state["question"], all_chunks)
    return {
        "refined_context": recompose(refined),
        "refined_chunks": [
            {"chunk_id": rc.chunk_id, "refined_text": rc.refined_text}
            for rc in refined
            if rc.refined_text.strip()
        ],
        "nodes_executed": ["refine_knowledge"],
    }


def _build_generation_context(state: CRAGState) -> list[Chunk]:
    """The single source of truth for what context an answer was
    actually generated from, refined_chunks merged with metadata from
    either reranked_chunks or web_results, whichever the chunk_id
    belongs to, falling back to reranked_chunks alone if refinement
    produced nothing. Shared between generate_node and
    output_guardrail_node specifically because they used to build this
    independently and drifted apart: output_guardrail_node only ever
    checked reranked_chunks, the internal-only pre-refinement set,
    while generate_node actually generated from this richer merged
    set. On a "relevant" route those two happened to be nearly
    identical, so the gap was invisible, but on "mixed" or
    "irrelevant" routes, where Tavily's results get merged in via
    refine_knowledge_node, the guardrail was checking the answer
    against a strict subset of what it was actually built from,
    confirmed directly: it repeatedly flagged legitimately web-sourced
    claims, including an honest, explicitly-attributed third-party
    citation, as "invented" simply because that content was never in
    the narrower context it was shown. Both nodes now call this same
    function, so they can never see a different picture of "what
    counts as this answer's context" again."""
    reranked_by_id = {d["chunk_id"]: d for d in state["reranked_chunks"]}
    web_by_id = {d["chunk_id"]: d for d in state["web_results"]}

    if state["refined_chunks"]:
        chunks = []
        for rc in state["refined_chunks"]:
            original = reranked_by_id.get(rc["chunk_id"]) or web_by_id.get(rc["chunk_id"])
            if original is None:
                continue
            merged = dict(original)
            merged["text"] = rc["refined_text"]
            chunks.append(_dict_to_chunk(merged))
        return chunks
    return [_dict_to_chunk(d) for d in state["reranked_chunks"]]


def generate_node(state: CRAGState) -> dict:
    generation_chunks = _build_generation_context(state)
    result = generate_answer(state["question"], generation_chunks, graph_facts=state["chunks_graph"])
    return {
        "answer": result.answer,
        "context_sources": state["retrieve_targets"],
        "citations": [{"chunk_id": cid} for cid in result.cited_chunk_ids],
        "nodes_executed": ["generate"],
    }


MAX_GUARDRAIL_RETRIES = 1


FAITHFULNESS_PASS_THRESHOLD = 0.7  # not 1.0: real testing showed even a
                                     # genuinely well-grounded answer can
                                     # have one borderline claim a strict
                                     # judge might flag, requiring every
                                     # single claim to pass would reproduce
                                     # the same over-strict failure mode
                                     # this whole redesign was fixing.
                                     # 0.7 tolerates a minor edge case
                                     # while still catching an answer
                                     # that's genuinely mostly fabricated.


def output_guardrail_node(state: CRAGState) -> dict:
    """REAL. Checks against _build_generation_context(state), the SAME
    context generate_node actually used, not reranked_chunks alone,
    a real, confirmed bug fixed here: on any "mixed"/"irrelevant" route
    where web results get merged in during refinement, an earlier
    version checked a strict subset of the real context, guaranteed to
    flag legitimately web-sourced claims as unsupported.

    guardrail_passed is now derived from compute_faithfulness_score's
    graded result (score >= FAITHFULNESS_PASS_THRESHOLD), not from a
    separate check_output call. That call is REMOVED from this node
    entirely, not just deprioritized: across three separate real runs,
    check_output's holistic judgment rejected answers that
    compute_faithfulness_score's per-claim judgment correctly scored as
    fully or mostly faithful (a valid synthesized comparison, content
    legitimately sourced from web results check_output couldn't see
    before the context-mismatch fix above, and a valid logical
    restatement of a stated requirement as a "prerequisite"). A
    repeated pattern across independent runs, not a one-off worth
    tolerating, so the more reliable, decomposed judge now does both
    the scoring and the gating, one Anthropic call instead of two,
    faster and more accurate at the same time.

    guardrail_reason now comes directly from the claim-level breakdown,
    real information about which specific claim(s) failed when it did,
    not a separately-computed holistic explanation that could disagree
    with the actual pass/fail decision."""
    context_chunks = _build_generation_context(state)
    faithfulness_score, reason = compute_faithfulness_score(state["answer"], context_chunks)
    passed = faithfulness_score >= FAITHFULNESS_PASS_THRESHOLD

    valid_chunk_ids = {d["chunk_id"] for d in state["reranked_chunks"]}
    valid_chunk_ids |= {d["chunk_id"] for d in state["web_results"]}
    citation_valid = all(c["chunk_id"] in valid_chunk_ids for c in state["citations"])

    retry_count = state["guardrail_retry_count"]
    if not passed:
        retry_count += 1

    return {
        "guardrail_passed": passed,
        "guardrail_reason": reason,
        "citation_valid": citation_valid,
        "faithfulness_score": faithfulness_score,
        "guardrail_retry_count": retry_count,
        "nodes_executed": ["output_guardrail"],
    }


def fallback_response_node(state: CRAGState) -> dict:
    return {
        "answer": (
            "I wasn't able to produce a reliably grounded answer to this "
            "question from the available information. This could mean the "
            "documentation doesn't cover this topic in enough detail, or "
            "the question may need to be rephrased. Please try asking "
            "again with more specific wording, or consult the "
            "documentation directly."
        ),
        "citations": [],
        "nodes_executed": ["fallback_response"],
    }


def score_response_node(state: CRAGState) -> dict:
    context_relevance = compute_context_relevance(state["relevance_grades"])
    answer_relevancy = compute_answer_relevancy(state["question"], state["answer"])
    return {
        "context_relevance_score": context_relevance,
        "answer_relevancy_score": answer_relevancy,
        "nodes_executed": ["score_response"],
    }