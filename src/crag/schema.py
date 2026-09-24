"""
src/crag/schema.py

CRAGState: the LangGraph state every node reads from and writes to.
ChunkMetadata: one canonical record per chunk, referenced by chunk_id
from ChromaDB, the BM25 index, and Neo4j rather than duplicated in each.

Chunk and Section (the chunker's own intermediate types) stay in
chunker.py, not here, to avoid a circular import between ingestion
and schema for types nothing outside chunking needs.
"""
import operator
from dataclasses import dataclass
from datetime import date
from typing import Annotated, TypedDict


class CRAGState(TypedDict):
    question: str                        # original user question
    rewritten_queries: list[str]         # query rewrites, used for retrieval and Tavily
    injection_flagged: bool              # input_guardrail: the only thing it blocks on
    retrieve_targets: list[str]          # classify_query: which retrievers to run
    chunks_text: list[dict]              # retrieve_text output
    chunks_semantic: list[dict]          # retrieve_semantic output
    chunks_graph: list[dict]             # retrieve_graph output, kept separate from fuse_rerank
    reranked_chunks: list[dict]          # fuse_rerank: RRF-fused, BAAI/bge-reranker-base reranked
    relevance_grades: list[dict]         # grade_relevance output, per chunk
    route: str                           # relevant, irrelevant, or mixed
    web_results: list[dict]              # web_search (Tavily) output
    web_grades: list[dict]               # grade_web_results output
    refined_context: str                 # refine_knowledge output, flat text (display/debug, not generation's source)
    refined_chunks: list[dict]           # refine_knowledge output, chunk_id preserved per refined piece, this is what generate reads from
    answer: str                          # generate output
    faithfulness_score: float            # output_guardrail, RAGAS Faithfulness, reused by score_response
    citation_valid: bool                 # output_guardrail, deterministic set lookup
    guardrail_passed: bool               # output_guardrail routing flag
    guardrail_reason: str                # output_guardrail: check_output's actual reason, previously discarded, confirmed costly to lose during real debugging
    guardrail_retry_count: int           # output_guardrail: incremented each time a check fails, caps retries before falling back to a safe response instead of looping or letting an ungrounded answer through
    context_relevance_score: float       # score_response, reused from relevance_grades
    answer_relevancy_score: float        # score_response, the one new call it makes
    context_sources: list[str]           # generate: which retrievers plus web contributed
    citations: list[dict]                # generate: chunk ids or web URLs actually cited
    nodes_executed: Annotated[list[str], operator.add]   # trace list, needs a reducer


@dataclass
class ChunkMetadata:
    chunk_id: str            # content hash, stable primary key
    doc_id: str               # source PDF identifier
    doc_title: str            # human readable page title, for citation display
    section_heading: str      # from structure-aware chunking
    page_start: int
    page_end: int
    extraction_method: str    # "text_layer" or "ocr", a quality flag
    source_doc_hash: str      # hash of the source PDF file, the dedup key
    ingested_at: date         # as_of date, flows into every graph node/edge this chunk feeds