"""
src/crag/graph/refine_knowledge.py

CRAG's decompose-then-recompose step: for each relevant chunk, extracts
only the sentences that are actually relevant to the question, verbatim,
discarding the rest even within an otherwise-good chunk. This is
different from grade_relevance, which judges a whole chunk as relevant
or not, this operates within a chunk, catching the case where a chunk
is mostly on-topic but has a tangential sentence or two dragging in
noise that would otherwise reach generation untouched.

Refinement calls run CONCURRENTLY, same reasoning and same real-world
motivation as grade_relevance's parallelization: sequential per-chunk
calls were a major contributor to a 70-second total query latency
observed in live UI testing. max_workers is capped for the same
rate-limit reason.

Uses claude-haiku, same reasoning as elsewhere, this is an extraction
task against provided text, not open-ended generation.
"""
from concurrent.futures import ThreadPoolExecutor
import contextvars
from pydantic import BaseModel, Field
from crag.ingestion.chunker import Chunk
from crag.observability.anthropic_client import get_traced_client

REFINE_MODEL = "claude-haiku-4-5-20251001"
MAX_CONCURRENT_REFINEMENTS = 8

_client = get_traced_client()


class RefinedChunk(BaseModel):
    chunk_id: str
    refined_text: str = Field(description="Only the relevant sentences, verbatim, in original order; empty string if nothing survived")


REFINE_TOOL = {
    "name": "record_refined_text",
    "description": "Record only the parts of this passage relevant to the question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "refined_text": {
                "type": "string",
                "description": "The relevant sentences or clauses, verbatim from the passage, in their original order, concatenated. Empty string if nothing in the passage is relevant.",
            },
        },
        "required": ["refined_text"],
    },
}

REFINE_SYSTEM_PROMPT = """You refine a passage down to only the parts relevant to a question,
for a Databricks Unity Catalog documentation assistant.

Extract and return ONLY the sentences or clauses from the passage that directly help answer
the question, verbatim, in their original order. Drop sentences that are off-topic or only
tangentially related, even if they're about the same general subject as the passage.

Do not paraphrase, summarize, or rewrite, return the actual original text with irrelevant
parts removed, not your own restatement of what remains. If nothing in the passage is
relevant, return an empty string rather than forcing something weak through."""


def refine_chunk(question: str, chunk: Chunk) -> RefinedChunk:
    response = _client.messages.create(
        model=REFINE_MODEL,
        max_tokens=600,
        system=REFINE_SYSTEM_PROMPT,
        tools=[REFINE_TOOL],
        tool_choice={"type": "tool", "name": "record_refined_text"},
        messages=[
            {"role": "user", "content": f"Question: {question}\n\nPassage:\n{chunk.text}"}
        ],
    )
    tool_use_block = next(b for b in response.content if b.type == "tool_use")
    return RefinedChunk(chunk_id=chunk.chunk_id, **tool_use_block.input)


def refine_chunks(question: str, chunks: list[Chunk]) -> list[RefinedChunk]:
    """Refines every chunk concurrently via a thread pool. Each task
    gets its OWN copied context, same fix and same reasoning as
    grade_relevance.py's grade_chunks: a single shared Context object
    cannot be entered by more than one thread at once, confirmed as a
    real crash in live testing when this file shared one context
    across all workers. Results preserve input order (futures
    collected in submission order, not completion order), same
    determinism guarantee as before."""
    if not chunks:
        return []
    with ThreadPoolExecutor(max_workers=min(len(chunks), MAX_CONCURRENT_REFINEMENTS)) as executor:
        futures = [
            executor.submit(contextvars.copy_context().run, refine_chunk, question, chunk)
            for chunk in chunks
        ]
        return [f.result() for f in futures]


def recompose(refined_chunks: list[RefinedChunk]) -> str:
    """Joins every non-empty refined_text into the single flat string
    CRAGState.refined_context expects. Chunks that refined down to
    nothing (refined_text == "") are dropped entirely, not joined as
    empty lines, so refined_context never carries visible gaps from
    chunks that contributed nothing."""
    pieces = [rc.refined_text for rc in refined_chunks if rc.refined_text.strip()]
    return "\n\n".join(pieces)