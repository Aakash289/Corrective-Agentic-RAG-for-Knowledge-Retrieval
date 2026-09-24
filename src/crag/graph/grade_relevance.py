"""
src/crag/graph/grade_relevance.py

The core CRAG step: grading each reranked chunk as relevant or
irrelevant to the question, then aggregating those per-chunk grades
into route, the decision that actually drives the graph's corrective
behavior, "relevant" skips straight to refine_knowledge, "irrelevant"
or "mixed" triggers the Tavily web search fallback in edges.py's
route_after_grading.

Grades run CONCURRENTLY, not sequentially, one call per chunk still,
same reasoning as before (a more reliable signal than batching, no
anchoring on an earlier chunk's obvious verdict), but real UI testing
showed 8 sequential calls contributing heavily to a 70-second total
query latency. Running them in parallel doesn't change the per-chunk
call pattern or the quality tradeoff that pattern was chosen for, it
just lets independent calls happen concurrently instead of one after
another. max_workers is capped, not unbounded, to avoid slamming the
API with a burst large enough to risk rate limiting on a big chunk set.
"""
from concurrent.futures import ThreadPoolExecutor
import contextvars
from pydantic import BaseModel, Field
from crag.ingestion.chunker import Chunk
from crag.observability.anthropic_client import get_traced_client

GRADE_MODEL = "claude-haiku-4-5-20251001"
MAX_CONCURRENT_GRADES = 8

_client = get_traced_client()


class RelevanceGrade(BaseModel):
    chunk_id: str
    relevant: bool
    reason: str = Field(description="Brief explanation of the relevance judgment")


GRADE_TOOL = {
    "name": "record_grade",
    "description": "Record whether this chunk is relevant to the question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "relevant": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["relevant", "reason"],
    },
}

GRADE_SYSTEM_PROMPT = """You judge whether a retrieved passage is relevant to a question,
for a Databricks Unity Catalog documentation assistant.

A passage is relevant if it contains information that would help answer the question,
even partially, even if it doesn't fully answer it on its own. A passage is irrelevant if
it's about a different topic entirely, or only shares surface-level keyword overlap with
the question without actually addressing what's being asked.

Be reasonably generous, a passage that provides useful background or a related concept
counts as relevant, this isn't asking for a passage that alone fully answers the question,
only whether it genuinely contributes toward one."""


def grade_chunk_relevance(question: str, chunk: Chunk) -> RelevanceGrade:
    """Grades a single chunk. Called concurrently, once per chunk, by
    grade_chunks below."""
    response = _client.messages.create(
        model=GRADE_MODEL,
        max_tokens=300,
        system=GRADE_SYSTEM_PROMPT,
        tools=[GRADE_TOOL],
        tool_choice={"type": "tool", "name": "record_grade"},
        messages=[
            {"role": "user", "content": f"Question: {question}\n\nPassage:\n{chunk.text}"}
        ],
    )
    tool_use_block = next(b for b in response.content if b.type == "tool_use")
    return RelevanceGrade(chunk_id=chunk.chunk_id, **tool_use_block.input)


def grade_chunks(question: str, chunks: list[Chunk]) -> list[RelevanceGrade]:
    """Grades every chunk concurrently via a thread pool. Each task gets
    its OWN copied context (contextvars.copy_context() called once PER
    submission, not once for the whole batch), a single Context object
    cannot be entered by more than one thread at the same time, sharing
    one across all workers caused a real, confirmed crash in live UI
    testing: RuntimeError: cannot enter context, ... is already entered,
    the moment two grading calls ran concurrently. Each fresh copy still
    correctly sees the same underlying UsageSummary instance (copying a
    context copies the reference a ContextVar points to, not a deep copy
    of the object), so usage tracking still works, just without threads
    colliding over a single shared context."""
    if not chunks:
        return []
    with ThreadPoolExecutor(max_workers=min(len(chunks), MAX_CONCURRENT_GRADES)) as executor:
        futures = [
            executor.submit(contextvars.copy_context().run, grade_chunk_relevance, question, chunk)
            for chunk in chunks
        ]
        return [f.result() for f in futures]


TOP_K_FOR_ROUTING = 4  # only the top-K reranked chunks vote on route, not
                         # the full shortlist. Confirmed necessary directly:
                         # across an entire real debugging session, the one
                         # chunk that actually defined the answer to a
                         # specific question was graded relevant in EVERY
                         # single run, always near the top of the reranked
                         # order, yet the question still routed to "mixed"
                         # repeatedly because 4-5 lower-ranked, tangentially
                         # related chunks (mentioning the same keywords
                         # without explaining them) dragged the full-set
                         # majority below 50%. reranked_chunks is already
                         # ordered best-first by the cross-encoder, that
                         # ordering IS the relevance signal, a chunk in
                         # position 8 diluting the vote as much as the
                         # chunk in position 1 wastes the reranker's own
                         # judgment rather than using it.


def decide_route(grades: list[RelevanceGrade]) -> str:
    """Aggregates per-chunk grades into one route decision, majority-
    based over only the TOP_K_FOR_ROUTING highest-ranked grades, not
    the full grades list. grades arrives in the same order as
    reranked_chunks (grade_chunks preserves input order regardless of
    which concurrent call finishes first), which is already sorted
    best-first by the reranker, so grades[:TOP_K_FOR_ROUTING] is
    genuinely the most relevant subset, not an arbitrary slice.

    This was originally a majority over the FULL grades list (see the
    project history: before that, an even stricter all-or-nothing rule
    was replaced with full-list majority after real testing showed 80%
    and 62.5%-relevant runs both getting flagged "mixed"). Full-list
    majority fixed that but introduced a different, subtler problem:
    a handful of low-ranked, tangentially-related chunks (mentioning
    the right keywords without actually explaining them) could still
    outvote a small number of highly relevant top-ranked chunks,
    confirmed directly on a real, repeatedly-tested question where the
    single correct chunk graded relevant on every run but kept losing
    the vote to noise further down the list. Restricting the vote to
    the top-K chunks the reranker already identified as most relevant
    uses that ranking signal instead of discarding it.

    "irrelevant" when none of the top-K pass, "relevant" when more
    than half of the top-K do, "mixed" for a genuine split within the
    top-K. An empty grades list is treated as "irrelevant" by vacuous
    default, there's nothing to trust either way."""
    if not grades:
        return "irrelevant"
    top_grades = grades[:TOP_K_FOR_ROUTING]
    relevant_count = sum(1 for g in top_grades if g.relevant)
    if relevant_count == 0:
        return "irrelevant"
    if relevant_count > len(top_grades) / 2:
        return "relevant"
    return "mixed"