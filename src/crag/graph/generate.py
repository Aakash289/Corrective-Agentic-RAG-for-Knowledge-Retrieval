"""
src/crag/graph/generate.py

Synthesizes the final answer, the piece nothing in this project has
built until now. Every other file has been retrieval, extraction, or
guardrails, this is where an answer actually gets written from what all
of that assembled.

Two separate calls, not one: generating the answer, then a second,
smaller call asking specifically which chunk_ids the finished answer
actually drew from. This was originally one call producing both the
answer text and a cited_chunk_ids list together, but real testing
showed that combination triggering claude-sonnet-5 to malform its tool
call output, the model would jam something resembling a completely
different XML-style tool-call syntax ("</answer>", "<parameter
name=...>") as literal text inside the answer field, and
cited_chunk_ids would never appear in the actual tool input at all.
Splitting the two concerns into separate calls, each with a single,
simple output shape, avoids whatever about the combination triggered
that failure mode.
"""
from pydantic import BaseModel, Field
from crag.ingestion.chunker import Chunk
from crag.retrieval.source_credibility import credibility_label
from crag.observability.anthropic_client import get_traced_client

GENERATE_MODEL = "claude-sonnet-5"
CITE_MODEL = "claude-haiku-4-5-20251001"  # citation extraction is a much
                                            # simpler judgment call than
                                            # generation itself, the cheaper
                                            # model is the right tool here

_client = get_traced_client()

# Stray fragments this project has seen claude-sonnet-5 occasionally
# append to an otherwise-valid answer string, remnants of a different,
# XML-style tool-call syntax the model sometimes reaches for even when
# the actual tool call itself succeeded. Confirmed recurring: the two-
# call split (see module docstring) fixed the case where this corrupted
# the whole tool input and crashed generation, but doesn't fully stop
# a trailing fragment from riding along inside an otherwise-valid
# answer field. Stripped defensively rather than left visible to users.
_STRAY_ARTIFACT_PATTERNS = ["</answer>", "</invoke>", "<answer>", "<invoke>"]


def _strip_stray_tool_artifacts(text: str) -> str:
    for pattern in _STRAY_ARTIFACT_PATTERNS:
        text = text.replace(pattern, "")
    return text.strip()


class GeneratedAnswer(BaseModel):
    answer: str
    cited_chunk_ids: list[str] = Field(default_factory=list)


ANSWER_TOOL = {
    "name": "record_answer",
    "description": "Record the generated answer text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
        },
        "required": ["answer"],
    },
}

GENERATE_SYSTEM_PROMPT = """You answer questions about Databricks Unity Catalog, using ONLY
the provided context. Each context chunk is labeled with its chunk_id, but you do not need to
reference chunk_ids in your answer itself, just write a clear, direct answer.

Write a clear, direct answer grounded entirely in the context, do not add information from
general knowledge that the context doesn't support, that's exactly what a later faithfulness
check is built to catch. If the context only partially answers the question, answer what it
supports and say plainly what it doesn't cover, rather than filling the gap yourself.

Some context chunks come from the web and are labeled with a source tier: "Official
Databricks/Microsoft documentation", "Databricks Community" (user-generated, not official), or
"Third-party source, not Databricks-affiliated". Prefer official sources when they cover the
same point as a lower-tier one. A third-party or community source can still be used, it may be
accurate, but if a specific claim in your answer relies ONLY on a community or third-party
source with no official source corroborating it, say so explicitly in the answer (for example,
"according to a third-party source" or "per the Databricks Community"), rather than presenting
it with the same unqualified confidence as an officially documented fact.

If graph facts are provided, you may use them to corroborate or add structural context to your
answer."""

CITE_TOOL = {
    "name": "record_citations",
    "description": "Record which context chunks this answer actually relied on.",
    "input_schema": {
        "type": "object",
        "properties": {
            "cited_chunk_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "chunk_ids of context chunks the answer actually drew from, not every chunk it was given",
            },
        },
        "required": ["cited_chunk_ids"],
    },
}

CITE_SYSTEM_PROMPT = """Given an answer and the context chunks it was generated from, identify
which chunk_ids the answer actually relied on. A chunk counts as cited only if its content
genuinely supports something stated in the answer, not just because it was present in context.
Return an empty list if none of the chunks meaningfully contributed."""


def generate_answer(question: str, chunks: list[Chunk], graph_facts: list[dict] | None = None) -> GeneratedAnswer:
    """Generates the answer with one call, then identifies citations
    with a second, separate call. See module docstring for why these
    are split rather than combined.

    Web-sourced chunks (doc_id is a URL, set that way in
    tavily_search.py) get a credibility label prepended, "Official
    Databricks/Microsoft documentation", "Databricks Community", or
    "Third-party source", so the model sees the tier alongside the
    content itself rather than treating a random blog and Databricks's
    own docs identically, the gap this whole labeling exists to close.
    Internal chunks (doc_id like "Document_3") get no such label, that
    distinction doesn't apply to the corpus's own documents."""
    context_lines = []
    for c in chunks:
        if c.doc_id.startswith("http"):
            label = credibility_label(c.doc_id)
            context_lines.append(f"[chunk_id: {c.chunk_id}] [{c.section_heading}] [Source: {label}]\n{c.text}")
        else:
            context_lines.append(f"[chunk_id: {c.chunk_id}] [{c.section_heading}]\n{c.text}")
    context_text = "\n\n".join(context_lines)

    graph_text = ""
    if graph_facts:
        graph_lines = "\n".join(
            f"- {f['source']} --{f['relation']}--> {f['target']} (evidence: {f['evidence']})"
            for f in graph_facts
        )
        graph_text = f"\n\nRelated facts from the knowledge graph (for context, not citation):\n{graph_lines}"

    answer_text = None
    last_malformed_input = None
    for attempt in range(3):
        # Retried, not just raised-once, because this exact failure mode,
        # a forced tool call coming back with a required field missing
        # or the input dict entirely empty, has now shown up three times
        # in this project (target_type in graph extraction, reason in
        # the output guardrail, answer here), a real, recurring
        # characteristic of claude-sonnet-5's forced tool calls, not a
        # one-off fluke worth crashing over on the first occurrence.
        # answer specifically is too central to default around the way
        # reason was, an empty string masquerading as a real answer
        # would be worse than retrying for a genuine one.
        answer_response = _client.messages.create(
            model=GENERATE_MODEL,
            max_tokens=1000,
            system=GENERATE_SYSTEM_PROMPT,
            tools=[ANSWER_TOOL],
            tool_choice={"type": "tool", "name": "record_answer"},
            messages=[
                {"role": "user", "content": f"Context:\n{context_text}{graph_text}\n\nQuestion: {question}"}
            ],
        )
        answer_block = next(b for b in answer_response.content if b.type == "tool_use")
        answer_input = answer_block.input
        if "answer" in answer_input:
            answer_text = _strip_stray_tool_artifacts(answer_input["answer"])
            break
        last_malformed_input = answer_input

    if answer_text is None:
        # All 3 attempts came back malformed, genuinely worth raising
        # loudly at this point rather than retrying indefinitely or
        # silently degrading, three consecutive failures on the same
        # simple single-field call is worth surfacing as a real problem,
        # not something to keep quietly retrying past.
        raise ValueError(f"generate_answer: 'answer' missing after 3 attempts, last input: {last_malformed_input!r}")

    cite_response = _client.messages.create(
        model=CITE_MODEL,
        max_tokens=300,
        system=CITE_SYSTEM_PROMPT,
        tools=[CITE_TOOL],
        tool_choice={"type": "tool", "name": "record_citations"},
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nAnswer:\n{answer_text}",
            }
        ],
    )
    cite_block = next(b for b in cite_response.content if b.type == "tool_use")
    cited_chunk_ids = cite_block.input.get("cited_chunk_ids", [])

    return GeneratedAnswer(answer=answer_text, cited_chunk_ids=cited_chunk_ids)