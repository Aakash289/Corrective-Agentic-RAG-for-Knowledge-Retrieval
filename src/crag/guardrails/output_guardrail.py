"""
src/crag/guardrails/output_guardrail.py

Runs after generation, before the answer reaches the user. Checks
whether the generated answer is actually grounded in the retrieved
chunks it was supposedly built from, not whether it's a good answer in
some general sense, faithfulness to the provided context specifically.
This is a runtime gate, not the same thing as the RAGAS faithfulness
metric used in offline evaluation (src/crag/eval), this blocks one bad
answer at generation time, RAGAS scores the system's aggregate quality
across a whole eval set after the fact. They check a similar underlying
property but serve different purposes and run at different points.

Two functions here, doing genuinely different jobs, not two versions of
the same check: check_output is a fast, holistic pass/fail, one call,
used to gate the retry logic in output_guardrail_node, it needs to be
cheap since it runs on every single generation. compute_faithfulness_score
is a slower, more detailed diagnostic: it decomposes the answer into
individual factual claims and judges each one separately, producing a
genuine 0-1 fraction (claims supported / claims total) rather than a
boolean. This replaces the earlier boolean-derived faithfulness_score
(1.0 or 0.0 from check_output's verdict) with a real graded number, at
the real cost of one additional API call per generation.

Same claude-haiku model as input_guardrail.py, same reasoning, both are
judgment calls against provided evidence, not open-ended generation.
"""
from pydantic import BaseModel, Field
from crag.guardrails.guardrail_schema import GuardrailResult
from crag.ingestion.chunker import Chunk
from crag.observability.anthropic_client import get_traced_client

GUARDRAIL_MODEL = "claude-haiku-4-5-20251001"

_client = get_traced_client()

OUTPUT_GUARDRAIL_TOOL = {
    "name": "record_output_check",
    "description": "Record whether this answer is grounded in the provided context.",
    "input_schema": {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["passed", "reason"],
    },
}

OUTPUT_GUARDRAIL_SYSTEM_PROMPT = """You check whether a generated answer is actually
supported by the context it was given, for a Databricks Unity Catalog documentation assistant.

Fail the answer (passed: false) if it contains a specific factual claim, a privilege name, a
relationship between two objects, a concrete step, that is not stated in or reasonably implied
by the provided context. A claim the context doesn't address at all, filled in from general
knowledge instead, is exactly the failure this check exists to catch.

Pass the answer if every concrete claim it makes traces back to something the context actually
says. General framing language, transitions, or a reasonable summary of what the context states
is fine and should not cause a failure, this check is about invented facts, not writing style.

A synthesized comparison or contrast built from two or more individually-supported facts is NOT
an invented claim, even though it isn't a verbatim sentence anywhere in the context. If the
context separately states "SELECT reads data" and "MODIFY writes data," an answer saying "SELECT
governs read access while MODIFY governs write access" is a valid, grounded synthesis, not
fabrication, confirmed as a real failure mode this check was incorrectly rejecting: check each
component fact underneath a comparison for support individually, don't fail the whole statement
just because the comparative framing itself isn't a direct quote."""


def check_output(answer: str, context_chunks: list[Chunk]) -> GuardrailResult:
    context_text = "\n\n".join(f"[{c.section_heading}]\n{c.text}" for c in context_chunks)
    response = _client.messages.create(
        model=GUARDRAIL_MODEL,
        max_tokens=300,
        system=OUTPUT_GUARDRAIL_SYSTEM_PROMPT,
        tools=[OUTPUT_GUARDRAIL_TOOL],
        tool_choice={"type": "tool", "name": "record_output_check"},
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nGenerated answer:\n{answer}",
            }
        ],
    )
    tool_use_block = next(b for b in response.content if b.type == "tool_use")
    return GuardrailResult(**tool_use_block.input)


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool


FAITHFULNESS_TOOL = {
    "name": "record_claim_verdicts",
    "description": "Break the answer into individual factual claims and judge whether each is supported by the context.",
    "input_schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "supported": {"type": "boolean"},
                    },
                    "required": ["claim", "supported"],
                },
            },
        },
        "required": ["claims"],
    },
}

FAITHFULNESS_SYSTEM_PROMPT = """You break a generated answer down into individual, checkable
factual claims, then judge whether each one is genuinely supported by the given context, for a
Databricks Unity Catalog documentation assistant.

A claim is one discrete factual assertion, a privilege name, a relationship, a specific step or
limit, not every sentence, a single sentence can contain multiple claims and should be split
accordingly. Do not extract general framing, transitions, or hedging language ("the context
also mentions...") as separate claims, those aren't factual assertions to check.

For each claim, supported is true only if the context genuinely states or directly implies it,
not merely that the topic is related. If the answer makes no real factual claims at all (for
example, a message saying the question can't be answered from the available information),
return an empty claims list."""


def compute_faithfulness_score(answer: str, context_chunks: list[Chunk]) -> tuple[float, str]:
    """Returns (score, reason). score is claims_supported / claims_total,
    a genuine graded 0-1 score. reason is built from the actual
    per-claim breakdown, not a separate holistic judgment, listing
    which specific claims (if any) weren't supported, or confirming
    all were.

    This function is now the SOLE basis for output_guardrail_node's
    pass/fail decision, check_output (below) is kept in this file but
    no longer called from the live gating path. Across three separate
    real runs, this decomposed, per-claim judge and check_output's
    holistic judge disagreed, and this one was right every time,
    check_output rejected genuinely well-grounded answers (a valid
    synthesized comparison, a valid logical restatement of "must also
    have X" as "X is a prerequisite") that this function correctly
    scored as fully or mostly faithful. Judging claim by claim appears
    to be a meaningfully more reliable signal than one holistic
    yes/no verdict for this kind of reasoning-heavy content, worth
    trusting the more careful judge over the coarser one once that
    pattern repeated three times, not a one-off tolerance.

    An answer with no extractable claims (an honest "I don't have
    enough information" response, say) is treated as vacuously
    faithful, 1.0, with no unsupported claims to report."""
    context_text = "\n\n".join(f"[{c.section_heading}]\n{c.text}" for c in context_chunks)
    response = _client.messages.create(
        model=GUARDRAIL_MODEL,
        max_tokens=800,
        system=FAITHFULNESS_SYSTEM_PROMPT,
        tools=[FAITHFULNESS_TOOL],
        tool_choice={"type": "tool", "name": "record_claim_verdicts"},
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nAnswer:\n{answer}",
            }
        ],
    )
    tool_use_block = next(b for b in response.content if b.type == "tool_use")
    raw_claims = tool_use_block.input.get("claims", [])

    if not raw_claims:
        return 1.0, "No specific factual claims to verify."

    claims = [ClaimVerdict(**c) for c in raw_claims]
    supported_count = sum(1 for c in claims if c.supported)
    score = supported_count / len(claims)

    if score == 1.0:
        reason = f"All {len(claims)} claim(s) supported by the provided context."
    else:
        unsupported = [c.claim for c in claims if not c.supported]
        reason = (
            f"{supported_count} of {len(claims)} claim(s) supported. "
            f"Unsupported: {'; '.join(unsupported)}"
        )
    return score, reason