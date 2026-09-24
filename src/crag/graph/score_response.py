"""
src/crag/graph/score_response.py

Inline evaluation scoring, run once per request, distinct from any
offline RAGAS evaluation over a fixed eval set (whether inline
per-request scoring is even the right design here, versus offline-only,
is a real open question worth revisiting, this builds what CRAGState
already implies exists).

context_relevance_score is computed deterministically from
relevance_grades already gathered during grade_relevance, no new call,
matching CRAGState's own comment that it's "reused from
relevance_grades". answer_relevancy_score is the one genuinely new call
this step makes, judging whether the final answer actually addresses
the question asked, independent of whether it's grounded, that's
faithfulness's job, not this one's.
"""
from crag.observability.anthropic_client import get_traced_client

RELEVANCY_MODEL = "claude-haiku-4-5-20251001"

_client = get_traced_client()

RELEVANCY_TOOL = {
    "name": "record_relevancy_score",
    "description": "Record how relevant the answer is to the question asked.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "description": "0.0 to 1.0: how directly and completely the answer addresses the question asked, independent of whether the answer is factually correct or grounded",
            },
        },
        "required": ["score"],
    },
}

RELEVANCY_SYSTEM_PROMPT = """You judge how well an answer addresses the question it was
supposed to answer, for a Databricks Unity Catalog documentation assistant.

Score 1.0 if the answer directly and completely addresses what was asked. Score lower if the
answer is off-topic, evasive, answers a different question than the one asked, or leaves out
something the question clearly asked for. This is about relevance to the question, not
correctness or grounding, a wrong but on-topic answer still scores on relevance here, a correct
but off-topic answer scores low."""


def compute_context_relevance(relevance_grades: list[dict]) -> float:
    """Deterministic, no LLM call: the fraction of retrieved chunks
    that graded relevant. Reuses grade_relevance's own output rather
    than re-judging the same chunks a second time."""
    if not relevance_grades:
        return 0.0
    relevant_count = sum(1 for g in relevance_grades if g["relevant"])
    return relevant_count / len(relevance_grades)


def compute_answer_relevancy(question: str, answer: str) -> float:
    response = _client.messages.create(
        model=RELEVANCY_MODEL,
        max_tokens=200,
        system=RELEVANCY_SYSTEM_PROMPT,
        tools=[RELEVANCY_TOOL],
        tool_choice={"type": "tool", "name": "record_relevancy_score"},
        messages=[{"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"}],
    )
    tool_use_block = next(b for b in response.content if b.type == "tool_use")
    return float(tool_use_block.input.get("score", 0.0))