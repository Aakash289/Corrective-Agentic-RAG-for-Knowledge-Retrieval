"""
src/crag/graph/classify_query.py

Real routing logic for classify_query_node: decides which retrievers
are worth running for a given question. text and semantic always run,
they're cheap and reliably useful. graph runs only when the question
plausibly benefits from relationship or hierarchy information, since
retrieve_graph_node's entity-extraction-plus-Cypher-traversal has its
own real cost, an extra Haiku call and a Neo4j round trip, not worth
paying on every query. "How do I connect to a SQL warehouse" doesn't
need a graph traversal, "what privileges are required to create a
table" does, that's a hierarchy/requirement question.
"""
from crag.observability.anthropic_client import get_traced_client

CLASSIFY_MODEL = "claude-haiku-4-5-20251001"

_client = get_traced_client()

CLASSIFY_TOOL = {
    "name": "record_targets",
    "description": "Record whether graph-based retrieval would help answer this question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "use_graph": {
                "type": "boolean",
                "description": "True if the question is about relationships, hierarchy, or requirements between entities (what does X require, what is Y part of, what does Z enable). False for a specific fact, definition, or procedure that doesn't hinge on entity relationships.",
            },
        },
        "required": ["use_graph"],
    },
}

CLASSIFY_SYSTEM_PROMPT = """You decide whether a question about Databricks Unity Catalog would
benefit from graph-based relationship retrieval, in addition to standard document retrieval,
which always runs regardless of your answer.

Set use_graph to true for questions about relationships, hierarchy, or requirements between
entities: what privileges does X require, what is Y part of, what does Z enable, how do
concepts relate to each other. Set it to false for a specific fact, definition, or step-by-step
procedure that doesn't hinge on understanding relationships between entities, "how do I connect
to a SQL warehouse" is a procedure question, not a relationship question."""


def classify_query(question: str) -> list[str]:
    """text and semantic always run, unconditionally, they're the
    reliable baseline. graph is added only when genuinely warranted."""
    response = _client.messages.create(
        model=CLASSIFY_MODEL,
        max_tokens=200,
        system=CLASSIFY_SYSTEM_PROMPT,
        tools=[CLASSIFY_TOOL],
        tool_choice={"type": "tool", "name": "record_targets"},
        messages=[{"role": "user", "content": f"Question: {question}"}],
    )
    tool_use_block = next(b for b in response.content if b.type == "tool_use")
    targets = ["text", "semantic"]
    if tool_use_block.input.get("use_graph", False):
        targets.append("graph")
    return targets