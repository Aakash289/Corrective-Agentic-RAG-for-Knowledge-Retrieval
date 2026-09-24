"""
src/crag/graph/edges.py

Conditional routing functions for build_graph.py's StateGraph. Each
takes CRAGState and returns a string naming which node to run next,
matching LangGraph's add_conditional_edges pattern.
"""
from crag.schema import CRAGState
from crag.graph.nodes import MAX_GUARDRAIL_RETRIES


def route_after_input_guardrail(state: CRAGState) -> str:
    """If injection_flagged, skip straight to the end rather than
    running retrieval and generation on a manipulation attempt, no
    point spending API calls on a request that was never going to get
    a real answer anyway."""
    return "blocked" if state["injection_flagged"] else "classify_query"


def route_after_grading(state: CRAGState) -> str:
    """route is set by grade_relevance_node: "relevant" skips straight
    to refine_knowledge, "irrelevant" or "mixed" trigger the Tavily
    fallback first."""
    if state["route"] == "relevant":
        return "refine_knowledge"
    return "web_search"


def route_after_output_guardrail(state: CRAGState) -> str:
    """REAL retry logic, previously this always proceeded to
    score_response regardless of guardrail_passed, meaning a failed
    check had no consequence. Now: passed goes straight to
    score_response as before. Failed, with retries remaining, loops
    back to generate for one more attempt at the same refined context,
    generation has shown real run-to-run variability in this project
    (confirmed during testing, the same inputs don't always produce
    identical output), so a second attempt is a genuine chance at a
    different, better result, not a guaranteed no-op. Failed, with
    retries exhausted, routes to fallback_response_node instead of
    retrying again or silently letting the bad answer through."""
    if state["guardrail_passed"]:
        return "score_response"
    if state["guardrail_retry_count"] <= MAX_GUARDRAIL_RETRIES:
        return "retry_generate"
    return "fallback"