"""
src/crag/graph/build_graph.py

Assembles CRAGState and every node/edge into a runnable LangGraph
StateGraph.

classify_query fans out to all three retrievers as parallel branches
(three unconditional edges from one node), which merge back at
fuse_rerank once all three complete. This is safe under LangGraph's
state-merging rules because each retriever writes to a different
CRAGState key (chunks_text, chunks_semantic, chunks_graph), no shared
key gets written by two branches in the same step without a reducer,
which is the case that would actually need one.

output_guardrail now has real retry behavior: a failed check loops
back to generate for one more attempt (MAX_GUARDRAIL_RETRIES, defined
in nodes.py), and only routes to fallback_response_node once retries
are exhausted, rather than always proceeding to score_response
regardless of the guardrail's verdict the way it originally did.
"""
from langgraph.graph import StateGraph, END
from crag.schema import CRAGState
from crag.graph.nodes import (
    input_guardrail_node,
    classify_query_node,
    retrieve_text_node,
    retrieve_semantic_node,
    retrieve_graph_node,
    fuse_rerank_node,
    grade_relevance_node,
    web_search_node,
    grade_web_results_node,
    refine_knowledge_node,
    generate_node,
    output_guardrail_node,
    fallback_response_node,
    score_response_node,
)
from crag.graph.edges import (
    route_after_input_guardrail,
    route_after_grading,
    route_after_output_guardrail,
)


def build_graph():
    workflow = StateGraph(CRAGState)

    workflow.add_node("input_guardrail", input_guardrail_node)
    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("retrieve_text", retrieve_text_node)
    workflow.add_node("retrieve_semantic", retrieve_semantic_node)
    workflow.add_node("retrieve_graph", retrieve_graph_node)
    workflow.add_node("fuse_rerank", fuse_rerank_node)
    workflow.add_node("grade_relevance", grade_relevance_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("grade_web_results", grade_web_results_node)
    workflow.add_node("refine_knowledge", refine_knowledge_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("output_guardrail", output_guardrail_node)
    workflow.add_node("fallback_response", fallback_response_node)
    workflow.add_node("score_response", score_response_node)

    workflow.set_entry_point("input_guardrail")

    workflow.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {"blocked": END, "classify_query": "classify_query"},
    )

    workflow.add_edge("classify_query", "retrieve_text")
    workflow.add_edge("classify_query", "retrieve_semantic")
    workflow.add_edge("classify_query", "retrieve_graph")
    workflow.add_edge("retrieve_text", "fuse_rerank")
    workflow.add_edge("retrieve_semantic", "fuse_rerank")
    workflow.add_edge("retrieve_graph", "fuse_rerank")

    workflow.add_edge("fuse_rerank", "grade_relevance")

    workflow.add_conditional_edges(
        "grade_relevance",
        route_after_grading,
        {"refine_knowledge": "refine_knowledge", "web_search": "web_search"},
    )

    workflow.add_edge("web_search", "grade_web_results")
    workflow.add_edge("grade_web_results", "refine_knowledge")

    workflow.add_edge("refine_knowledge", "generate")
    workflow.add_edge("generate", "output_guardrail")

    workflow.add_conditional_edges(
        "output_guardrail",
        route_after_output_guardrail,
        {
            "score_response": "score_response",
            "retry_generate": "generate",
            "fallback": "fallback_response",
        },
    )

    workflow.add_edge("fallback_response", "score_response")
    workflow.add_edge("score_response", END)

    return workflow.compile()