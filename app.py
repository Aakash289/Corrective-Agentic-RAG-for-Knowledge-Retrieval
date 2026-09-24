"""
app.py

Gradio front end for the Corrective Agentic RAG system. Wires the
compiled LangGraph pipeline (crag.graph.build_graph) directly, no
shortcuts, this UI runs the real system, the same one every smoke test
in this project has exercised.

Layout: a centered header (project name, byline), a persistent left
column (about the project, about the corpus) visible on every tab, and
three tabs: Ask a Question (the main Q&A flow), Evaluation (ragas
metrics, read from the last offline eval run, with a button to trigger
a fresh one), and Pipeline Trace (a step-by-step breakdown of the most
recent query's run through the graph).

Cost and latency are computed locally per query (crag.observability.
usage_tracker), not pulled from LangSmith, LangSmith ingests traces
asynchronously and isn't a reliable source for a number the UI needs
immediately after a query completes. LangSmith remains valuable for
deeper trace exploration outside this app, in the LangSmith dashboard
itself.
"""
import time
import pandas as pd
import gradio as gr

from crag.graph.build_graph import build_graph
from crag.observability.usage_tracker import start_tracking, get_usage_summary

PROJECT_NAME = "Corrective Agentic RAG for Knowledge Retrieval"
AUTHOR_NAME = "Aakash Bhanushali"
EVAL_CSV_PATH = "data/eval_results.csv"

_graph = build_graph()

# ---------------------------------------------------------------------------
# Styling: monochrome body with a single blue accent, plus three reserved
# status colors (green/amber/red) used only for the route badge and
# guardrail indicators, functional signaling, not decoration.
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root {
    --bg: #B8DED0;
    --bg-card: #FFFFFF;
    --text: #1F2D3D;
    --text-muted: #5C7080;
    --border: #A9E5D9;
    --accent: #14B8A6;
    --accent-hover: #0D9488;
    --support: #3B82F6;
    --status-green: #16A34A;
    --status-amber: #D97706;
    --status-red: #DC2626;
    --header-bg: #1F2D3D;
}

html, body {
    background: var(--bg) !important;
    margin: 0;
    padding: 0;
}

.gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 1rem !important;
}

.tabs button.selected,
.tab-nav button.selected,
[role="tablist"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-color: var(--accent) !important;
}

#header-band {
    background: var(--header-bg);
    padding: 1.75rem 1rem 1.5rem 1rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    width: 100%;
}

#header-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 600;
    color: #FFFFFF;
    margin-bottom: 0.15rem;
}

#header-byline {
    text-align: center;
    font-size: 0.9rem;
    color: #A9E5D9;
}

.side-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04);
}

.side-card h4 {
    margin-top: 0;
    margin-bottom: 0.5rem;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text);
}

.side-card p, .side-card li {
    font-size: 0.85rem;
    color: var(--text-muted);
    line-height: 1.55;
}

.route-badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    color: white;
}
.route-relevant { background: var(--status-green); }
.route-mixed { background: var(--status-amber); }
.route-irrelevant { background: var(--status-amber); }
.route-fallback { background: var(--status-red); }

.metric-strip {
    display: flex;
    gap: 1.5rem;
    padding: 0.6rem 0;
    font-size: 0.85rem;
    color: var(--text-muted);
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    margin: 0.75rem 0;
}
.metric-strip b { color: var(--text); }

.citation-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.5rem;
    font-size: 0.83rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}
.citation-item .src { color: var(--text-muted); font-size: 0.75rem; }

button.primary {
    background: var(--accent) !important;
    border: none !important;
}
button.primary:hover {
    background: var(--accent-hover) !important;
}

.gradio-container a {
    color: var(--support) !important;
}

.results-group, .citations-accordion, .citations-body {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    padding: 1.25rem 1.4rem !important;
}

.blend-into-parent {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.trace-step {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.55rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}
.trace-step .step-name { font-weight: 600; font-size: 0.88rem; }
.trace-step .step-detail { color: var(--text-muted); font-size: 0.82rem; }
"""

EXAMPLE_QUESTIONS = [
    "What privileges are required to create a table in a schema?",
    "How can I access a SQL warehouse from a notebook?",
    "How does privilege inheritance work for container objects in Unity Catalog?",
]


# ---------------------------------------------------------------------------
# Graph invocation
# ---------------------------------------------------------------------------
def _make_initial_state(question: str) -> dict:
    return {
        "question": question,
        "rewritten_queries": [],
        "injection_flagged": False,
        "retrieve_targets": [],
        "chunks_text": [],
        "chunks_semantic": [],
        "chunks_graph": [],
        "reranked_chunks": [],
        "relevance_grades": [],
        "route": "",
        "web_results": [],
        "web_grades": [],
        "refined_context": "",
        "refined_chunks": [],
        "answer": "",
        "faithfulness_score": 0.0,
        "citation_valid": False,
        "guardrail_passed": False,
        "guardrail_reason": "",
        "guardrail_retry_count": 0,
        "context_relevance_score": 0.0,
        "answer_relevancy_score": 0.0,
        "context_sources": [],
        "citations": [],
        "nodes_executed": [],
    }


def _route_badge(result: dict) -> str:
    if result["injection_flagged"]:
        return '<span class="route-badge route-fallback">Blocked</span>'
    if "fallback_response" in result["nodes_executed"]:
        return '<span class="route-badge route-fallback">Fallback (ungrounded, could not verify)</span>'
    route = result.get("route", "")
    label = {
        "relevant": "Relevant (internal knowledge)",
        "mixed": "Mixed (web search used)",
        "irrelevant": "Web search only",
    }.get(route, route or "n/a")
    css_class = {"relevant": "route-relevant", "mixed": "route-mixed", "irrelevant": "route-irrelevant"}.get(route, "route-mixed")
    return f'<span class="route-badge {css_class}">{label}</span>'


def _format_citations(result: dict) -> str:
    citations = result.get("citations", [])
    if not citations:
        return "*No specific chunks were cited for this answer.*"

    reranked_by_id = {d["chunk_id"]: d for d in result["reranked_chunks"]}
    web_by_id = {d["chunk_id"]: d for d in result["web_results"]}

    lines = []
    for c in citations:
        cid = c["chunk_id"]
        source = reranked_by_id.get(cid) or web_by_id.get(cid)
        if source is None:
            continue
        is_web = source["doc_id"].startswith("http")
        src_label = source["doc_id"] if is_web else source["doc_id"]
        preview = source["text"][:180].strip()
        lines.append(
            f'<div class="citation-item"><b>{source["section_heading"]}</b>'
            f'<div class="src">{src_label}</div>{preview}...</div>'
        )
    return "\n".join(lines) if lines else "*Cited chunks could not be resolved for display.*"


def _format_metric_strip(result: dict, latency: float, cost: float, input_tok: int, output_tok: int) -> str:
    return (
        f'<div class="metric-strip">'
        f"<span>Latency: <b>{latency:.2f}s</b></span>"
        f"<span>Cost: <b>${cost:.4f}</b></span>"
        f"<span>Tokens: <b>{input_tok:,} in / {output_tok:,} out</b></span>"
        f"<span>Guardrail retries: <b>{result['guardrail_retry_count']}</b></span>"
        f"</div>"
    )


_TRACE_LABELS = {
    "input_guardrail": "Input Guardrail",
    "classify_query": "Classify Query",
    "retrieve_text": "Retrieve (BM25 / keyword)",
    "retrieve_semantic": "Retrieve (semantic / vector)",
    "retrieve_graph": "Retrieve (knowledge graph)",
    "fuse_rerank": "Fuse & Rerank",
    "grade_relevance": "Grade Relevance",
    "web_search": "Web Search (Tavily fallback)",
    "grade_web_results": "Grade Web Results",
    "refine_knowledge": "Refine Knowledge",
    "generate": "Generate Answer",
    "output_guardrail": "Output Guardrail",
    "fallback_response": "Fallback Response",
    "score_response": "Score Response",
}


def _trace_detail(step: str, result: dict) -> str:
    """One-line detail per step, pulled from the FINAL state, not a
    per-attempt history. If a step like "generate" ran twice due to a
    guardrail retry, both occurrences show in the step list, but the
    detail line reflects only the final state's values, CRAGState
    doesn't preserve what an earlier, failed attempt actually produced."""
    if step == "input_guardrail":
        return "Blocked (possible manipulation attempt)" if result["injection_flagged"] else "Passed"
    if step == "classify_query":
        return f"Retrievers selected: {', '.join(result['retrieve_targets'])}"
    if step == "retrieve_text":
        return f"{len(result['chunks_text'])} chunks retrieved"
    if step == "retrieve_semantic":
        return f"{len(result['chunks_semantic'])} chunks retrieved"
    if step == "retrieve_graph":
        return f"{len(result['chunks_graph'])} graph facts retrieved"
    if step == "fuse_rerank":
        return f"{len(result['reranked_chunks'])} chunks after fusion + reranking"
    if step == "grade_relevance":
        n_relevant = sum(1 for g in result["relevance_grades"] if g["relevant"])
        return f"{n_relevant}/{len(result['relevance_grades'])} chunks graded relevant \u2192 route: {result['route']}"
    if step == "web_search":
        return f"{len(result['web_results'])} results from Tavily"
    if step == "grade_web_results":
        n_relevant = sum(1 for g in result["web_grades"] if g["relevant"])
        return f"{n_relevant}/{len(result['web_grades'])} web results graded relevant"
    if step == "refine_knowledge":
        return f"{len(result['refined_chunks'])} chunks survived refinement"
    if step == "generate":
        return f"Answer generated, {len(result['citations'])} citation(s)"
    if step == "output_guardrail":
        status = "Passed" if result["guardrail_passed"] else "Failed"
        detail = f"{status} \u2014 citation_valid: {result['citation_valid']}"
        if not result["guardrail_passed"] and result.get("guardrail_reason"):
            detail += f"<br><span style=\"font-style: italic;\">{result['guardrail_reason']}</span>"
        return detail
    if step == "fallback_response":
        return "Guardrail retries exhausted, safe fallback message returned"
    if step == "score_response":
        return (
            f"Faithfulness: {result['faithfulness_score']:.2f}, "
            f"Context relevance: {result['context_relevance_score']:.2f}, "
            f"Answer relevancy: {result['answer_relevancy_score']:.2f}"
        )
    return ""


def _format_trace(result: dict) -> str:
    steps = result.get("nodes_executed", [])
    if not steps:
        return "*No trace available yet, ask a question first.*"

    html = []
    for i, step in enumerate(steps, 1):
        label = _TRACE_LABELS.get(step, step)
        detail = _trace_detail(step, result)
        html.append(
            f'<div class="trace-step"><div class="step-name">{i}. {label}</div>'
            f'<div class="step-detail">{detail}</div></div>'
        )
    return "\n".join(html)


def ask_question(question: str):
    if not question or not question.strip():
        return (
            "*Please enter a question.*", "", "", "", "", None
        )

    start_tracking()
    t0 = time.perf_counter()
    result = _graph.invoke(_make_initial_state(question))
    latency = time.perf_counter() - t0
    usage = get_usage_summary()

    answer_md = result["answer"]
    citations_md = _format_citations(result)
    badge_html = _route_badge(result)
    metrics_html = _format_metric_strip(
        result, latency, usage.total_cost_usd, usage.total_input_tokens, usage.total_output_tokens
    )
    trace_html = _format_trace(result)

    return answer_md, citations_md, badge_html, metrics_html, trace_html, result


# ---------------------------------------------------------------------------
# Evaluation tab
#
# Redesigned to score whatever question was most recently asked in the
# "Ask a Question" tab, rather than a fixed 3-question benchmark set.
# This is a genuinely different feature, not a variant of the old one:
# the old design let scores be compared meaningfully across separate
# runs over time (same fixed questions each time), this one is a
# quick, ad-hoc quality check on whatever you just asked, with no
# cross-run comparability, each click scores a potentially different
# question.
#
# Only Faithfulness and AnswerRelevancy are computed here, not all
# four ragas metrics. ContextPrecision and ContextRecall both need a
# pre-written reference (ground-truth) answer to judge against, which
# a live, ad-hoc question genuinely doesn't have, there's no honest
# way to fake one. The old fixed-question design could compute all
# four because its questions came with hand-written references.
# ---------------------------------------------------------------------------
ZERO_EVAL_HTML = (
    '<div class="metric-strip">'
    "<span>Faithfulness: <b>\u2014</b></span>"
    "<span>Answer Relevancy: <b>\u2014</b></span>"
    "</div>"
    '<p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem;">'
    "Ask a question in the \u201cAsk a Question\u201d tab first, then come back here "
    "and click \u201cScore This Answer.\u201d Context Precision and Context Recall "
    "aren't shown here, both need a pre-written reference answer to judge against, "
    "which an ad-hoc question doesn't have.</p>"
)


def _eval_summary_html(df: pd.DataFrame) -> str:
    parts = []
    for m in ["faithfulness", "answer_relevancy"]:
        if m in df.columns:
            parts.append(f"<span>{m.replace('_', ' ').title()}: <b>{df[m].iloc[0]:.2f}</b></span>")
    return '<div class="metric-strip">' + "".join(parts) + "</div>"


def score_last_answer(last_result: dict | None, progress=gr.Progress()):
    """Scores the most recent result from the Ask a Question tab,
    stored in last_result_state, no new graph.invoke() needed, that
    already happened when the question was first asked, this only
    spends the (much cheaper) ragas judge calls."""
    from crag.eval.run_eval import collect_retrieved_context
    from crag.eval.ragas_metrics import build_sample, run_ragas_evaluation

    if not last_result:
        return (
            '<div class="metric-strip">No question asked yet.</div>'
            '<p style="color: var(--text-muted); font-size: 0.85rem;">'
            "Go to the \u201cAsk a Question\u201d tab, ask something, then come back here.</p>",
            pd.DataFrame(),
        )

    progress(0.3, desc="Scoring with ragas...")
    retrieved_contexts = collect_retrieved_context(last_result)
    sample = build_sample(
        question=last_result["question"],
        answer=last_result["answer"],
        retrieved_contexts=retrieved_contexts,
    )

    try:
        ragas_result = run_ragas_evaluation([sample], include_reference_metrics=False)
    except Exception as e:
        return (
            f'<div class="metric-strip">Scoring failed: {e}</div>',
            pd.DataFrame(),
        )

    progress(1.0, desc="Done")
    df = ragas_result.to_pandas()
    summary = _eval_summary_html(df)
    table = df[["user_input", "response", "faithfulness", "answer_relevancy"]]
    return summary, table


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks(title=PROJECT_NAME) as demo:
    gr.HTML(
        f'<div id="header-band">'
        f'<div id="header-title">{PROJECT_NAME}</div>'
        f'<div id="header-byline">Created by {AUTHOR_NAME}</div>'
        f'</div>'
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(
                '<div class="side-card">'
                '<h4>About this project</h4>'
                '<p>A corrective, agentic retrieval-augmented generation system for answering '
                'questions about Databricks Unity Catalog. It combines keyword search, semantic '
                'search, and a knowledge graph, grades its own retrieval quality, and falls back '
                'to a live web search when internal documentation doesn\u2019t cover a question.</p>'
                '</div>'
            )
            gr.HTML(
                '<div class="side-card">'
                '<h4>About the data</h4>'
                '<p>Answers are grounded in five Databricks documentation pages: database objects, '
                'Unity Catalog privileges, the permissions model, connecting to SQL warehouses, '
                'and SQL release notes.</p>'
                '<p><b>Good questions to ask:</b></p>'
                '<ul>'
                '<li>Specific privilege or permission requirements</li>'
                '<li>Relationships between catalogs, schemas, and tables</li>'
                '<li>SQL warehouse connection and access</li>'
                '</ul>'
                '<p>Questions outside this scope will trigger a live web search instead.</p>'
                '</div>'
            )

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Ask a Question"):
                    with gr.Group(elem_classes=["results-group"]):
                        question_box = gr.Textbox(
                            label="Your question", placeholder="Ask about Databricks Unity Catalog...", lines=2
                        )
                        gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=[question_box], label="Try one of these")
                        ask_btn = gr.Button("Ask", variant="primary")

                    with gr.Group(elem_classes=["results-group"]):
                        badge_out = gr.HTML(elem_classes=["blend-into-parent"])
                        answer_out = gr.Markdown(label="Answer", elem_classes=["blend-into-parent"])
                        metrics_out = gr.HTML(elem_classes=["citations-body"])
                        with gr.Accordion("Citations", open=True, elem_classes=["citations-accordion"]):
                            citations_out = gr.HTML(elem_classes=["citations-body"])

                with gr.Tab("Evaluation"):
                    gr.Markdown(
                        "Scores the most recently asked question (from the \u201cAsk a Question\u201d "
                        "tab) using ragas, reusing the answer already generated, no new pipeline "
                        "run needed. Only Faithfulness and Answer Relevancy are shown, the other "
                        "two ragas metrics need a pre-written reference answer this ad-hoc mode "
                        "doesn't have."
                    )
                    eval_summary_out = gr.HTML(value=ZERO_EVAL_HTML, elem_classes=["citations-body"])
                    eval_table_out = gr.Dataframe(interactive=False, wrap=True)
                    eval_btn = gr.Button("Score This Answer")

                with gr.Tab("Pipeline Trace"):
                    gr.Markdown(
                        "Step-by-step trace of the most recent question's run through the graph. "
                        "If a guardrail retry occurred, only the final attempt's details are shown."
                    )
                    trace_out = gr.HTML(elem_classes=["citations-body"])

    last_result_state = gr.State(value=None)

    ask_btn.click(
        ask_question,
        inputs=[question_box],
        outputs=[answer_out, citations_out, badge_out, metrics_out, trace_out, last_result_state],
    )
    question_box.submit(
        ask_question,
        inputs=[question_box],
        outputs=[answer_out, citations_out, badge_out, metrics_out, trace_out, last_result_state],
    )

    eval_btn.click(score_last_answer, inputs=[last_result_state], outputs=[eval_summary_out, eval_table_out])


if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)