"""
src/crag/eval/run_eval.py

Offline evaluation: runs a fixed set of questions through the actual
compiled LangGraph pipeline (build_graph.py, no shortcuts, this is the
real system, not a stripped-down eval harness), collects what each run
actually retrieved and answered, then scores the whole set with real
ragas metrics (ragas_metrics.py).

3 questions here, one per category, kept deliberately small to control
cost.

Usage:
    uv run python -m crag.eval.run_eval
"""
from crag.graph.build_graph import build_graph
from crag.eval.ragas_metrics import build_sample, run_ragas_evaluation

EVAL_QUESTIONS = [
    {
        "question": "How can I access a SQL warehouse from a notebook?",
        "reference": "You can attach a notebook to a pro or serverless SQL warehouse, and SQL warehouses also appear in the compute drop-down menus of workspace UIs such as the query editor, Catalog Explorer, and dashboards.",
    },
    {
        "question": "What privileges are required to create a table in a schema?",
        "reference": "USE CATALOG on the parent catalog, USE SCHEMA on the parent schema, and CREATE TABLE on the schema (or on the catalog, if CREATE TABLE was granted at the catalog level and inherits down).",
    },
    {
        "question": "How do I configure autoscaling for a Databricks cluster?",
        "reference": "Enable autoscaling in the compute configuration and set minimum and maximum worker counts; Databricks then scales the cluster within that range based on signals like pending task backlog, with the specific algorithm (standard vs optimized vs enhanced) depending on the workspace's pricing tier and compute type.",
    },
]


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
        "guardrail_reason": "",  # same class of missing-field bug as
        # guardrail_retry_count below, added proactively this time
        # rather than waiting to hit the identical crash again
        "guardrail_retry_count": 0,  # BUG FIX: this field was added to
        # CRAGState during the retry-logic work (item 4), but this
        # function was never updated to include it, causing a real
        # crash, KeyError: 'guardrail_retry_count', in output_guardrail_
        # node the moment it ran, confirmed directly via the Evaluation
        # tab's "Run Evaluation" button in the live Gradio app.
        "context_relevance_score": 0.0,
        "answer_relevancy_score": 0.0,
        "context_sources": [],
        "citations": [],
        "nodes_executed": [],
    }


def collect_retrieved_context(result: dict) -> list[str]:
    if result["refined_chunks"]:
        contexts = [rc["refined_text"] for rc in result["refined_chunks"]]
    else:
        contexts = [c["text"] for c in result["reranked_chunks"]]

    for f in result["chunks_graph"]:
        contexts.append(f"{f['source']} --{f['relation']}--> {f['target']} (evidence: {f['evidence']})")

    return contexts


def main():
    graph = build_graph()
    samples = []
    failed_questions = []

    print(f"Running {len(EVAL_QUESTIONS)} questions through the full graph...\n")

    for i, item in enumerate(EVAL_QUESTIONS, 1):
        print(f"[{i}/{len(EVAL_QUESTIONS)}] {item['question']}")
        try:
            result = graph.invoke(_make_initial_state(item["question"]))
        except Exception as e:
            print(f"  FAILED: {e}\n")
            failed_questions.append((item["question"], str(e)))
            continue

        retrieved_contexts = collect_retrieved_context(result)
        samples.append(
            build_sample(
                question=item["question"],
                answer=result["answer"],
                retrieved_contexts=retrieved_contexts,
                reference=item["reference"],
            )
        )
        print(f"  route: {result['route']}, guardrail_passed: {result['guardrail_passed']}\n")

    if failed_questions:
        print(f"\n{len(failed_questions)} of {len(EVAL_QUESTIONS)} questions failed and were excluded from scoring:")
        for q, err in failed_questions:
            print(f"  - {q}: {err}")
        print()

    if not samples:
        print("No questions succeeded, nothing to score.")
        return

    print("Scoring with ragas...\n")
    ragas_result = run_ragas_evaluation(samples)

    print("=" * 60)
    print("Aggregate scores")
    print("=" * 60)
    print(ragas_result)

    df = ragas_result.to_pandas()
    print("\n" + "=" * 60)
    print("Per-question scores")
    print("=" * 60)
    print(df.to_string())

    df.to_csv("data/eval_results.csv", index=False)
    print("\nFull results saved to data/eval_results.csv")


if __name__ == "__main__":
    main()