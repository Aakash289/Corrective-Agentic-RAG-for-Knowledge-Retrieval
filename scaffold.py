"""
scaffold.py

Creates the folder and file skeleton from the discussion doc's section 11,
so VS Code's file explorer shows the full project structure in one shot
instead of creating two dozen files and folders by hand. Safe to re-run,
it never overwrites a file that already has content.

Usage: uv run python scaffold.py
"""
from pathlib import Path

ROOT = Path(__file__).parent

# path -> placeholder content. A trailing slash with no content means
# "just create this directory", used for data/ subfolders that start empty.
FILES = {
    "data/raw_pdfs/.gitkeep": "",
    "data/processed/.gitkeep": "",
    "notebooks/01_ingestion_pipeline.ipynb": "",  # created properly by VS Code
    "notebooks/02_rag_pipeline.ipynb": "",         # on first open, leave empty here
    "src/crag/__init__.py": "",
    "src/crag/config.py": None,   # written separately below, not overwritten
    "src/crag/schema.py": None,   # written separately below, not overwritten
    "src/crag/tracing.py": "# @traceable wrapped nodes, section 5\n",
    "src/crag/ingestion/__init__.py": "",
    "src/crag/ingestion/pdf_extract.py": "# see discussion doc section 7\n",
    "src/crag/ingestion/chunker.py": "# see discussion doc section 7\n",
    "src/crag/ingestion/embed_index.py": "# builds the ChromaDB index, section 6\n",
    "src/crag/ingestion/bm25_index.py": "# builds the rank_bm25 index, section 6\n",
    "src/crag/ingestion/graph_extract.py": "# see discussion doc section 7\n",
    "src/crag/retrieval/__init__.py": "",
    "src/crag/retrieval/text_retriever.py": "# retrieve_text node\n",
    "src/crag/retrieval/semantic_retriever.py": "# retrieve_semantic node\n",
    "src/crag/retrieval/graph_retriever.py": "# retrieve_graph node, Cypher queries, section 7\n",
    "src/crag/retrieval/fuse_rerank.py": "# RRF + BAAI/bge-reranker-base, section 3\n",
    "src/crag/graph/__init__.py": "",
    "src/crag/graph/nodes.py": "# every LangGraph node function\n",
    "src/crag/graph/edges.py": "# conditional edge and routing functions\n",
    "src/crag/graph/build_graph.py": "# StateGraph construction, section 5\n",
    "src/crag/guardrails/__init__.py": "",
    "src/crag/guardrails/input_guardrail.py": "# see discussion doc section 10\n",
    "src/crag/guardrails/output_guardrail.py": "# see discussion doc section 10\n",
    "src/crag/eval/__init__.py": "",
    "src/crag/eval/ragas_metrics.py": "# RAGAS wired to Haiku via LangchainLLMWrapper, section 8\n",
    "src/crag/eval/run_eval.py": "# batch eval runner over eval_questions.json\n",
    "app/gradio_app.py": "# four outputs per query: answer, source, citations, metrics\n",
    "scripts/ingest.py": "# CLI entry point, no debug cells, mirrors notebook 01\n",
    "README.md": "# Corrective Agentic RAG for Knowledge Retrieval\n\nSee the discussion and overview docs for full design context.\n",
}


def main():
    created = []
    skipped = []
    for rel_path, content in FILES.items():
        path = ROOT / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            # config.py and schema.py are handled by their own files,
            # never touch them here even if this script re-runs
            continue
        if path.exists() and path.stat().st_size > 0:
            skipped.append(rel_path)
            continue
        path.write_text(content, encoding="utf-8")
        created.append(rel_path)

    print(f"Created {len(created)} files.")
    for f in created:
        print(f"  + {f}")
    if skipped:
        print(f"\nSkipped {len(skipped)} existing non-empty files (left untouched).")


if __name__ == "__main__":
    main()
