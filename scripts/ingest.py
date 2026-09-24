"""
scripts/ingest.py

The single real entry point for building this project's full corpus
index, consolidating what has, until now, only ever happened piecemeal
across individual test scripts in scratch/smoke_test.py. Running this
once (or after any change to data/raw_pdfs/) is what should keep BM25,
Chroma, and optionally Neo4j in sync with the actual five-document
corpus, rather than relying on whichever documents happened to get
touched during whatever was being tested that day, which is exactly
how Document_2, Document_3, and Document_5 silently ended up missing
from Chroma for a large stretch of this project's development.

Three steps, run independently:
  1. BM25 index (build_bm25_index + save_bm25_index)
  2. Chroma embeddings (embed_chunks, per document)
  3. Neo4j graph extraction (extract_edges_from_chunk + write_edges_to_neo4j)

Step 3 is OFF by default. It is genuinely expensive, one Claude call per
chunk, roughly 200 calls across the full corpus, real cost and real
time, not something that should run silently as a side effect of
someone re-running this script to fix a BM25 or Chroma gap. Pass
--include-graph explicitly to run it.

Usage:
    uv run python scripts/ingest.py                  # BM25 + Chroma only
    uv run python scripts/ingest.py --include-graph   # + full graph extraction
    uv run python scripts/ingest.py --skip-bm25 --skip-embed --include-graph
        # graph extraction only, e.g. after already fixing BM25/Chroma separately
"""
import argparse
from crag.ingestion.chunker import extract_text_and_sections, Chunk
from crag.ingestion.pdf_extract import file_content_hash
from crag.ingestion.bm25_index import build_bm25_index, save_bm25_index
from crag.ingestion.embed_index import embed_chunks
from crag.ingestion.graph_extract import extract_edges_from_chunk, write_edges_to_neo4j

CORPUS = {
    "Document_1": "Database objects in Databricks",
    "Document_2": "Unity Catalog privileges reference",
    "Document_3": "Unity Catalog permissions model concepts",
    "Document_4": "Connect to a SQL warehouse",
    "Document_5": "Databricks SQL release notes 2024",
}
PDF_DIR = "data/raw_pdfs"


def load_all_chunks() -> dict[str, tuple[list[Chunk], bool]]:
    """OCRs and chunks every document exactly once, returning each
    document's chunks and whether OCR was used, so no downstream step
    (BM25, embedding, graph extraction) has to repeat the OCR pass
    that already happened here."""
    result = {}
    for doc_id in CORPUS:
        pdf_path = f"{PDF_DIR}/{doc_id}.pdf"
        print(f"\nProcessing {doc_id} ({CORPUS[doc_id]})...")
        _, used_ocr, chunks = extract_text_and_sections(pdf_path, doc_id)
        print(f"  {len(chunks)} chunks")
        result[doc_id] = (chunks, used_ocr)
    return result


def run_bm25(all_chunks: dict[str, tuple[list[Chunk], bool]]) -> None:
    print("\n" + "=" * 60)
    print("BM25 index")
    print("=" * 60)
    flat_chunks = [c for chunks, _ in all_chunks.values() for c in chunks]
    bm25, chunks = build_bm25_index(flat_chunks)
    save_bm25_index(bm25, chunks)
    print(f"Saved BM25 index: {len(chunks)} chunks total")


def run_embed(all_chunks: dict[str, tuple[list[Chunk], bool]]) -> None:
    print("\n" + "=" * 60)
    print("Chroma embeddings")
    print("=" * 60)
    for doc_id, (chunks, used_ocr) in all_chunks.items():
        pdf_path = f"{PDF_DIR}/{doc_id}.pdf"
        print(f"\nEmbedding {doc_id} ({CORPUS[doc_id]})...")
        embed_chunks(
            chunks,
            doc_title=CORPUS[doc_id],
            extraction_method="ocr" if used_ocr else "text_layer",
            source_doc_hash=file_content_hash(pdf_path),
        )
        print(f"  {len(chunks)} chunks embedded")


def run_graph(all_chunks: dict[str, tuple[list[Chunk], bool]]) -> None:
    print("\n" + "=" * 60)
    print("Neo4j graph extraction (this makes ~200 Claude API calls, one per chunk)")
    print("=" * 60)
    from crag.retrieval.graph_retrieve import get_graph_driver
    driver = get_graph_driver()
    for doc_id, (chunks, _) in all_chunks.items():
        print(f"\nExtracting edges from {doc_id} ({len(chunks)} chunks)...")
        all_edges = []
        for chunk in chunks:
            edges = extract_edges_from_chunk(chunk.text, doc_id)
            all_edges.extend(edges)
        write_edges_to_neo4j(all_edges, doc_id, driver)
        print(f"  {len(all_edges)} edges extracted and written")


def main():
    parser = argparse.ArgumentParser(description="Build the full corpus index for BM25, Chroma, and optionally Neo4j.")
    parser.add_argument("--skip-bm25", action="store_true", help="Skip building the BM25 index")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding into Chroma")
    parser.add_argument("--include-graph", action="store_true", help="Run graph extraction into Neo4j (expensive: ~200 API calls)")
    args = parser.parse_args()

    need_chunks = not args.skip_bm25 or not args.skip_embed or args.include_graph
    if not need_chunks:
        print("Nothing to do: all steps skipped.")
        return

    all_chunks = load_all_chunks()

    if not args.skip_bm25:
        run_bm25(all_chunks)
    else:
        print("\nSkipping BM25 (--skip-bm25)")

    if not args.skip_embed:
        run_embed(all_chunks)
    else:
        print("\nSkipping Chroma embeddings (--skip-embed)")

    if args.include_graph:
        run_graph(all_chunks)
    else:
        print("\nSkipping graph extraction (pass --include-graph to run it, ~200 API calls)")

    print("\n" + "=" * 60)
    print("Ingestion complete")
    print("=" * 60)


if __name__ == "__main__":
    main()