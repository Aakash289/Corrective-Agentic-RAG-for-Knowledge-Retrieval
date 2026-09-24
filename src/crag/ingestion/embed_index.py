"""
src/crag/ingestion/embed_index.py

Embeds chunks with sentence-transformers (all-MiniLM-L6-v2) and writes
them into ChromaDB, one ChunkMetadata record per chunk stored alongside
its vector, since ChromaDB is the one index in this stack built to hold
arbitrary metadata next to an embedding.
"""
from datetime import date
import chromadb
from sentence_transformers import SentenceTransformer
from crag.ingestion.chunker import Chunk
from crag.schema import ChunkMetadata

COLLECTION_NAME = "crag_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # same model already decided in the
                                       # discussion doc's tool stack

_model = None  # loaded lazily, the model load is the slow part, don't pay
                # for it if this module is imported but embed_chunks never
                # actually gets called


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_chroma_collection(persist_dir: str = "data/chroma"):
    """Persistent client, not the in-memory default, so the index survives
    between notebook restarts. An in-memory client would mean re-embedding
    the whole corpus every time the notebook reopens."""
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(COLLECTION_NAME)


def embed_chunks(
    chunks: list[Chunk],
    doc_title: str,
    extraction_method: str,
    source_doc_hash: str,
    collection=None,
) -> None:
    """Embeds every chunk from one document and writes it into ChromaDB,
    one ChunkMetadata record per chunk. Batches the embedding call across
    all chunks at once rather than one at a time, sentence-transformers
    is meaningfully faster batched than looped."""
    if not chunks:
        return
    if collection is None:
        collection = get_chroma_collection()

    model = _get_model()
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    ids = [c.chunk_id for c in chunks]
    metadatas = []
    for c in chunks:
        meta = ChunkMetadata(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            doc_title=doc_title,
            section_heading=c.section_heading,
            page_start=c.page_start,
            page_end=c.page_end,
            extraction_method=extraction_method,
            source_doc_hash=source_doc_hash,
            ingested_at=date.today(),
        ).__dict__
        # Chroma's metadata values must be str, int, float, or bool, a date
        # object isn't serializable there, stringify it here rather than
        # inside ChunkMetadata itself, which should stay a plain data holder
        meta["ingested_at"] = meta["ingested_at"].isoformat()
        metadatas.append(meta)

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


def query_similar(query_text: str, n_results: int = 10, collection=None) -> dict:
    """Quick similarity query for testing the index, not the real
    retrieve_semantic node (that belongs in retrieval/semantic_retriever.py),
    just enough here to spot-check that embedding and retrieval round-trip
    correctly before that node exists."""
    if collection is None:
        collection = get_chroma_collection()
    model = _get_model()
    query_embedding = model.encode([query_text]).tolist()
    return collection.query(query_embeddings=query_embedding, n_results=n_results)