"""
src/crag/graph/grade_web_results.py

Same grading concept as grade_relevance.py, applied to Tavily's web
results instead of internal retrieval. Reuses grade_chunk_relevance
directly rather than duplicating grading logic, a web result and an
internal chunk are graded the same way, relevant or not to the
question, only the source differs.
"""
from crag.ingestion.chunker import Chunk
from crag.graph.grade_relevance import grade_chunk_relevance, RelevanceGrade


def grade_web_chunks(question: str, chunks: list[Chunk]) -> list[RelevanceGrade]:
    return [grade_chunk_relevance(question, chunk) for chunk in chunks]