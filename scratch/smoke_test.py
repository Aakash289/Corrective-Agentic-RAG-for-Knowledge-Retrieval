from crag.ingestion.chunker import extract_text_and_sections
from crag.retrieval.tavily_search import web_search_fallback
from crag.graph.grade_web_results import grade_web_chunks
from crag.graph.refine_knowledge import refine_chunks
from crag.graph.generate import generate_answer
from crag.guardrails.output_guardrail import check_output

question = "How do I configure autoscaling for a Databricks cluster?"

print("Running web search + grading + refinement...")
web_results = web_search_fallback(question)
web_grades = grade_web_chunks(question, web_results)
relevant_chunks = [c for c, g in zip(web_results, web_grades) if g.relevant]
print(f"{len(relevant_chunks)} of {len(web_results)} web results graded relevant\n")

for c in relevant_chunks:
    print(f"  {c.doc_id}")

refined = refine_chunks(question, relevant_chunks)
refined_chunks = [
    type(relevant_chunks[0])(chunk_id=rc.chunk_id, doc_id=c.doc_id, section_heading=c.section_heading,
                              text=rc.refined_text, page_start=c.page_start, page_end=c.page_end)
    for rc, c in zip(refined, relevant_chunks) if rc.refined_text.strip()
]
print(f"\n{len(refined_chunks)} chunks survived refinement\n")

print("=" * 60)
print("Generating answer...")
print("=" * 60)
result = generate_answer(question, refined_chunks, graph_facts=None)
print(f"\nanswer:\n{result.answer}")
print(f"\ncited_chunk_ids: {result.cited_chunk_ids}")

print("\n" + "=" * 60)
print("Checking output guardrail...")
print("=" * 60)
guardrail_result = check_output(result.answer, refined_chunks)
print(f"passed: {guardrail_result.passed}")
print(f"reason: {guardrail_result.reason}")

print("\n\nAnswer text repr (checking for stray artifacts):")
print(repr(result.answer[-50:]))