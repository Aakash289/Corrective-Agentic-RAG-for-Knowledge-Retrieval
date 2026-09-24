"""
src/crag/ingestion/graph_extract.py

Two responsibilities: calling Claude to extract ExtractedEdge objects from
a chunk's text (extract_edges_from_chunk), and writing those edges into
Neo4j with MERGE, keyed on normalized entity names, so the same entity
mentioned in two different chunks resolves to one node rather than a
duplicate (write_edges_to_neo4j). Determinism for extraction comes from
forcing the record_edges tool via tool_choice, not from a temperature
setting, current Claude models reject temperature as a parameter entirely.
"""
import anthropic
from pydantic import ValidationError
from neo4j import Driver
from crag.ingestion.graph_schema import ExtractedEdge
from crag.observability.anthropic_client import get_traced_client

EXTRACTION_MODEL = "claude-sonnet-5"  # generation-tier model, not the cheap
                                       # Haiku one used for classification and
                                       # guardrail checks elsewhere, extraction
                                       # quality directly determines graph
                                       # retrieval's ceiling, worth the cost

_client = get_traced_client()

EXTRACTION_TOOL = {
    "name": "record_edges",
    "description": "Record every relationship found in the text as a list of edges.",
    "input_schema": {
        "type": "object",
        "properties": {
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "source_type": {
                            "type": "string",
                            "enum": ["SecurableObject", "Privilege", "Role", "PermissionLevel", "Feature", "Concept"],
                        },
                        "relation": {
                            "type": "string",
                            "enum": ["REQUIRES", "GRANTS", "PART_OF", "APPLIES_TO", "ENABLES"],
                        },
                        "target": {"type": "string"},
                        "target_type": {
                            "type": "string",
                            "enum": ["SecurableObject", "Privilege", "Role", "PermissionLevel", "Feature", "Concept"],
                        },
                        "evidence": {"type": "string"},
                    },
                    "required": ["source", "source_type", "relation", "target", "target_type", "evidence"],
                },
            },
        },
        "required": ["edges"],
    },
}

EXTRACTION_SYSTEM_PROMPT = """You extract structured relationships from Databricks documentation text.

Only extract a relationship if the text explicitly supports it, do not infer a
relationship that seems plausible but isn't actually stated. Every edge must
include an evidence field quoting the exact sentence or phrase that justifies
it. If a chunk contains no clear relationships between entities in the allowed
categories, return an empty edges list rather than forcing a weak extraction.

For the PART_OF relation specifically, always extract it as child --PART_OF-->
parent, regardless of how the source sentence is phrased. "A catalog contains
schemas" and "a schema is part of a catalog" describe the same hierarchy and
must produce the same edge direction: Schema --PART_OF--> Catalog. Do not
default to subject-first extraction, identify which entity is structurally
the child before choosing source and target.

Do not extract a category label as if it were a concrete entity. A sentence
like "the following are container objects: catalogs, schemas" is naming the
group that catalog and schema both belong to, not stating that a catalog is
part of, or a member of, something called "container objects." If a term
in the text is functioning as a category description rather than a specific
named thing, do not use it as a source or target, either extract edges
between the concrete items the category introduces, or skip that sentence
if no concrete relationship is stated."""


def extract_edges_from_chunk(chunk_text: str, doc_id: str) -> list[ExtractedEdge]:
    """One Claude call per chunk, temperature=0, forced through the
    record_edges tool so the response is always structured JSON matching
    ExtractedEdge's shape, not free text that then needs separate parsing
    and can silently drift out of schema over time."""
    response = _client.messages.create(
        model=EXTRACTION_MODEL,
        max_tokens=2000,
        # No temperature parameter: current Claude models, claude-sonnet-5
        # included, reject it outright (a 400 invalid_request_error, not
        # just an SDK-level deprecation), determinism for this structured-
        # output task comes from the forced tool_choice below instead, not
        # from a sampling parameter that no longer exists for this model
        system=EXTRACTION_SYSTEM_PROMPT,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "record_edges"},
        messages=[
            {"role": "user", "content": f"Extract relationships from this text (doc_id: {doc_id}):\n\n{chunk_text}"}
        ],
    )

    tool_use_block = next(b for b in response.content if b.type == "tool_use")
    raw_edges = tool_use_block.input.get("edges", [])

    edges = []
    for raw_edge in raw_edges:
        try:
            edges.append(ExtractedEdge(**raw_edge))
        except ValidationError as e:
            # One malformed edge (a missing or invalid field) should not
            # discard this chunk's other, valid edges, and definitely
            # should not crash a long batch run over hundreds of chunks,
            # confirmed the hard way when this exact failure mode took
            # down the rest of a 83-chunk document partway through a
            # full-corpus ingestion run. Skip it, warn loudly so it's
            # visible, keep going.
            print(f"  WARNING: skipped malformed edge in {doc_id}: {raw_edge!r} ({e})")
    return edges


def _normalize_entity_name(name: str) -> str:
    """Lowercases and strips whitespace, the normalization key MERGE uses
    to decide two mentions are the same entity. Deliberately simple,
    "SQL Warehouse" and "sql warehouse " collapse to the same node, but
    "SQL warehouses" (plural) currently does not match "SQL warehouse",
    that's a real limitation worth knowing, not silently hidden, a
    stemming or singularization step would catch more true duplicates
    but risks merging genuinely distinct entities too, worth revisiting
    once real extraction output shows how much this matters in practice."""
    return name.strip().lower()


def write_edges_to_neo4j(edges: list[ExtractedEdge], doc_id: str, driver: Driver) -> None:
    """MERGE on normalized name plus type for both nodes, MERGE on the
    relationship between them, using the edge's actual relation
    (PART_OF, REQUIRES, and so on) as the real Neo4j relationship type,
    not a generic RELATION type with the relation stored as a property.
    This matters for the Browser's graph visualization specifically,
    native relationship types render as readable edge labels directly,
    a generic type would show every edge labeled the same regardless of
    its real meaning.

    Neo4j relationship types can't be parameterized the normal way
    (MERGE (s)-[r:$relation]->(t) isn't valid Cypher), types have to be
    written literally into the query text. Interpolating edge.relation
    directly is safe here specifically because it was already validated
    against the fixed five-value EDGE_TYPES Literal by Pydantic before
    this function ever sees it, it can only ever be one of REQUIRES,
    GRANTS, PART_OF, APPLIES_TO, or ENABLES, never arbitrary text."""
    with driver.session() as session:
        for edge in edges:
            query = f"""
                MERGE (s:Entity {{name: $source_name, type: $source_type}})
                MERGE (t:Entity {{name: $target_name, type: $target_type}})
                MERGE (s)-[r:{edge.relation}]->(t)
                SET r.evidence = $evidence, r.doc_id = $doc_id
            """
            session.run(
                query,
                source_name=_normalize_entity_name(edge.source),
                source_type=edge.source_type,
                target_name=_normalize_entity_name(edge.target),
                target_type=edge.target_type,
                evidence=edge.evidence,
                doc_id=doc_id,
            )