"""
src/crag/retrieval/graph_retrieve.py

The read path Neo4j never had, graph_extract.py only writes edges in,
nothing until now queries them back out for a given question. Returns
graph facts as plain dicts (source, relation, target, evidence), not as
Chunk objects, an edge has no page numbers or single text body the way
a document chunk does, forcing it into Chunk's shape would mean faking
fields that don't mean anything for a graph fact.

Two steps: extract candidate entity names from the question with a
cheap Haiku call, then traverse outward from any matching Entity nodes
in Neo4j. This is a fuzzy substring match, not exact, "SQL warehouse"
in a question should match a "sql warehouse" node without needing the
question to name an entity with perfect precision.
"""
from neo4j import Driver, GraphDatabase
from crag.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from crag.observability.anthropic_client import get_traced_client

ENTITY_EXTRACT_MODEL = "claude-haiku-4-5-20251001"
GRAPH_MAX_RESULTS = 10

_client = get_traced_client()
_driver: Driver | None = None


def get_graph_driver() -> Driver:
    """Lazy singleton, one driver for the process's life rather than a
    new connection per query, same reasoning as the embedding and
    reranker models being loaded once at module level elsewhere in this
    project."""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    return _driver


ENTITY_EXTRACT_TOOL = {
    "name": "record_entities",
    "description": "Record the candidate entity names mentioned or implied in the question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short entity names, e.g. 'catalog', 'SELECT privilege', 'SQL warehouse', not full phrases",
            },
        },
        "required": ["entities"],
    },
}

ENTITY_EXTRACT_SYSTEM_PROMPT = """You extract candidate entity names from a question about
Databricks Unity Catalog, for a graph lookup. Entities are things like catalog, schema, table,
a specific privilege name (SELECT, MODIFY, USE CATALOG), a role, or a feature like SQL warehouse.

Extract short entity names, not the whole question restated, and not verbs or actions, "create
a table" should yield "table", not "create a table". If the question mentions no clear entity
from this domain, return an empty list rather than guessing."""


def extract_entity_candidates(question: str) -> list[str]:
    response = _client.messages.create(
        model=ENTITY_EXTRACT_MODEL,
        max_tokens=200,
        system=ENTITY_EXTRACT_SYSTEM_PROMPT,
        tools=[ENTITY_EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "record_entities"},
        messages=[{"role": "user", "content": f"Question: {question}"}],
    )
    tool_use_block = next(b for b in response.content if b.type == "tool_use")
    return tool_use_block.input.get("entities", [])


def query_graph_facts(question: str, driver: Driver, max_results: int = GRAPH_MAX_RESULTS) -> list[dict]:
    """The real entry point: extracts entity candidates from the
    question, then for each one, finds Entity nodes whose name contains
    it (case-insensitive substring, not exact match) and returns their
    immediate outgoing relationships as facts. Deduplicates across
    candidates, since two different extracted entities can easily
    surface the same edge (asking about both "catalog" and "schema"
    would both match the catalog-PART_OF-metastore edge's neighborhood,
    say)."""
    entities = extract_entity_candidates(question)
    if not entities:
        return []

    seen = set()
    facts = []
    with driver.session() as session:
        for entity in entities:
            result = session.run(
                """
                MATCH (s:Entity)-[r]->(t:Entity)
                WHERE toLower(s.name) CONTAINS toLower($entity)
                   OR toLower(t.name) CONTAINS toLower($entity)
                RETURN s.name AS source, type(r) AS relation, t.name AS target, r.evidence AS evidence
                LIMIT $limit
                """,
                entity=entity,
                limit=max_results,
            )
            for record in result:
                key = (record["source"], record["relation"], record["target"])
                if key in seen:
                    continue
                seen.add(key)
                facts.append(
                    {
                        "source": record["source"],
                        "relation": record["relation"],
                        "target": record["target"],
                        "evidence": record["evidence"],
                    }
                )

    return facts[:max_results]