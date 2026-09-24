"""
src/crag/ingestion/graph_schema.py

The controlled vocabulary for graph extraction. NODE_TYPES and EDGE_TYPES
are Literal types, not free strings, so the extraction LLM call is
constrained to a fixed set of categories rather than inventing a new node
type per chunk, which is what would turn the graph into an unqueryable
pile of one-off labels instead of a structure with real, repeatable
shape. This is the file to edit if extraction quality calibration turns
up types that are too coarse or too narrow for this corpus, everything
downstream (graph_extract.py, the Neo4j MERGE keys, later the graph
retriever) reads its shape from here.
"""
from typing import Literal
from pydantic import BaseModel, Field

# Securable and conceptual objects that actually appear across this
# corpus: Unity Catalog's object hierarchy, permissions and privileges,
# and SQL warehouse concepts. Kept deliberately small and specific to
# what the five source documents actually talk about, rather than a
# generic ontology, a controlled vocabulary only helps if it's tight
# enough that two different chunks extracting the same real-world thing
# actually land on the same type.
NODE_TYPES = Literal[
    "SecurableObject",  # catalog, schema, table, view, volume, function, model
    "Privilege",         # e.g. SELECT, MODIFY, USE CATALOG
    "Role",               # e.g. account admin, metastore admin, workspace user
    "PermissionLevel",   # e.g. Can Manage, Can Monitor, Can Use
    "Feature",            # e.g. SQL warehouse, Genie, Catalog Explorer
    "Concept",            # e.g. permissions model, ownership, ACL
]

EDGE_TYPES = Literal[
    "REQUIRES",     # X requires Y to happen/exist (a privilege requires a role)
    "GRANTS",       # X grants Y (a role grants a privilege)
    "PART_OF",      # X is structurally part of Y (a table is part of a schema)
    "APPLIES_TO",   # X applies to Y (a privilege applies to a securable object)
    "ENABLES",      # X enables Y (a permission level enables an action)
]


class ExtractedEdge(BaseModel):
    """One extracted relationship. evidence is required, not optional,
    every edge in the graph needs to be traceable back to the exact
    sentence that justified it, this is what keeps graph retrieval
    citeable the same way chunk retrieval already is via chunk_id."""
    source: str = Field(description="The source entity's name, as it appears in the text")
    source_type: NODE_TYPES
    relation: EDGE_TYPES
    target: str = Field(description="The target entity's name, as it appears in the text")
    target_type: NODE_TYPES
    evidence: str = Field(description="The exact sentence or phrase from the chunk that supports this edge")