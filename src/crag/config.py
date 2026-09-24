"""
src/crag/config.py

Every tunable value in one place, per section 11: the RRF constant, the
reranker cutoff, chunk sizing, guardrail thresholds, Tavily's call budget.
Change a threshold here, not by hunting through node files.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API keys and connection info, read from .env ---
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.environ.get("LANGCHAIN_TRACING_V2", "true")

# --- Models ---
# Haiku for high-volume, narrow tasks: grading, guardrail checks, extraction.
# Sonnet for generation and as the judge model, a stronger model than the
# one doing the grading it is judging, so it is not grading its own homework.
MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_SONNET = "claude-sonnet-5"

# --- Chunking, section 7 ---
CHUNK_SIZE_WORDS = 700
CHUNK_OVERLAP_WORDS = 100
HEADING_FONT_SIZE = 13.0       # PyMuPDF span size threshold for a section heading
MIN_CHARS_PER_PAGE = 20        # below this, a PDF page has no usable text layer, OCR it

# --- Retrieval fusion and reranking, section 3 ---
RRF_K = 60                     # standard default for Reciprocal Rank Fusion
RERANK_TOP_N = 8               # how many reranked candidates move on to grading

# --- Web search fallback, section 4 ---
TAVILY_MAX_RESULTS = 3
TAVILY_SEARCH_DEPTH = "basic"  # not "advanced", the snippet is enough for this corpus
TAVILY_MAX_RETRIES = 2         # worst case: 2 calls, 6 graded results per fallback

# --- Guardrails, section 10 ---
FAITHFULNESS_THRESHOLD = 0.7   # output_guardrail's groundedness gate

# --- Neo4j controlled vocabulary, section 7 ---
NODE_TYPES = ("ProductArea", "Feature", "Edition", "Object", "ReleaseNote")
EDGE_TYPES = ("GOVERNS", "QUERIES", "REQUIRES_EDITION", "DEPRECATES", "DOCUMENTED_IN")
