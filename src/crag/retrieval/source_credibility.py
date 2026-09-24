"""
src/crag/retrieval/source_credibility.py

Classifies a web source's domain into a credibility tier, deterministic,
no LLM call, this is a factual property of a URL (is it Databricks's own
domain or not), not a judgment call. Motivated directly by a real case
from this project's Tavily testing: the autoscaling question's citations
included oneuptime.com, a third-party blog, alongside four official
Databricks docs pages, all treated identically once each passed
relevance grading. This doesn't filter that source out, it was
topically accurate, silently dropping it would throw away real
information, instead it labels sources so generation can weight and
hedge appropriately, preferring official docs and flagging explicitly
when a claim rests only on an unverified one.
"""
from urllib.parse import urlparse

OFFICIAL_DOMAINS = {
    "docs.databricks.com",
    "databricks.com",
    "www.databricks.com",
    "learn.microsoft.com",  # Azure Databricks docs are hosted here, not
                              # on a databricks.com domain, still genuinely
                              # official Databricks/Microsoft documentation
}

COMMUNITY_DOMAINS = {
    "community.databricks.com",  # Databricks-run, but user-generated
                                    # content, not the same authority as
                                    # the official docs themselves
}


def classify_source(url: str) -> str:
    """Returns "official", "community", or "unverified". Matches on the
    exact domain plus subdomains of it (docs.databricks.com matches
    under the databricks.com entry too), not a substring match, so a
    domain like "notdatabricks.com" or "databricks.com.evil.example"
    doesn't falsely qualify."""
    try:
        domain = urlparse(url).netloc.lower()
    except Exception:
        return "unverified"

    for official in OFFICIAL_DOMAINS:
        if domain == official or domain.endswith(f".{official}"):
            return "official"
    for community in COMMUNITY_DOMAINS:
        if domain == community or domain.endswith(f".{community}"):
            return "community"
    return "unverified"


def credibility_label(url: str) -> str:
    """Human-readable label for the tier, used directly in prompts so
    the model sees the credibility signal alongside the content itself,
    not as separate metadata it has to cross-reference."""
    tier = classify_source(url)
    return {
        "official": "Official Databricks/Microsoft documentation",
        "community": "Databricks Community (user-generated, not official docs)",
        "unverified": "Third-party source, not Databricks-affiliated",
    }[tier]