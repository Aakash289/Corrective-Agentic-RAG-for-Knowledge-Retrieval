"""
src/crag/observability/usage_tracker.py

Per-query token and cost tracking, computed locally in real time, not
pulled from LangSmith. LangSmith ingests traces asynchronously, there's
no guarantee a full trace is queryable the instant graph.invoke()
returns, so it's the wrong source for a number the UI needs to show
immediately after a query completes. This module instead accumulates
usage directly from each real Anthropic API response as it happens.

Uses contextvars, not a plain module-level list, because Gradio can
serve concurrent users, a plain global would let one user's query
tokens leak into another's running total. contextvars gives each
request its own isolated accumulator automatically.

Pricing is current as of this file being written (confirmed directly
against platform.claude.com/docs/en/about-claude/pricing): Sonnet 5 at
$2/$10 per million input/output tokens, Haiku 4.5 at $1/$5. Anthropic
has changed pricing before during this project (Sonnet 5's scheduled
increase to $3/$15 was cancelled), worth re-checking this table if a
displayed cost ever looks obviously wrong.
"""
import contextvars
from dataclasses import dataclass, field

MODEL_PRICING = {
    # (input $ per million tokens, output $ per million tokens)
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}

# Tavily web search is priced separately, $10 per 1,000 searches, not
# token-based, and isn't tracked here since it goes through a different
# client entirely (TavilyClient, not the Anthropic client this module
# hooks into). A query where the corrective fallback fires will
# genuinely cost slightly more than the total shown here, that gap is
# real and currently unaccounted for, not hidden on purpose, just not
# yet wired up. Worth adding record_tavily_search() here and a call to
# it from tavily_search.py if exact total cost including search ever
# matters more than it does today.
TAVILY_COST_PER_SEARCH = 0.01


@dataclass
class UsageRecord:
    model: str
    input_tokens: int
    output_tokens: int


@dataclass
class UsageSummary:
    records: list[UsageRecord] = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def total_calls(self) -> int:
        return len(self.records)

    @property
    def total_cost_usd(self) -> float:
        total = 0.0
        for r in self.records:
            in_price, out_price = MODEL_PRICING.get(r.model, (0.0, 0.0))
            total += (r.input_tokens / 1_000_000) * in_price
            total += (r.output_tokens / 1_000_000) * out_price
        return total

    def cost_by_model(self) -> dict[str, float]:
        by_model: dict[str, float] = {}
        for r in self.records:
            in_price, out_price = MODEL_PRICING.get(r.model, (0.0, 0.0))
            cost = (r.input_tokens / 1_000_000) * in_price + (r.output_tokens / 1_000_000) * out_price
            by_model[r.model] = by_model.get(r.model, 0.0) + cost
        return by_model


_current_usage: contextvars.ContextVar[UsageSummary | None] = contextvars.ContextVar(
    "current_usage", default=None
)


def start_tracking() -> None:
    """Call this once, right before graph.invoke(), to reset the
    accumulator for a new query. Each call to this starts a fresh
    UsageSummary in this context, isolated from any other concurrent
    request's tracking."""
    _current_usage.set(UsageSummary())


def record_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    """Called automatically by the traced Anthropic client after every
    real API response, appends one record to whichever UsageSummary is
    active in the current context. Silently does nothing if
    start_tracking() was never called, so this is safe to call from
    contexts that don't care about tracking (a standalone script test,
    say) without raising."""
    summary = _current_usage.get()
    if summary is not None:
        summary.records.append(UsageRecord(model=model, input_tokens=input_tokens, output_tokens=output_tokens))


def get_usage_summary() -> UsageSummary:
    """Call this after graph.invoke() returns, to read the totals
    accumulated during that one query. Returns an empty UsageSummary
    (all zeros) if start_tracking() was never called first."""
    summary = _current_usage.get()
    return summary if summary is not None else UsageSummary()