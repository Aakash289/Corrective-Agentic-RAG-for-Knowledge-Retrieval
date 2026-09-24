"""
src/crag/observability/anthropic_client.py

Single shared factory for the Anthropic client every module in this
project uses. Two things get layered onto the plain client here:
LangSmith tracing and local usage tracking.

Traces a single function (traced_create), not the whole client object,
via the @traceable decorator. This project's installed langsmith
version's wrap_anthropic (client-wrapping approach) crashed outright,
AttributeError: 'Anthropic' object has no attribute 'completions', it
internally expects a legacy .completions namespace the currently
installed anthropic SDK no longer exposes (the same SDK generation
that already dropped temperature as a direct parameter elsewhere in
this project). Decorating one function instead of wrapping the whole
client sidesteps that internal attribute check entirely, @traceable
just wraps an ordinary Python function, it never inspects the client's
attribute surface the way wrap_anthropic does.

Usage across the project changes slightly from the client-wrapping
design: instead of _client.messages.create(...), call
traced_create(...) with the same keyword arguments. get_traced_client()
is kept as a thin compatibility object exposing .messages.create so
existing call sites (_client.messages.create(...)) don't all need
rewriting, it just routes through traced_create underneath.
"""
import anthropic
from langsmith import traceable
from crag.config import ANTHROPIC_API_KEY
from crag.observability.usage_tracker import record_usage

_raw_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


@traceable(run_type="llm")
def _traced_create(**kwargs):
    response = _raw_client.messages.create(**kwargs)
    record_usage(
        model=kwargs.get("model", "unknown"),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
    return response


class _MessagesNamespace:
    def create(self, **kwargs):
        return _traced_create(**kwargs)


class _TracedClient:
    """Thin stand-in with the same .messages.create(...) shape every
    existing call site already uses, so no changes are needed anywhere
    else in the project, only this file's internals changed."""
    def __init__(self):
        self.messages = _MessagesNamespace()


def get_traced_client() -> _TracedClient:
    return _TracedClient()