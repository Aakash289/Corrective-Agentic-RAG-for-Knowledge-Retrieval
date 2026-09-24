"""
src/crag/guardrails/guardrail_schema.py

Shared result shape for both guardrail layers. One model, not two,
since input and output guardrails answer the same fundamental question
(does this pass or not, and why), even though they check different
things at different points in the pipeline.

reason has a default, not required: confirmed during real eval testing
that a forced tool call can come back with passed set correctly but
reason omitted entirely, the same missing-required-field failure mode
graph_extract.py hit earlier with target_type. passed is the field
that actually gates behavior, so it stays required, a missing pass/fail
verdict should be loud, not silently defaulted. reason is explanatory
only, used in logs and adapted into user-facing messages on failure,
so a missing one degrading to a generic fallback is safe and doesn't
weaken what the guardrail actually blocks or allows.
"""
from pydantic import BaseModel, Field


class GuardrailResult(BaseModel):
    passed: bool
    reason: str = Field(
        default="(no reason provided by the model)",
        description="Brief explanation of the decision, shown in logs and, on failure, adapted into the user-facing message",
    )