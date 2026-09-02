"""Small structured trace models that exclude prompts, secrets, and hidden reasoning."""

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import Field

from tft_agent.core.schemas import (
    AgentStatus,
    NumericClaimStatus,
    QualitativeClaimStatus,
    StrictModel,
    utc_now,
)


class TokenUsage(StrictModel):
    """Provider-reported token counts when available."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class LLMCallTrace(StrictModel):
    """Non-sensitive summary of one provider call."""

    call_index: int = Field(ge=1)
    model: str
    latency_ms: float = Field(ge=0)
    outcome: str
    token_usage: TokenUsage | None = None


class NumericClaimTrace(StrictModel):
    """Non-sensitive provenance validation result for one declared claim."""

    value: int | float | str
    meaning: str
    evidence_ref: str | None = None
    status: NumericClaimStatus


class NumericClaimValidationTrace(StrictModel):
    """Aggregate claim enforcement cost and outcome for one final generation."""

    total_claims: int = Field(default=0, ge=0)
    supported_claims: int = Field(default=0, ge=0)
    unsupported_claims: int = Field(default=0, ge=0)
    mismatched_claims: int = Field(default=0, ge=0)
    invalid_reference_claims: int = Field(default=0, ge=0)
    validation_latency_ms: float = Field(default=0, ge=0)
    correction_attempted: bool = False
    correction_succeeded: bool = False
    correction_latency_ms: float | None = Field(default=None, ge=0)
    fallback_used: bool = False
    initial_answer: str | None = None
    claims: list[NumericClaimTrace] = Field(default_factory=list)


class QualitativeRecommendationTrace(StrictModel):
    status: QualitativeClaimStatus
    evidence_refs: list[str] = Field(default_factory=list)
    preserved_after_repair: bool = False


class ToolArgumentValidationTrace(StrictModel):
    """Aggregate pre-execution argument provenance enforcement."""

    proposed_tool_calls: int = Field(default=0, ge=0)
    executed_tool_calls: int = Field(default=0, ge=0)
    argument_claim_count: int = Field(default=0, ge=0)
    supported_argument_claims: int = Field(default=0, ge=0)
    unsupported_argument_claims: int = Field(default=0, ge=0)
    mismatched_argument_claims: int = Field(default=0, ge=0)
    invalid_reference_argument_claims: int = Field(default=0, ge=0)
    blocked_tool_calls: int = Field(default=0, ge=0)
    request_context_due_to_argument_grounding: int = Field(default=0, ge=0)


class AgentTrace(StrictModel):
    """Lifecycle and LLM summaries for one bounded agent task."""

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    status: AgentStatus = AgentStatus.RUNNING
    llm_calls: list[LLMCallTrace] = Field(default_factory=list)
    available_tool_names: list[str] = Field(default_factory=list)
    final_used_tools: list[str] = Field(default_factory=list)
    final_sources: list[str] = Field(default_factory=list)
    game_state_snapshot: dict[str, Any] = Field(default_factory=dict)
    grounded_evidence_fields: list[str] = Field(default_factory=list)
    numeric_claim_validation: NumericClaimValidationTrace | None = None
    qualitative_recommendation_validation: QualitativeRecommendationTrace | None = None
    tool_argument_validation: ToolArgumentValidationTrace = Field(
        default_factory=ToolArgumentValidationTrace
    )

    @property
    def total_latency_ms(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds() * 1000
