"""Shared, transport-safe domain schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


class StrictModel(BaseModel):
    """Base model that rejects misspelled or unknown fields."""

    model_config = ConfigDict(extra="forbid")


class UnitState(StrictModel):
    """A unit currently owned or visible to the player."""

    name: str = Field(min_length=1)
    star_level: int = Field(default=1, ge=1, le=4)
    items: list[str] = Field(default_factory=list)


class TraitState(StrictModel):
    """An observed trait count; breakpoint may be omitted when unknown."""

    name: str = Field(min_length=1)
    count: int = Field(ge=0)
    breakpoint: int | None = Field(default=None, ge=1)


class GameState(StrictModel):
    """Objective facts supplied about the current match."""

    patch: str | None = None
    stage: str | None = None
    level: int | None = Field(default=None, ge=1, le=10)
    xp: int | None = Field(default=None, ge=0)
    gold: int | None = Field(default=None, ge=0)
    hp: int | None = Field(default=None, ge=0, le=100)
    board: list[UnitState] = Field(default_factory=list)
    bench: list[UnitState] = Field(default_factory=list)
    items: list[str] = Field(default_factory=list)
    traits: list[TraitState] = Field(default_factory=list)
    augments: list[str] = Field(default_factory=list)
    shop: list[UnitState] = Field(default_factory=list)
    target_champion: str | None = None
    target_cost: int | None = Field(default=None, ge=1, le=5)
    comp_name: str | None = None
    core_units: list[str] = Field(default_factory=list)
    user_question: str | None = None


class ToolErrorResolution(StrEnum):
    """Deterministic control action attached to a structured tool failure."""

    RETRY = "retry"
    REQUEST_CONTEXT = "request_context"
    TERMINAL = "terminal"


class ToolError(StrictModel):
    """A structured error safe to return across the tool boundary."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False
    resolution: ToolErrorResolution = ToolErrorResolution.TERMINAL

    @model_validator(mode="after")
    def validate_resolution(self) -> ToolError:
        if self.resolution == ToolErrorResolution.RETRY and not self.retryable:
            raise ValueError("retry resolution requires retryable=true")
        if self.resolution != ToolErrorResolution.RETRY and self.retryable:
            raise ValueError("retryable=true requires retry resolution")
        return self


class ToolMetadata(StrictModel):
    """Provenance and version metadata for a tool response."""

    game_patch: str | None = None
    data_version: str | None = None
    set_name: str | None = None
    source: str | None = None
    source_type: str | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)
    freshness: str | None = None


class ToolResult(StrictModel):
    """Normalized result returned by every tool."""

    tool_call_id: str = Field(min_length=1)
    success: bool
    data: Any | None = None
    metadata: ToolMetadata = Field(default_factory=ToolMetadata)
    error: ToolError | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> ToolResult:
        if self.success and self.error is not None:
            raise ValueError("successful tool results cannot contain an error")
        if not self.success and self.error is None:
            raise ValueError("failed tool results must contain an error")
        return self


class ToolCallRecord(StrictModel):
    """Timing and outcome record for one tool invocation."""

    tool_call: ToolCall
    result: ToolResult
    started_at: datetime
    finished_at: datetime
    latency_ms: float = Field(ge=0)


class StrategyRecommendation(StrictModel):
    """A qualitative strategy conclusion with current-task knowledge references."""

    text: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)


class FinalResponse(StrictModel):
    """User-facing recommendation produced by the agent."""

    recommendation: str = Field(min_length=1)
    reasoning: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    used_tools: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    strategy_recommendation: StrategyRecommendation | None = None


class AgentStatus(StrEnum):
    RUNNING = "running"
    WAITING_FOR_CONTEXT = "waiting_for_context"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_STEPS_REACHED = "max_steps_reached"


class AgentAction(StrEnum):
    TOOL_CALL = "tool_call"
    FINAL = "final"
    REQUEST_CONTEXT = "request_context"


class NumericClaimStatus(StrEnum):
    """Deterministic validation outcome for one declared user-facing number."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    MISMATCHED = "mismatched"
    INVALID_REFERENCE = "invalid_reference"


class QualitativeClaimStatus(StrEnum):
    """Small deterministic status set for strategy recommendation provenance."""

    SUPPORTED = "supported"
    INVALID_REFERENCE = "invalid_reference"
    WRONG_SOURCE_TYPE = "wrong_source_type"
    WRONG_VERSION = "wrong_version"
    IRRELEVANT_EVIDENCE = "irrelevant_evidence"


class NumericDerivation(StrictModel):
    """Small arithmetic derivation over evidence references and numeric literals."""

    operation: str
    inputs: list[str | int | float] = Field(min_length=2)


class ToolArgumentClaim(StrictModel):
    """Current-task provenance for one proposed tool argument."""

    argument_name: str = Field(min_length=1)
    value: Any
    evidence_ref: str | None = None
    derivation: NumericDerivation | None = None

    @model_validator(mode="after")
    def validate_source_shape(self) -> ToolArgumentClaim:
        if self.evidence_ref is not None and self.derivation is not None:
            raise ValueError("tool argument claim cannot have both evidence_ref and derivation")
        return self


class ToolCall(StrictModel):
    """A model-requested tool invocation with provider-internal provenance."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    argument_claims: list[ToolArgumentClaim] = Field(default_factory=list)


class NumericClaim(StrictModel):
    """Provider-declared provenance for one user-visible numerical claim."""

    value: int | float | str
    unit: str | None = None
    meaning: str = Field(min_length=1)
    evidence_ref: str | None = None
    derivation: NumericDerivation | None = None

    @model_validator(mode="after")
    def validate_source_shape(self) -> NumericClaim:
        if self.evidence_ref is not None and self.derivation is not None:
            raise ValueError("numeric claim cannot have both evidence_ref and derivation")
        return self


class AgentDecision(StrictModel):
    """Exactly one high-level action selected by the LLM for a runtime step."""

    action: AgentAction
    tool_call: ToolCall | None = None
    final_response: FinalResponse | None = None
    missing_context: list[str] = Field(default_factory=list)
    reason: str | None = None
    numeric_claims: list[NumericClaim] = Field(default_factory=list)
    numeric_claims_declared: bool = False

    @model_validator(mode="after")
    def validate_action_payload(self) -> AgentDecision:
        if self.action == AgentAction.TOOL_CALL:
            if (
                self.tool_call is None
                or self.final_response is not None
                or self.missing_context
                or self.numeric_claims
                or self.numeric_claims_declared
            ):
                raise ValueError("tool_call action requires only tool_call payload")
        elif self.action == AgentAction.FINAL:
            if self.final_response is None or self.tool_call is not None or self.missing_context:
                raise ValueError("final action requires only final_response payload")
        elif (
            not self.missing_context
            or self.tool_call is not None
            or self.final_response is not None
            or self.numeric_claims
            or self.numeric_claims_declared
        ):
            raise ValueError("request_context action requires only non-empty missing_context")
        return self
