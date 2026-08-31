"""Shared, transport-safe domain schemas."""

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
    user_question: str | None = None


class ToolCall(StrictModel):
    """A model-requested tool invocation."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


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
    def validate_resolution(self) -> "ToolError":
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
    def validate_outcome(self) -> "ToolResult":
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


class FinalResponse(StrictModel):
    """User-facing recommendation produced by the agent."""

    recommendation: str = Field(min_length=1)
    reasoning: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    used_tools: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


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


class AgentDecision(StrictModel):
    """Exactly one high-level action selected by the LLM for a runtime step."""

    action: AgentAction
    tool_call: ToolCall | None = None
    final_response: FinalResponse | None = None
    missing_context: list[str] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def validate_action_payload(self) -> "AgentDecision":
        if self.action == AgentAction.TOOL_CALL:
            if self.tool_call is None or self.final_response is not None or self.missing_context:
                raise ValueError("tool_call action requires only tool_call payload")
        elif self.action == AgentAction.FINAL:
            if self.final_response is None or self.tool_call is not None or self.missing_context:
                raise ValueError("final action requires only final_response payload")
        elif (
            not self.missing_context
            or self.tool_call is not None
            or self.final_response is not None
        ):
            raise ValueError("request_context action requires only non-empty missing_context")
        return self
