"""Mutable execution state kept separate from factual game state."""

from pydantic import Field, model_validator

from tft_agent.agent.trace import AgentTrace
from tft_agent.core.schemas import (
    AgentStatus,
    FinalResponse,
    GameState,
    StrictModel,
    ToolCallRecord,
    ToolResult,
)
from tft_agent.vision.schemas import StateConflict, VisionObservation, VisionTrace


class AgentMessage(StrictModel):
    """Provider-neutral message retained for the current run."""

    role: str = Field(min_length=1)
    content: str = Field(min_length=1)


class AgentState(StrictModel):
    """Bounded state of one agent execution; this is not long-term memory."""

    game_state: GameState
    goal: str = Field(min_length=1)
    messages: list[AgentMessage] = Field(default_factory=list)
    observations: list[ToolResult] = Field(default_factory=list)
    tool_history: list[ToolCallRecord] = Field(default_factory=list)
    current_patch: str | None = None
    step_count: int = Field(default=0, ge=0)
    max_steps: int = Field(default=8, ge=1, le=100)
    status: AgentStatus = AgentStatus.RUNNING
    final_response: FinalResponse | None = None
    missing_context: list[str] = Field(default_factory=list)
    failure_reason: str | None = None
    vision_observations: list[VisionObservation] = Field(default_factory=list)
    state_conflicts: list[StateConflict] = Field(default_factory=list)
    vision_traces: list[VisionTrace] = Field(default_factory=list)
    trace: AgentTrace = Field(default_factory=AgentTrace)

    @model_validator(mode="after")
    def validate_progress(self) -> "AgentState":
        if self.step_count > self.max_steps:
            raise ValueError("step_count cannot exceed max_steps")
        if self.status == AgentStatus.COMPLETED and self.final_response is None:
            raise ValueError("completed state requires final_response")
        if self.status == AgentStatus.WAITING_FOR_CONTEXT and not self.missing_context:
            raise ValueError("waiting state requires missing_context")
        return self
