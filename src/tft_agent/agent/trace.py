"""Small structured trace models that exclude prompts, secrets, and hidden reasoning."""

from datetime import datetime
from uuid import uuid4

from pydantic import Field

from tft_agent.core.schemas import AgentStatus, StrictModel, utc_now


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

    @property
    def total_latency_ms(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds() * 1000
