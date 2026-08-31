"""LLM client contract consumed by the agent runtime."""

from collections.abc import Sequence
from typing import Protocol

from tft_agent.agent.state import AgentState
from tft_agent.core.schemas import AgentDecision
from tft_agent.mcp_client.client import MCPToolDefinition


class LLMClient(Protocol):
    """Generate exactly one structured decision from current execution state."""

    async def generate(
        self, state: AgentState, tools: Sequence[MCPToolDefinition]
    ) -> AgentDecision: ...
