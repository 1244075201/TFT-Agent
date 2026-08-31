"""Bounded agent loop and decision routing."""

import logging
from time import perf_counter

from tft_agent.agent.state import AgentState
from tft_agent.core.schemas import (
    AgentAction,
    AgentStatus,
    ToolCall,
    ToolCallRecord,
    ToolError,
    ToolResult,
    utc_now,
)
from tft_agent.llm.client import LLMClient
from tft_agent.mcp_client.client import MCPClient

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Route typed decisions while enforcing termination and observability."""

    def __init__(self, llm_client: LLMClient, mcp_client: MCPClient) -> None:
        self._llm_client = llm_client
        self._mcp_client = mcp_client

    async def run(self, state: AgentState) -> AgentState:
        """Run until a terminal decision, failure, or configured step limit."""

        if state.status != AgentStatus.RUNNING:
            return state

        try:
            tools = await self._mcp_client.list_tools()
            while state.step_count < state.max_steps:
                decision = await self._llm_client.generate(state, tools)
                state.step_count += 1

                if decision.action == AgentAction.FINAL:
                    state.final_response = decision.final_response
                    state.status = AgentStatus.COMPLETED
                    return state

                if decision.action == AgentAction.REQUEST_CONTEXT:
                    state.missing_context = decision.missing_context
                    state.status = AgentStatus.WAITING_FOR_CONTEXT
                    return state

                # AgentDecision validation guarantees this payload exists.
                await self._execute_tool(state, decision.tool_call)  # type: ignore[arg-type]

            state.status = AgentStatus.MAX_STEPS_REACHED
            return state
        except Exception as exc:  # adapter failures terminate the bounded run cleanly
            logger.exception("Agent runtime failed")
            state.status = AgentStatus.FAILED
            state.failure_reason = str(exc)
            return state

    async def _execute_tool(self, state: AgentState, tool_call: ToolCall) -> None:
        started_at = utc_now()
        started_clock = perf_counter()
        try:
            result = await self._mcp_client.call_tool(tool_call)
        except Exception as exc:
            logger.warning("Tool call %s failed: %s", tool_call.name, exc)
            result = ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                error=ToolError(code="tool_execution_failed", message=str(exc), retryable=False),
            )
        finished_at = utc_now()
        state.observations.append(result)
        state.tool_history.append(
            ToolCallRecord(
                tool_call=tool_call,
                result=result,
                started_at=started_at,
                finished_at=finished_at,
                latency_ms=(perf_counter() - started_clock) * 1000,
            )
        )
