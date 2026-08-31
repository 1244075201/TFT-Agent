"""Executable fallback tests for environments where pytest cannot be installed."""

import unittest
from collections.abc import Sequence

from pydantic import ValidationError

from tft_agent.agent.runtime import AgentRuntime
from tft_agent.agent.state import AgentState
from tft_agent.core.schemas import (
    AgentAction,
    AgentDecision,
    AgentStatus,
    FinalResponse,
    GameState,
    ToolCall,
    ToolError,
    ToolResult,
)
from tft_agent.mcp_client.client import MCPToolDefinition


class StubLLM:
    def __init__(self, decisions: list[AgentDecision]) -> None:
        self._decisions = iter(decisions)

    async def generate(
        self, state: AgentState, tools: Sequence[MCPToolDefinition]
    ) -> AgentDecision:
        return next(self._decisions)


class StubMCP:
    def __init__(self, fail: bool = False) -> None:
        self._fail = fail

    async def list_tools(self) -> list[MCPToolDefinition]:
        return [MCPToolDefinition(name="lookup", description="Fixture", input_schema={})]

    async def call_tool(self, tool_call: ToolCall) -> ToolResult:
        if self._fail:
            raise RuntimeError("fixture failure")
        return ToolResult(tool_call_id=tool_call.id, success=True, data={"fixture": True})


class SchemaTests(unittest.TestCase):
    def test_invalid_game_state_and_tool_result_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            GameState(gold=-1)
        with self.assertRaises(ValidationError):
            ToolResult(tool_call_id="x", success=False)

    def test_structured_tool_failure_is_valid(self) -> None:
        result = ToolResult(
            tool_call_id="x",
            success=False,
            error=ToolError(code="missing", message="not found", retryable=False),
        )
        self.assertFalse(result.success)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_failure_is_observed_before_final(self) -> None:
        decisions = [
            AgentDecision(
                action=AgentAction.TOOL_CALL,
                tool_call=ToolCall(id="x", name="lookup", arguments={}),
            ),
            AgentDecision(
                action=AgentAction.FINAL,
                final_response=FinalResponse(recommendation="done"),
            ),
        ]
        state = AgentState(game_state=GameState(), goal="test", max_steps=2)

        result = await AgentRuntime(StubLLM(decisions), StubMCP(fail=True)).run(state)

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(result.step_count, 2)
        self.assertFalse(result.observations[0].success)

    async def test_request_context_and_max_steps_are_terminal(self) -> None:
        waiting = await AgentRuntime(
            StubLLM([AgentDecision(action=AgentAction.REQUEST_CONTEXT, missing_context=["shop"])]),
            StubMCP(),
        ).run(AgentState(game_state=GameState(), goal="test"))
        self.assertEqual(waiting.status, AgentStatus.WAITING_FOR_CONTEXT)

        tool_decision = AgentDecision(
            action=AgentAction.TOOL_CALL,
            tool_call=ToolCall(id="x", name="lookup", arguments={}),
        )
        limited = await AgentRuntime(StubLLM([tool_decision]), StubMCP()).run(
            AgentState(game_state=GameState(), goal="test", max_steps=1)
        )
        self.assertEqual(limited.status, AgentStatus.MAX_STEPS_REACHED)


if __name__ == "__main__":
    unittest.main()
