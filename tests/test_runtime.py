from collections.abc import Sequence

import pytest

from tft_agent.agent.runtime import AgentRuntime
from tft_agent.agent.state import AgentState
from tft_agent.core.schemas import (
    AgentAction,
    AgentDecision,
    AgentStatus,
    FinalResponse,
    GameState,
    ToolCall,
    ToolMetadata,
    ToolResult,
)
from tft_agent.mcp_client.client import MCPToolDefinition


class FakeLLMClient:
    def __init__(self, decisions: list[AgentDecision]) -> None:
        self.decisions = decisions
        self.calls = 0

    async def generate(
        self, state: AgentState, tools: Sequence[MCPToolDefinition]
    ) -> AgentDecision:
        decision = self.decisions[self.calls]
        self.calls += 1
        return decision


class FakeMCPClient:
    def __init__(self, fail_calls: bool = False) -> None:
        self.calls: list[ToolCall] = []
        self.fail_calls = fail_calls

    async def list_tools(self) -> list[MCPToolDefinition]:
        return [
            MCPToolDefinition(
                name="lookup", description="Look up a fixture", input_schema={"type": "object"}
            )
        ]

    async def call_tool(self, tool_call: ToolCall) -> ToolResult:
        self.calls.append(tool_call)
        if self.fail_calls:
            raise RuntimeError("transport unavailable")
        return ToolResult(
            tool_call_id=tool_call.id,
            success=True,
            data={"fixture": True},
            metadata=ToolMetadata(source="test fixture"),
        )


def make_state(max_steps: int = 3) -> AgentState:
    return AgentState(game_state=GameState(), goal="decide", max_steps=max_steps)


@pytest.mark.asyncio
async def test_runtime_routes_tool_then_final() -> None:
    tool_call = ToolCall(id="1", name="lookup", arguments={})
    llm = FakeLLMClient(
        [
            AgentDecision(action=AgentAction.TOOL_CALL, tool_call=tool_call),
            AgentDecision(
                action=AgentAction.FINAL,
                final_response=FinalResponse(recommendation="Hold this round"),
            ),
        ]
    )
    mcp = FakeMCPClient()

    result = await AgentRuntime(llm, mcp).run(make_state())

    assert result.status == AgentStatus.COMPLETED
    assert result.step_count == 2
    assert result.final_response is not None
    assert len(result.observations) == len(result.tool_history) == 1


@pytest.mark.asyncio
async def test_runtime_routes_request_context() -> None:
    llm = FakeLLMClient(
        [AgentDecision(action=AgentAction.REQUEST_CONTEXT, missing_context=["current shop"])]
    )

    result = await AgentRuntime(llm, FakeMCPClient()).run(make_state())

    assert result.status == AgentStatus.WAITING_FOR_CONTEXT
    assert result.missing_context == ["current shop"]


@pytest.mark.asyncio
async def test_runtime_stops_at_max_steps() -> None:
    calls = [
        AgentDecision(
            action=AgentAction.TOOL_CALL,
            tool_call=ToolCall(id=str(index), name="lookup", arguments={}),
        )
        for index in range(2)
    ]

    result = await AgentRuntime(FakeLLMClient(calls), FakeMCPClient()).run(make_state(2))

    assert result.status == AgentStatus.MAX_STEPS_REACHED
    assert result.step_count == 2


@pytest.mark.asyncio
async def test_tool_exception_becomes_observation_and_loop_continues() -> None:
    llm = FakeLLMClient(
        [
            AgentDecision(
                action=AgentAction.TOOL_CALL,
                tool_call=ToolCall(id="broken", name="lookup", arguments={}),
            ),
            AgentDecision(
                action=AgentAction.FINAL,
                final_response=FinalResponse(recommendation="Ask for another source"),
            ),
        ]
    )

    result = await AgentRuntime(llm, FakeMCPClient(fail_calls=True)).run(make_state())

    assert result.status == AgentStatus.COMPLETED
    assert result.observations[0].success is False
    assert result.observations[0].error is not None
    assert result.observations[0].error.code == "tool_execution_failed"


@pytest.mark.asyncio
async def test_llm_failure_marks_run_failed() -> None:
    result = await AgentRuntime(FakeLLMClient([]), FakeMCPClient()).run(make_state())

    assert result.status == AgentStatus.FAILED
    assert result.failure_reason is not None
