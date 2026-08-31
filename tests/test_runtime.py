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
    ToolError,
    ToolErrorResolution,
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
                name="lookup",
                description="Look up a fixture",
                input_schema={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "additionalProperties": False,
                },
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


class ResultMCPClient(FakeMCPClient):
    def __init__(self, result: ToolResult) -> None:
        super().__init__()
        self.result = result

    async def call_tool(self, tool_call: ToolCall) -> ToolResult:
        self.calls.append(tool_call)
        return self.result.model_copy(update={"tool_call_id": tool_call.id})


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
                final_response=FinalResponse(
                    recommendation="Hold this round",
                    used_tools=["invented_tool"],
                    sources=["invented source"],
                ),
            ),
        ]
    )
    mcp = FakeMCPClient()

    result = await AgentRuntime(llm, mcp).run(make_state())

    assert result.status == AgentStatus.COMPLETED
    assert result.step_count == 2
    assert result.final_response is not None
    assert result.final_response.used_tools == ["lookup"]
    assert result.final_response.sources == ["test fixture"]
    assert result.trace.available_tool_names == ["lookup"]
    assert result.trace.final_used_tools == ["lookup"]
    assert result.trace.final_sources == ["test fixture"]
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
async def test_tool_exception_becomes_terminal_observation_without_second_llm_call() -> None:
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
    assert llm.calls == 1
    assert result.final_response is not None
    assert result.final_response.recommendation == "transport unavailable"


@pytest.mark.asyncio
async def test_ambiguity_stops_adversarial_llm_before_candidate_followup() -> None:
    llm = FakeLLMClient(
        [
            AgentDecision(
                action=AgentAction.TOOL_CALL,
                tool_call=ToolCall(id="ambiguous", name="lookup", arguments={}),
            ),
            AgentDecision(
                action=AgentAction.TOOL_CALL,
                tool_call=ToolCall(
                    id="unauthorized",
                    name="lookup",
                    arguments={"name": "candidate-1"},
                ),
            ),
        ]
    )
    mcp = ResultMCPClient(
        ToolResult(
            tool_call_id="placeholder",
            success=False,
            data={"candidates": [{"champion_id": "candidate-1", "name": "Duplicate", "cost": 2}]},
            error=ToolError(
                code="AMBIGUOUS_FIXTURE",
                message="Multiple records matched; choose an ID",
                resolution=ToolErrorResolution.REQUEST_CONTEXT,
            ),
        )
    )

    result = await AgentRuntime(llm, mcp).run(make_state())

    assert result.status == AgentStatus.WAITING_FOR_CONTEXT
    assert llm.calls == 1
    assert len(mcp.calls) == 1
    assert result.missing_context == [
        "Multiple records matched; choose an ID",
        "candidate: champion_id=candidate-1, name=Duplicate, cost=2",
    ]


@pytest.mark.asyncio
async def test_terminal_error_blocks_adversarial_followup() -> None:
    llm = FakeLLMClient(
        [
            AgentDecision(
                action=AgentAction.TOOL_CALL,
                tool_call=ToolCall(id="missing", name="lookup", arguments={}),
            ),
            AgentDecision(
                action=AgentAction.TOOL_CALL,
                tool_call=ToolCall(id="guess", name="lookup", arguments={}),
            ),
        ]
    )
    mcp = ResultMCPClient(
        ToolResult(
            tool_call_id="placeholder",
            success=False,
            error=ToolError(code="NOT_FOUND", message="No reliable record was found"),
        )
    )

    result = await AgentRuntime(llm, mcp).run(make_state())

    assert result.status == AgentStatus.COMPLETED
    assert llm.calls == 1
    assert len(mcp.calls) == 1
    assert result.final_response is not None
    assert result.final_response.recommendation == "No reliable record was found"


@pytest.mark.asyncio
async def test_llm_failure_marks_run_failed() -> None:
    result = await AgentRuntime(FakeLLMClient([]), FakeMCPClient()).run(make_state())

    assert result.status == AgentStatus.FAILED
    assert result.failure_reason is not None


@pytest.mark.asyncio
async def test_runtime_rejects_arguments_outside_discovered_mcp_schema() -> None:
    llm = FakeLLMClient(
        [
            AgentDecision(
                action=AgentAction.TOOL_CALL,
                tool_call=ToolCall(id="invalid", name="lookup", arguments={"unknown": True}),
            ),
            AgentDecision(
                action=AgentAction.FINAL,
                final_response=FinalResponse(recommendation="Arguments were rejected"),
            ),
        ]
    )
    mcp = FakeMCPClient()

    result = await AgentRuntime(llm, mcp).run(make_state())

    assert result.status == AgentStatus.COMPLETED
    assert mcp.calls == []
    assert result.observations[0].error is not None
    assert result.observations[0].error.code == "INVALID_TOOL_ARGUMENTS"
