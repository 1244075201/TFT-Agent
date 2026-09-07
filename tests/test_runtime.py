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
    ToolArgumentClaim,
    ToolCall,
    ToolError,
    ToolErrorResolution,
    ToolMetadata,
    ToolResult,
)
from tft_agent.mcp_client.client import MCPToolDefinition
from tft_agent.vision.schemas import StateConflict


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


class ArgumentMCPClient(FakeMCPClient):
    async def list_tools(self) -> list[MCPToolDefinition]:
        return [
            MCPToolDefinition(
                name="lookup",
                description="Look up a numeric fixture",
                input_schema={
                    "type": "object",
                    "properties": {"level": {"type": "integer"}},
                    "required": ["level"],
                    "additionalProperties": False,
                },
            )
        ]


class KnowledgeMCPClient(FakeMCPClient):
    async def list_tools(self) -> list[MCPToolDefinition]:
        return [
            MCPToolDefinition(
                name="search_game_knowledge",
                description="Search controlled knowledge",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "strategy_topic": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            )
        ]


def make_state(max_steps: int = 3) -> AgentState:
    return AgentState(game_state=GameState(), goal="decide", max_steps=max_steps)


def test_rich_state_scopes_out_unrequested_champion_lookup() -> None:
    tools = [
        MCPToolDefinition(name="get_champion_info", description="identity", input_schema={}),
        MCPToolDefinition(name="get_shop_odds", description="odds", input_schema={}),
        MCPToolDefinition(name="search_game_knowledge", description="knowledge", input_schema={}),
    ]
    game_state = GameState.model_validate(
        {
            "board": [{"name": "Ahri", "star_level": 2}],
            "user_question": "根据我的棋盘给出定性调整建议",
        }
    )

    scoped = AgentRuntime._scope_tools_for_state(tools, game_state)

    assert scoped == []


def test_rich_state_keeps_champion_lookup_for_explicit_cost_request() -> None:
    tools = [MCPToolDefinition(name="get_champion_info", description="identity", input_schema={})]
    game_state = GameState.model_validate(
        {
            "board": [{"name": "Ahri"}],
            "user_question": "请查询 Ahri 是几费英雄",
        }
    )

    assert AgentRuntime._scope_tools_for_state(tools, game_state) == tools


def test_named_comp_strategy_scopes_out_unneeded_shop_odds() -> None:
    tools = [
        MCPToolDefinition(name="get_shop_odds", description="odds", input_schema={}),
        MCPToolDefinition(name="search_game_knowledge", description="knowledge", input_schema={}),
    ]
    game_state = GameState(
        level=7,
        comp_name="Vanguard Aphelios",
        user_question="这套阵容现在该搜还是上人口？",
    )

    assert [tool.name for tool in AgentRuntime._scope_tools_for_state(tools, game_state)] == [
        "search_game_knowledge"
    ]


def test_unknown_community_author_question_scopes_out_knowledge_search() -> None:
    tools = [
        MCPToolDefinition(name="search_game_knowledge", description="knowledge", input_schema={})
    ]
    game_state = GameState(user_question="某社区作者 FictionalCoach 推荐什么阵容？")

    assert AgentRuntime._scope_tools_for_state(tools, game_state) == []


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
async def test_runtime_attributes_strategy_sources_only_from_tool_results() -> None:
    result_payload = ToolResult(
        tool_call_id="placeholder",
        success=True,
        data={
            "results": [
                {
                    "title": "Blossom Ahri guide",
                    "url": "https://example.test/guide",
                    "source_type": "community_strategy",
                }
            ]
        },
    )
    llm = FakeLLMClient(
        [
            AgentDecision(
                action=AgentAction.TOOL_CALL,
                tool_call=ToolCall(id="rag", name="lookup", arguments={}),
            ),
            AgentDecision(
                action=AgentAction.FINAL,
                final_response=FinalResponse(
                    recommendation="Use the retrieved community view cautiously.",
                    sources=["https://invented.invalid"],
                ),
            ),
        ]
    )

    result = await AgentRuntime(llm, ResultMCPClient(result_payload)).run(make_state())

    assert result.final_response is not None
    assert result.final_response.sources == ["Blossom Ahri guide — https://example.test/guide"]


@pytest.mark.asyncio
async def test_runtime_routes_request_context() -> None:
    llm = FakeLLMClient(
        [AgentDecision(action=AgentAction.REQUEST_CONTEXT, missing_context=["current shop"])]
    )

    result = await AgentRuntime(llm, FakeMCPClient()).run(make_state())

    assert result.status == AgentStatus.WAITING_FOR_CONTEXT
    assert result.missing_context == ["current shop"]


@pytest.mark.asyncio
async def test_runtime_requests_material_user_vision_conflict_before_llm() -> None:
    state = AgentState(
        game_state=GameState(level=8, target_cost=4, user_question="我该D还是升人口？"),
        goal="我该D还是升人口？",
        state_conflicts=[
            StateConflict(
                field="level",
                user_value=8,
                vision_value=7,
                sources=["user", "vision:obs-1"],
            )
        ],
    )
    llm = FakeLLMClient([])

    result = await AgentRuntime(llm, FakeMCPClient()).run(state)

    assert result.status == AgentStatus.WAITING_FOR_CONTEXT
    assert "conflicting level" in result.missing_context[0]
    assert llm.calls == 0


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
                tool_call=ToolCall(id="invalid", name="lookup", arguments={"level": 7}),
            ),
            AgentDecision(
                action=AgentAction.FINAL,
                final_response=FinalResponse(recommendation="Arguments were rejected"),
            ),
        ]
    )
    mcp = FakeMCPClient()

    grounded_state = make_state()
    grounded_state.game_state.level = 7
    result = await AgentRuntime(llm, mcp).run(grounded_state)

    assert result.status == AgentStatus.COMPLETED
    assert mcp.calls == []
    assert result.observations[0].error is not None
    assert result.observations[0].error.code == "INVALID_TOOL_ARGUMENTS"


@pytest.mark.asyncio
async def test_unsupported_tool_argument_waits_without_execution() -> None:
    llm = FakeLLMClient(
        [
            AgentDecision(
                action=AgentAction.TOOL_CALL,
                tool_call=ToolCall(id="unsupported", name="lookup", arguments={"level": 1}),
            )
        ]
    )
    mcp = ArgumentMCPClient()

    result = await AgentRuntime(llm, mcp).run(make_state())

    assert result.status == AgentStatus.WAITING_FOR_CONTEXT
    assert mcp.calls == []
    assert result.trace.tool_argument_validation.blocked_tool_calls == 1


@pytest.mark.asyncio
async def test_mismatched_tool_argument_claim_is_blocked() -> None:
    call = ToolCall(
        id="mismatch",
        name="lookup",
        arguments={"level": 8},
        argument_claims=[
            ToolArgumentClaim(argument_name="level", value=8, evidence_ref="game_state.level")
        ],
    )
    state = make_state()
    state.game_state.level = 7
    mcp = ArgumentMCPClient()

    result = await AgentRuntime(
        FakeLLMClient([AgentDecision(action=AgentAction.TOOL_CALL, tool_call=call)]), mcp
    ).run(state)

    assert result.status == AgentStatus.WAITING_FOR_CONTEXT
    assert mcp.calls == []
    assert result.trace.tool_argument_validation.mismatched_argument_claims == 1


@pytest.mark.asyncio
async def test_supported_tool_argument_executes() -> None:
    call = ToolCall(
        id="supported",
        name="lookup",
        arguments={"level": 7},
        argument_claims=[
            ToolArgumentClaim(argument_name="level", value=7, evidence_ref="game_state.level")
        ],
    )
    state = make_state()
    state.game_state.level = 7
    mcp = ArgumentMCPClient()
    llm = FakeLLMClient(
        [
            AgentDecision(action=AgentAction.TOOL_CALL, tool_call=call),
            AgentDecision(
                action=AgentAction.FINAL,
                final_response=FinalResponse(recommendation="Grounded result"),
            ),
        ]
    )

    result = await AgentRuntime(llm, mcp).run(state)

    assert result.status == AgentStatus.COMPLETED
    assert len(mcp.calls) == 1
    assert result.trace.tool_argument_validation.executed_tool_calls == 1


@pytest.mark.asyncio
async def test_knowledge_query_is_canonicalized_to_original_user_question() -> None:
    state = AgentState(
        game_state=GameState(user_question="Original complete question"),
        goal="Original complete question",
    )
    mcp = KnowledgeMCPClient()
    llm = FakeLLMClient(
        [
            AgentDecision(
                action=AgentAction.TOOL_CALL,
                tool_call=ToolCall(
                    id="search",
                    name="search_game_knowledge",
                    arguments={"query": "provider paraphrase"},
                ),
            ),
            AgentDecision(
                action=AgentAction.FINAL,
                final_response=FinalResponse(recommendation="Grounded result"),
            ),
        ]
    )

    result = await AgentRuntime(llm, mcp).run(state)

    assert result.status == AgentStatus.COMPLETED
    assert mcp.calls[0].arguments["query"] == "Original complete question"
    assert result.trace.tool_argument_validation.executed_tool_calls == 1


@pytest.mark.asyncio
async def test_knowledge_transport_default_is_not_treated_as_user_claim() -> None:
    state = AgentState(
        game_state=GameState(user_question="Original complete question"),
        goal="Original complete question",
    )
    mcp = KnowledgeMCPClient()
    llm = FakeLLMClient(
        [
            AgentDecision(
                action=AgentAction.TOOL_CALL,
                tool_call=ToolCall(
                    id="search",
                    name="search_game_knowledge",
                    arguments={"query": "Original complete question", "top_k": 4},
                ),
            ),
            AgentDecision(
                action=AgentAction.FINAL,
                final_response=FinalResponse(recommendation="Grounded result"),
            ),
        ]
    )

    result = await AgentRuntime(llm, mcp).run(state)

    assert result.status == AgentStatus.COMPLETED
    assert mcp.calls[0].arguments == {"query": "Original complete question"}


@pytest.mark.asyncio
async def test_unknown_strategy_topic_is_omitted_instead_of_forcing_false_no_result() -> None:
    state = AgentState(
        game_state=GameState(user_question="How should I play at Stage 4?"),
        goal="How should I play at Stage 4?",
    )
    mcp = KnowledgeMCPClient()
    llm = FakeLLMClient(
        [
            AgentDecision(
                action=AgentAction.TOOL_CALL,
                tool_call=ToolCall(
                    id="search",
                    name="search_game_knowledge",
                    arguments={
                        "query": "How should I play at Stage 4?",
                        "strategy_topic": "Stage 4",
                    },
                ),
            ),
            AgentDecision(
                action=AgentAction.FINAL,
                final_response=FinalResponse(recommendation="Grounded result"),
            ),
        ]
    )

    result = await AgentRuntime(llm, mcp).run(state)

    assert result.status == AgentStatus.COMPLETED
    assert mcp.calls[0].arguments == {"query": "How should I play at Stage 4?"}


@pytest.mark.asyncio
async def test_invalid_tool_argument_reference_is_blocked() -> None:
    call = ToolCall(
        id="invalid-ref",
        name="lookup",
        arguments={"level": 7},
        argument_claims=[
            ToolArgumentClaim(argument_name="level", value=7, evidence_ref="game_state.fake_level")
        ],
    )
    state = make_state()
    state.game_state.level = 7
    mcp = ArgumentMCPClient()

    result = await AgentRuntime(
        FakeLLMClient([AgentDecision(action=AgentAction.TOOL_CALL, tool_call=call)]), mcp
    ).run(state)

    assert result.status == AgentStatus.WAITING_FOR_CONTEXT
    assert mcp.calls == []
    assert result.trace.tool_argument_validation.invalid_reference_argument_claims == 1


@pytest.mark.asyncio
async def test_next_value_derivation_executes_when_current_value_is_evidence() -> None:
    call = ToolCall(id="derived", name="lookup", arguments={"level": 8})
    state = make_state()
    state.game_state.level = 7
    mcp = ArgumentMCPClient()
    llm = FakeLLMClient(
        [
            AgentDecision(action=AgentAction.TOOL_CALL, tool_call=call),
            AgentDecision(
                action=AgentAction.FINAL,
                final_response=FinalResponse(recommendation="Compared levels"),
            ),
        ]
    )

    result = await AgentRuntime(llm, mcp).run(state)

    assert result.status == AgentStatus.COMPLETED
    assert len(mcp.calls) == 1
    claim = mcp.calls[0].argument_claims[0]
    assert claim.derivation is not None
    assert claim.derivation.operation == "add"
