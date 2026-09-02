import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APITimeoutError, AsyncOpenAI

from tft_agent.agent.state import AgentState
from tft_agent.core.evidence import build_evidence_alias_registry
from tft_agent.core.exceptions import LLMProviderError
from tft_agent.core.schemas import (
    AgentAction,
    GameState,
    ToolCall,
    ToolCallRecord,
    ToolMetadata,
    ToolResult,
)
from tft_agent.llm.deepseek_client import DeepSeekLLMClient
from tft_agent.mcp_client.client import MCPToolDefinition


def tool_definition() -> MCPToolDefinition:
    return MCPToolDefinition(
        name="get_champion_info",
        description="Authoritative champion facts",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    )


def response(*, content: str | None = None, tool_calls: list[Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14),
    )


def provider_tool_call(call_id: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name="get_champion_info", arguments=arguments),
    )


def client_with_create(create: AsyncMock) -> DeepSeekLLMClient:
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return DeepSeekLLMClient(
        api_key="unit-test-placeholder",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        max_retries=0,
        client=cast(AsyncOpenAI, sdk),
    )


def state() -> AgentState:
    return AgentState(
        game_state=GameState(user_question="What is the cost?"),
        goal="What is the cost?",
    )


def numeric_envelope(answer: str, claims: list[dict[str, Any]]) -> SimpleNamespace:
    return response(content=json.dumps({"answer": answer, "numeric_claims": claims}))


def strategy_state() -> AgentState:
    agent_state = AgentState(
        game_state=GameState(comp_name="Blossom Ahri", user_question="Should I roll?"),
        goal="Should I roll?",
        current_patch="18.1",
    )
    now = datetime.now(UTC)
    champion_call = ToolCall(id="champion-1", name="get_champion_info", arguments={})
    champion_result = ToolResult(
        tool_call_id=champion_call.id,
        success=True,
        data={"name": "Ahri", "cost": 4},
        metadata=ToolMetadata(source="riot-data-dragon"),
    )
    strategy_call = ToolCall(id="search-1", name="search_game_knowledge", arguments={})
    strategy_result = ToolResult(
        tool_call_id=strategy_call.id,
        success=True,
        data={
            "results": [
                {
                    "chunk_id": "blossom:2",
                    "source_type": "community_strategy",
                    "document_type": "strategy_guide",
                    "game_patch": "18.1",
                    "strategy_comp_names": ["Blossom Ahri"],
                    "strategy_topics": ["roll_decision"],
                    "text": "Improve Ahri and Sett before leveling.",
                }
            ]
        },
    )
    for call, result in ((champion_call, champion_result), (strategy_call, strategy_result)):
        agent_state.tool_history.append(
            ToolCallRecord(
                tool_call=call,
                result=result,
                started_at=now,
                finished_at=now,
                latency_ms=0,
            )
        )
    return agent_state


@pytest.mark.asyncio
async def test_normal_final_response_and_non_thinking_request() -> None:
    create = AsyncMock(
        return_value=response(
            content=json.dumps(
                {
                    "action": "final",
                    "recommendation": "I can help.",
                    "reasoning": "The question needs no tool.",
                }
            )
        )
    )
    agent_state = state()

    decision = await client_with_create(create).generate(agent_state, [tool_definition()])

    assert decision.action == AgentAction.FINAL
    assert decision.final_response is not None
    assert decision.final_response.recommendation == "I can help."
    assert decision.final_response.reasoning == ["The question needs no tool."]
    assert create.await_args is not None
    request = create.await_args.kwargs
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "response_format" not in request
    assert request["tools"][0]["function"]["parameters"] == tool_definition().input_schema


@pytest.mark.asyncio
async def test_plain_visible_content_maps_to_final_response() -> None:
    create = AsyncMock(return_value=response(content="I can explain the available TFT tool."))

    decision = await client_with_create(create).generate(state(), [tool_definition()])

    assert decision.action == AgentAction.FINAL
    assert decision.final_response is not None
    assert decision.final_response.recommendation == "I can explain the available TFT tool."


@pytest.mark.asyncio
async def test_numeric_envelope_tolerates_literal_newline_in_answer() -> None:
    raw_envelope = (
        '{"answer":"You have 35 HP.\nKeep the recommendation qualitative.",'
        '"numeric_claims":[{"value":35,"unit":"hp","meaning":"current hp",'
        '"evidence_ref":"game_state.hp"}]}'
    )
    create = AsyncMock(return_value=response(content=raw_envelope))
    client = client_with_create(create)
    agent_state = state()
    agent_state.game_state.hp = 35

    decision = await client.generate(agent_state, [])

    assert decision.final_response is not None
    assert decision.final_response.recommendation == (
        "You have 35 HP.\nKeep the recommendation qualitative."
    )
    assert agent_state.trace.numeric_claim_validation is not None
    assert agent_state.trace.numeric_claim_validation.supported_claims == 1
    assert agent_state.trace.numeric_claim_validation.correction_attempted is False


@pytest.mark.asyncio
async def test_segmented_final_envelope_is_assembled() -> None:
    content = json.dumps(
        {
            "supported_facts": [
                {
                    "text": "You currently have 35 HP.",
                    "numeric_claims": [
                        {
                            "value": 35,
                            "unit": "hp",
                            "meaning": "current hp",
                            "evidence_ref": "game_state.hp",
                        }
                    ],
                }
            ],
            "qualitative_recommendation": "Keep the recommendation qualitative.",
        }
    )
    create = AsyncMock(return_value=response(content=content))
    agent_state = state()
    agent_state.game_state.hp = 35

    decision = await client_with_create(create).generate(agent_state, [])

    assert decision.final_response is not None
    assert decision.final_response.recommendation == (
        "Keep the recommendation qualitative.\n\nYou currently have 35 HP."
    )
    assert agent_state.trace.numeric_claim_validation is not None
    assert agent_state.trace.numeric_claim_validation.supported_claims == 1


@pytest.mark.asyncio
async def test_segmented_envelope_is_extracted_from_provider_preface() -> None:
    envelope = json.dumps(
        {
            "supported_facts": [
                {
                    "text": "Current gold is 40.",
                    "numeric_claims": [
                        {
                            "value": 40,
                            "unit": "gold",
                            "meaning": "current gold",
                            "evidence_ref": "game_state.gold",
                        }
                    ],
                }
            ],
            "qualitative_recommendation": "This matches the supplied fact.",
        }
    )
    create = AsyncMock(return_value=response(content=f"Here is the result.\n{envelope}"))
    agent_state = state()
    agent_state.game_state.gold = 40

    decision = await client_with_create(create).generate(agent_state, [])

    assert decision.final_response is not None
    assert decision.final_response.recommendation.startswith("This matches the supplied fact.")
    assert create.await_count == 1
    assert agent_state.trace.finalization.normalized_wrapped_envelope_count == 1


@pytest.mark.asyncio
async def test_incomplete_json_envelope_is_repaired_once() -> None:
    create = AsyncMock(
        side_effect=[
            response(content='{"supported_facts":['),
            response(
                content=json.dumps(
                    {
                        "supported_facts": [],
                        "qualitative_recommendation": "Answer without unsupported precision.",
                    }
                )
            ),
        ]
    )
    agent_state = state()

    decision = await client_with_create(create).generate(agent_state, [])

    assert decision.action == AgentAction.FINAL
    assert create.await_count == 2
    assert agent_state.trace.finalization.invalid_envelope_count == 1
    assert agent_state.trace.finalization.repair_count == 1
    assert agent_state.trace.finalization.repair_success == 1
    assert len(agent_state.trace.llm_calls) == 2


@pytest.mark.asyncio
async def test_two_conflicting_envelopes_are_rejected_then_repaired() -> None:
    first = json.dumps({"supported_facts": [], "qualitative_recommendation": "First answer."})
    second = json.dumps(
        {"supported_facts": [], "qualitative_recommendation": "Conflicting answer."}
    )
    repaired = json.dumps({"supported_facts": [], "qualitative_recommendation": "Single answer."})
    create = AsyncMock(
        side_effect=[response(content=f"{first}\n{second}"), response(content=repaired)]
    )
    agent_state = state()

    decision = await client_with_create(create).generate(agent_state, [])

    assert decision.final_response is not None
    assert decision.final_response.recommendation == "Single answer."
    assert agent_state.trace.finalization.invalid_envelope_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported", ["你大概还能撑三五轮。", "建议留十来金币。"])
async def test_unicode_quantity_is_repaired_before_user_output(unsupported: str) -> None:
    create = AsyncMock(
        side_effect=[
            numeric_envelope(unsupported, []),
            numeric_envelope("当前证据只支持定性建议。", []),
        ]
    )
    agent_state = state()

    decision = await client_with_create(create).generate(agent_state, [])

    assert decision.final_response is not None
    assert unsupported not in decision.final_response.recommendation
    assert agent_state.trace.numeric_claim_validation is not None
    assert agent_state.trace.numeric_claim_validation.correction_attempted is True
    assert agent_state.trace.numeric_claim_validation.correction_succeeded is True


@pytest.mark.asyncio
async def test_numeric_repair_may_request_missing_context() -> None:
    create = AsyncMock(
        side_effect=[
            numeric_envelope(
                "Roll at level 7.",
                [{"value": 7, "unit": "level", "meaning": "guessed level"}],
            ),
            response(
                content=json.dumps(
                    {
                        "action": "request_context",
                        "missing_context": ["player level", "target cost"],
                    }
                )
            ),
        ]
    )

    decision = await client_with_create(create).generate(state(), [])

    assert decision.action == AgentAction.REQUEST_CONTEXT
    assert decision.missing_context == ["player level", "target cost"]


@pytest.mark.asyncio
async def test_explicit_request_context_envelope_is_normalized() -> None:
    create = AsyncMock(
        return_value=response(
            content=json.dumps({"action": "request_context", "missing_context": ["champion name"]})
        )
    )

    decision = await client_with_create(create).generate(state(), [tool_definition()])

    assert decision.action == AgentAction.REQUEST_CONTEXT
    assert decision.missing_context == ["champion name"]


@pytest.mark.asyncio
async def test_structured_missing_context_claim_is_normalized() -> None:
    create = AsyncMock(
        return_value=response(
            content=json.dumps(
                {
                    "action": "request_context",
                    "missing_context": [
                        {
                            "field": "champion name",
                            "reason": "needed for lookup",
                            "required_for": "champion identity",
                        }
                    ],
                }
            )
        )
    )

    decision = await client_with_create(create).generate(state(), [tool_definition()])

    assert decision.action == AgentAction.REQUEST_CONTEXT
    assert decision.missing_context == ["champion name"]


@pytest.mark.asyncio
async def test_premature_request_for_resolved_context_is_repaired() -> None:
    agent_state = strategy_state()
    strategy_alias = next(
        alias
        for alias, ref in build_evidence_alias_registry(agent_state).items()
        if ref == "knowledge.search-1.blossom:2"
    )
    create = AsyncMock(
        side_effect=[
            response(
                content=json.dumps(
                    {"action": "request_context", "missing_context": ["champion cost"]}
                )
            ),
            response(
                content=json.dumps(
                    {
                        "supported_facts": [],
                        "qualitative_recommendation": {
                            "text": "Improve the core pair before leveling.",
                            "evidence_refs": [strategy_alias],
                        },
                    }
                )
            ),
        ]
    )
    decision = await client_with_create(create).generate(agent_state, [])

    assert decision.action == AgentAction.FINAL
    assert agent_state.trace.finalization.premature_request_context_count == 1
    assert agent_state.trace.finalization.repair_count == 1
    assert agent_state.trace.finalization.repair_success == 1


@pytest.mark.asyncio
async def test_missing_corpus_evidence_is_not_treated_as_user_context() -> None:
    create = AsyncMock(
        side_effect=[
            response(
                content=json.dumps(
                    {"action": "request_context", "missing_context": ["strategy evidence"]}
                )
            ),
            response(
                content=json.dumps(
                    {
                        "supported_facts": [],
                        "qualitative_recommendation": (
                            "The available corpus does not establish an answer."
                        ),
                    }
                )
            ),
        ]
    )
    agent_state = state()

    decision = await client_with_create(create).generate(agent_state, [])

    assert decision.action == AgentAction.FINAL
    assert agent_state.trace.finalization.premature_request_context_count == 1


@pytest.mark.asyncio
async def test_current_task_evidence_alias_resolves_to_canonical_reference() -> None:
    agent_state = strategy_state()
    strategy_alias = next(
        alias
        for alias, ref in build_evidence_alias_registry(agent_state).items()
        if ref == "knowledge.search-1.blossom:2"
    )
    game_alias = next(
        alias
        for alias, ref in build_evidence_alias_registry(agent_state).items()
        if ref == "game_state.comp_name"
    )
    create = AsyncMock(
        return_value=response(
            content=json.dumps(
                {
                    "supported_facts": [],
                    "qualitative_recommendation": {
                        "text": "Improve the core pair before leveling.",
                        "evidence_refs": [strategy_alias, game_alias],
                    },
                }
            )
        )
    )

    decision = await client_with_create(create).generate(agent_state, [])

    assert decision.final_response is not None
    recommendation = decision.final_response.strategy_recommendation
    assert recommendation is not None
    assert recommendation.evidence_refs == ["knowledge.search-1.blossom:2"]


@pytest.mark.asyncio
async def test_unknown_evidence_alias_remains_invalid() -> None:
    agent_state = strategy_state()
    invalid = {
        "supported_facts": [],
        "qualitative_recommendation": {
            "text": "Improve the core pair before leveling.",
            "evidence_refs": ["E999"],
        },
    }
    create = AsyncMock(
        side_effect=[
            response(content=json.dumps(invalid)),
            response(content=json.dumps(invalid)),
        ]
    )

    decision = await client_with_create(create).generate(agent_state, [])

    assert decision.final_response is not None
    assert decision.final_response.strategy_recommendation is None
    assert agent_state.trace.finalization.invalid_evidence_ref_count == 1
    assert agent_state.trace.numeric_claim_validation is not None
    assert agent_state.trace.numeric_claim_validation.fallback_used is True


@pytest.mark.asyncio
async def test_final_and_request_context_are_mutually_exclusive() -> None:
    invalid = {
        "action": "request_context",
        "missing_context": ["level"],
        "supported_facts": [],
        "qualitative_recommendation": "Conflicting final.",
    }
    valid = {"supported_facts": [], "qualitative_recommendation": "Single final."}
    create = AsyncMock(
        side_effect=[
            response(content=json.dumps(invalid)),
            response(content=json.dumps(valid)),
        ]
    )
    agent_state = state()

    decision = await client_with_create(create).generate(agent_state, [])

    assert decision.action == AgentAction.FINAL
    assert agent_state.trace.finalization.invalid_envelope_count == 1


@pytest.mark.asyncio
async def test_accumulated_multi_tool_evidence_and_aliases_reach_final_prompt() -> None:
    agent_state = strategy_state()
    strategy_alias = next(
        alias
        for alias, ref in build_evidence_alias_registry(agent_state).items()
        if ref == "knowledge.search-1.blossom:2"
    )
    create = AsyncMock(
        return_value=response(
            content=json.dumps(
                {
                    "supported_facts": [],
                    "qualitative_recommendation": {
                        "text": "Improve the core pair before leveling.",
                        "evidence_refs": [strategy_alias],
                    },
                }
            )
        )
    )
    await client_with_create(create).generate(agent_state, [])

    assert create.await_args is not None
    system = create.await_args.kwargs["messages"][0]["content"]
    assert '"cost":4' in system
    assert '"chunk_id":"blossom:2"' in system
    assert f'"{strategy_alias}":{{"kind":"knowledge"' in system
    assert "knowledge.search-1.blossom:2" not in system


@pytest.mark.asyncio
async def test_native_tool_call_and_id_are_preserved() -> None:
    create = AsyncMock(
        return_value=response(
            tool_calls=[provider_tool_call("call-provider-1", '{"name":"Unique Hero"}')]
        )
    )

    decision = await client_with_create(create).generate(state(), [tool_definition()])

    assert decision.action == AgentAction.TOOL_CALL
    assert decision.tool_call == ToolCall(
        id="call-provider-1",
        name="get_champion_info",
        arguments={"name": "Unique Hero"},
    )


@pytest.mark.asyncio
async def test_tool_result_continuation_uses_native_protocol_messages() -> None:
    create = AsyncMock(
        return_value=response(
            content=json.dumps(
                {
                    "answer": "It costs 3.",
                    "numeric_claims": [
                        {
                            "value": 3,
                            "unit": "cost",
                            "meaning": "champion cost",
                            "evidence_ref": "tool.call-provider-2.data.cost",
                        }
                    ],
                }
            )
        )
    )
    agent_state = state()
    call = ToolCall(
        id="call-provider-2",
        name="get_champion_info",
        arguments={"name": "Unique Hero"},
    )
    result = ToolResult(
        tool_call_id=call.id,
        success=True,
        data={"name": "Unique Hero", "cost": 3},
        metadata=ToolMetadata(source="riot-data-dragon"),
    )
    now = datetime.now(UTC)
    agent_state.observations.append(result)
    agent_state.tool_history.append(
        ToolCallRecord(
            tool_call=call,
            result=result,
            started_at=now,
            finished_at=now,
            latency_ms=1,
        )
    )

    decision = await client_with_create(create).generate(agent_state, [tool_definition()])

    assert decision.action == AgentAction.FINAL
    assert create.await_args is not None
    messages = create.await_args.kwargs["messages"]
    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["tool_calls"][0]["id"] == "call-provider-2"
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == "call-provider-2"
    evidence = json.loads(messages[-1]["content"])
    assert evidence["observation_type"] == "tool_evidence"
    assert evidence["authoritative_fields"] == [
        "name",
        "cost",
        "metadata.source",
        "metadata.retrieved_at",
    ]
    assert "set_name" not in evidence["authoritative_fields"]
    assert decision.final_response is not None
    assert decision.final_response.used_tools == []


@pytest.mark.asyncio
async def test_capability_context_is_derived_from_live_tools_without_raw_schema() -> None:
    create = AsyncMock(return_value=response(content="Only champion facts are available."))

    await client_with_create(create).generate(state(), [tool_definition()])

    assert create.await_args is not None
    system_prompt = create.await_args.kwargs["messages"][0]["content"]
    assert "get_champion_info: Authoritative champion facts" in system_prompt
    assert '"properties"' not in system_prompt
    assert "item database" not in system_prompt


@pytest.mark.asyncio
async def test_malformed_function_arguments_fail_before_tool_execution() -> None:
    create = AsyncMock(return_value=response(tool_calls=[provider_tool_call("bad", "not-json")]))

    with pytest.raises(LLMProviderError) as error:
        await client_with_create(create).generate(state(), [tool_definition()])

    assert error.value.code == "MALFORMED_PROVIDER_RESPONSE"


@pytest.mark.asyncio
async def test_multiple_tool_calls_execute_only_the_first_in_returned_order() -> None:
    create = AsyncMock(
        return_value=response(
            tool_calls=[
                provider_tool_call("one", '{"name":"A"}'),
                provider_tool_call("two", '{"name":"B"}'),
            ]
        )
    )

    decision = await client_with_create(create).generate(state(), [tool_definition()])

    assert decision.tool_call is not None
    assert decision.tool_call.id == "one"


@pytest.mark.asyncio
async def test_request_context_contract_is_recognized_inside_visible_prose() -> None:
    create = AsyncMock(
        return_value=response(
            content='Please clarify. {"action":"request_context","missing_context":["level"]}'
        )
    )

    decision = await client_with_create(create).generate(state(), [tool_definition()])

    assert decision.action == AgentAction.REQUEST_CONTEXT
    assert decision.missing_context == ["level"]


@pytest.mark.asyncio
async def test_unsupported_claims_trigger_one_successful_correction() -> None:
    agent_state = AgentState(
        game_state=GameState(hp=35, user_question="Should I roll?"),
        goal="Should I roll?",
    )
    create = AsyncMock(
        side_effect=[
            numeric_envelope(
                "You have 35 HP. Roll to 20 gold; you may survive 3 rounds.",
                [
                    {
                        "value": 35,
                        "unit": "hp",
                        "meaning": "current hp",
                        "evidence_ref": "game_state.hp",
                    },
                    {"value": 20, "unit": "gold", "meaning": "roll threshold"},
                    {"value": 3, "unit": "rounds", "meaning": "survival estimate"},
                ],
            ),
            numeric_envelope(
                "You have 35 HP. Keep the recommendation qualitative.",
                [
                    {
                        "value": 35,
                        "unit": "hp",
                        "meaning": "current hp",
                        "evidence_ref": "game_state.hp",
                    }
                ],
            ),
        ]
    )

    decision = await client_with_create(create).generate(agent_state, [tool_definition()])

    assert create.await_count == 2
    assert decision.final_response is not None
    assert "20" not in decision.final_response.recommendation
    assert agent_state.trace.numeric_claim_validation is not None
    assert agent_state.trace.numeric_claim_validation.supported_claims == 1
    assert agent_state.trace.numeric_claim_validation.unsupported_claims == 2
    assert agent_state.trace.numeric_claim_validation.correction_succeeded is True
    assert agent_state.trace.numeric_claim_validation.fallback_used is False


@pytest.mark.asyncio
async def test_second_unsupported_claim_uses_safe_fallback_without_third_call() -> None:
    agent_state = AgentState(
        game_state=GameState(hp=35, user_question="Should I roll?"),
        goal="Should I roll?",
    )
    create = AsyncMock(
        side_effect=[
            numeric_envelope(
                "Roll to 20 gold.",
                [{"value": 20, "unit": "gold", "meaning": "roll threshold"}],
            ),
            numeric_envelope(
                "Roll to 25 gold.",
                [{"value": 25, "unit": "gold", "meaning": "replacement threshold"}],
            ),
        ]
    )

    decision = await client_with_create(create).generate(agent_state, [tool_definition()])

    assert create.await_count == 2
    assert decision.final_response is not None
    assert "precise quantities" in decision.final_response.recommendation
    assert agent_state.trace.numeric_claim_validation is not None
    assert agent_state.trace.numeric_claim_validation.fallback_used is True


@pytest.mark.asyncio
async def test_safe_fallback_preserves_grounded_qualitative_strategy_after_bad_number() -> None:
    agent_state = AgentState(
        game_state=GameState(hp=35, comp_name="Blossom Ahri", user_question="D还是升8？"),
        goal="D还是升8？",
        current_patch="18.1",
    )
    now = datetime.now(UTC)
    agent_state.tool_history.append(
        ToolCallRecord(
            tool_call=ToolCall(
                id="search-1",
                name="search_game_knowledge",
                arguments={
                    "strategy_comp_name": "Blossom Ahri",
                    "strategy_topic": "core_quality",
                },
            ),
            result=ToolResult(
                tool_call_id="search-1",
                success=True,
                data={
                    "results": [
                        {
                            "chunk_id": "blossom:2",
                            "source_type": "community_strategy",
                            "document_type": "strategy_guide",
                            "game_patch": "18.1",
                            "strategy_comp_names": ["Blossom Ahri"],
                            "strategy_topics": ["core_quality", "roll_decision"],
                            "text": "Improve Ahri and Sett before leveling.",
                        }
                    ]
                },
            ),
            started_at=now,
            finished_at=now,
            latency_ms=0,
        )
    )
    initial = {
        "supported_facts": [
            {
                "text": "D到20金币。",
                "numeric_claims": [{"value": 20, "unit": "gold", "meaning": "roll threshold"}],
            }
        ],
        "qualitative_recommendation": {
            "text": (
                "先在4-2阶段提升Ahri和Sett的核心质量，而不是升8。 "
                "第4阶段等两张牌二星后再走。 At Stage 4-2, do not rush level 8."
            ),
            "evidence_refs": ["knowledge.search-1.blossom:2"],
        },
    }
    create = AsyncMock(
        side_effect=[
            response(content=json.dumps(initial, ensure_ascii=False)),
            numeric_envelope(
                "D到25金币。",
                [{"value": 25, "unit": "gold", "meaning": "replacement threshold"}],
            ),
        ]
    )

    decision = await client_with_create(create).generate(agent_state, [tool_definition()])

    assert decision.final_response is not None
    assert "20" not in decision.final_response.recommendation
    assert "25" not in decision.final_response.recommendation
    assert "4-2" not in decision.final_response.recommendation
    assert "升8" not in decision.final_response.recommendation
    assert "Stage 4-2" not in decision.final_response.recommendation
    assert "level 8" not in decision.final_response.recommendation
    assert "第4" not in decision.final_response.recommendation
    assert "二星" not in decision.final_response.recommendation
    assert "两张" not in decision.final_response.recommendation
    assert "提升Ahri和Sett的核心质量" in decision.final_response.recommendation
    assert decision.final_response.strategy_recommendation is not None
    trace = agent_state.trace.qualitative_recommendation_validation
    assert trace is not None
    assert trace.preserved_after_repair is True


@pytest.mark.asyncio
async def test_provider_timeout_is_mapped_and_bounded() -> None:
    create = AsyncMock(
        side_effect=APITimeoutError(request=httpx.Request("POST", "https://api.deepseek.com"))
    )

    with pytest.raises(LLMProviderError) as error:
        await client_with_create(create).generate(state(), [tool_definition()])

    assert error.value.code == "PROVIDER_TIMEOUT"
    assert error.value.retryable is True
    assert create.await_count == 1


def test_missing_api_configuration_is_rejected_without_sdk_call() -> None:
    with pytest.raises(LLMProviderError) as error:
        DeepSeekLLMClient(
            api_key=" ",
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
        )

    assert error.value.code == "MISSING_CONFIGURATION"
