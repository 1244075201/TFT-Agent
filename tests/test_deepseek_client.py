import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APITimeoutError, AsyncOpenAI

from tft_agent.agent.state import AgentState
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
            content=json.dumps({"action": "final", "recommendation": "It costs 3."})
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
async def test_multiple_tool_calls_are_explicitly_rejected() -> None:
    create = AsyncMock(
        return_value=response(
            tool_calls=[
                provider_tool_call("one", '{"name":"A"}'),
                provider_tool_call("two", '{"name":"B"}'),
            ]
        )
    )

    with pytest.raises(LLMProviderError) as error:
        await client_with_create(create).generate(state(), [tool_definition()])

    assert error.value.code == "MULTIPLE_TOOL_CALLS_UNSUPPORTED"


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
