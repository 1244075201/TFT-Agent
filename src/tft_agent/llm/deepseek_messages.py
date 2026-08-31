"""DeepSeek/OpenAI-compatible message conversion and response normalization."""

import json
from typing import Any

from jsonschema import ValidationError as JSONSchemaValidationError
from jsonschema import validate

from tft_agent.agent.state import AgentState
from tft_agent.core.exceptions import LLMProviderError
from tft_agent.core.schemas import (
    AgentAction,
    AgentDecision,
    FinalResponse,
    ToolCall,
    ToolCallRecord,
)
from tft_agent.llm.evidence import build_evidence_context, evidence_field_paths
from tft_agent.llm.prompts import build_system_prompt
from tft_agent.llm.tool_adapter import to_function_schema
from tft_agent.mcp_client.client import MCPToolDefinition


def build_provider_messages(
    state: AgentState,
    available_tools: list[MCPToolDefinition],
) -> list[dict[str, Any]]:
    """Rebuild native tool-call history from provider-neutral execution state."""

    question = state.game_state.user_question or state.goal
    evidence = build_evidence_context(state)
    state.trace.grounded_evidence_fields = evidence_field_paths(evidence)
    user_content = question
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": build_system_prompt(available_tools, evidence),
        },
        {"role": "user", "content": user_content},
    ]
    for record in state.tool_history:
        call = record.tool_call
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": serialize_tool_evidence(record),
            }
        )
    return messages


def build_provider_tools(tools: list[MCPToolDefinition]) -> list[dict[str, Any]]:
    """Translate discovered MCP schemas without maintaining provider copies."""

    return [to_function_schema(tool) for tool in tools]


def normalize_provider_response(
    response: Any,
    available_tools: list[MCPToolDefinition],
    state: AgentState,
) -> AgentDecision:
    """Convert one provider response into a project decision model."""

    choices = getattr(response, "choices", None)
    if not choices or len(choices) != 1:
        raise _malformed("DeepSeek response must contain exactly one choice")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise _malformed("DeepSeek response choice has no message")
    tool_calls = list(getattr(message, "tool_calls", None) or [])
    if tool_calls:
        # Execute only the first returned call; the next step reassesses after real evidence.
        provider_call = tool_calls[0]
        function = getattr(provider_call, "function", None)
        call_id = getattr(provider_call, "id", None)
        name = getattr(function, "name", None)
        raw_arguments = getattr(function, "arguments", None)
        if not all(isinstance(value, str) and value for value in (call_id, name, raw_arguments)):
            raise _malformed("DeepSeek returned an incomplete tool call")
        assert isinstance(call_id, str)
        assert isinstance(name, str)
        assert isinstance(raw_arguments, str)
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise _malformed("DeepSeek returned malformed function arguments") from exc
        if not isinstance(arguments, dict):
            raise _malformed("DeepSeek function arguments must be a JSON object")
        definition = next((tool for tool in available_tools if tool.name == name), None)
        if definition is None:
            raise _malformed(f"DeepSeek requested an unavailable tool: {name}")
        try:
            validate(instance=arguments, schema=definition.input_schema)
        except JSONSchemaValidationError as exc:
            raise _malformed("DeepSeek function arguments do not match the MCP schema") from exc
        return AgentDecision(
            action=AgentAction.TOOL_CALL,
            tool_call=ToolCall(id=call_id, name=name, arguments=arguments),
        )

    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise _malformed("DeepSeek response contains neither a tool call nor final content")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        embedded = _embedded_request_context(content)
        return embedded if embedded is not None else _plain_final(content)
    if not isinstance(payload, dict):
        return _plain_final(content)
    if payload.get("action") == AgentAction.REQUEST_CONTEXT.value:
        missing = payload.get("missing_context")
        if (
            not isinstance(missing, list)
            or not missing
            or not all(isinstance(x, str) for x in missing)
        ):
            raise _malformed("request_context requires a non-empty string array")
        return AgentDecision(action=AgentAction.REQUEST_CONTEXT, missing_context=missing)
    if payload.get("action") != AgentAction.FINAL.value:
        return _plain_final(content)
    recommendation = payload.get("recommendation")
    if not isinstance(recommendation, str) or not recommendation.strip():
        raise _malformed("final action requires a recommendation")
    return AgentDecision(
        action=AgentAction.FINAL,
        final_response=FinalResponse(
            recommendation=recommendation,
            reasoning=_string_list(payload.get("reasoning")),
            next_steps=_string_list(payload.get("next_steps")),
            alternatives=_string_list(payload.get("alternatives")),
        ),
    )


def _embedded_request_context(content: str) -> AgentDecision | None:
    """Recognize the explicit decision contract even when surrounded by visible prose."""

    decoder = json.JSONDecoder()
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("action") != "request_context":
            continue
        missing = payload.get("missing_context")
        if isinstance(missing, list) and missing and all(isinstance(item, str) for item in missing):
            return AgentDecision(action=AgentAction.REQUEST_CONTEXT, missing_context=missing)
    return None


def _plain_final(content: str) -> AgentDecision:
    """Map visible non-tool output without guessing whether it is a question."""

    return AgentDecision(
        action=AgentAction.FINAL,
        final_response=FinalResponse(recommendation=content.strip()),
    )


def serialize_tool_evidence(record: ToolCallRecord) -> str:
    """Serialize one result with an explicit, field-derived grounding boundary."""

    result = record.result
    data = result.data if isinstance(result.data, dict) else result.data
    metadata = result.metadata.model_dump(mode="json", exclude_none=True)
    authoritative_fields = list(data) if isinstance(data, dict) else []
    authoritative_fields.extend(f"metadata.{field}" for field in metadata)
    payload = {
        "observation_type": "tool_evidence",
        "tool_name": record.tool_call.name,
        "tool_call_id": record.tool_call.id,
        "success": result.success,
        "authoritative_fields": authoritative_fields,
        "data": data,
        "provenance": metadata,
        "error": result.error.model_dump(mode="json") if result.error else None,
        "grounding_contract": (
            "Only authoritative_fields are established by this observation; do not infer "
            "missing version-sensitive facts from identifiers, names, or model memory."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _malformed("Final response list fields must be a string or string array")
    return value


def _malformed(message: str) -> LLMProviderError:
    return LLMProviderError(code="MALFORMED_PROVIDER_RESPONSE", message=message, retryable=False)
