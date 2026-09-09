"""DeepSeek/OpenAI-compatible message conversion and response normalization."""

import json
from typing import Any

from jsonschema import ValidationError as JSONSchemaValidationError
from jsonschema import validate
from pydantic import ValidationError as PydanticValidationError

from tft_agent.agent.state import AgentState
from tft_agent.core.exceptions import LLMProviderError
from tft_agent.core.schemas import (
    AgentAction,
    AgentDecision,
    FinalResponse,
    NumericClaim,
    StrategyRecommendation,
    ToolCall,
    ToolCallRecord,
)
from tft_agent.llm.evidence import (
    build_evidence_alias_registry,
    build_evidence_context,
    evidence_field_paths,
)
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
    ]
    messages.extend(
        {"role": item.role, "content": item.content}
        for item in state.messages[-20:]
        if item.role in {"user", "assistant"}
    )
    messages.append({"role": "user", "content": user_content})
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
        # OpenAI-compatible providers occasionally place literal newlines inside the
        # answer string. ``strict=False`` tolerates those control characters while
        # preserving JSON structure and claim validation at our domain boundary.
        payload = json.loads(content, strict=False)
    except json.JSONDecodeError:
        payload = _embedded_envelope(content, state)
        if payload is None:
            if "{" in content or "}" in content:
                raise _malformed(
                    "DeepSeek returned an incomplete or malformed JSON envelope"
                ) from None
            return _plain_final(content)
    if not isinstance(payload, dict):
        return _plain_final(content)
    if payload.get("action") == AgentAction.REQUEST_CONTEXT.value:
        if any(
            key in payload for key in ("supported_facts", "qualitative_recommendation", "answer")
        ):
            raise _malformed("final and request_context actions are mutually exclusive")
        missing = payload.get("missing_context")
        if isinstance(missing, str) and missing.strip():
            missing = [missing]
        elif isinstance(missing, list):
            missing = [_missing_context_field(item) for item in missing]
        if not isinstance(missing, list) or not missing or not all(missing):
            raise _malformed("request_context requires a non-empty string array")
        return AgentDecision(action=AgentAction.REQUEST_CONTEXT, missing_context=missing)
    if "answer" in payload:
        answer = payload.get("answer")
        raw_claims = payload.get("numeric_claims")
        if not isinstance(answer, str) or not answer.strip() or not isinstance(raw_claims, list):
            raise _malformed("numeric final envelope requires answer and numeric_claims array")
        try:
            claims = [
                NumericClaim.model_validate(_resolve_claim_aliases(item, state))
                for item in raw_claims
            ]
        except PydanticValidationError as exc:
            raise _malformed("numeric final envelope contains an invalid claim") from exc
        return AgentDecision(
            action=AgentAction.FINAL,
            final_response=FinalResponse(recommendation=answer.strip()),
            numeric_claims=claims,
            numeric_claims_declared=True,
        )
    if "supported_facts" in payload:
        raw_facts = payload.get("supported_facts")
        raw_recommendation = payload.get("qualitative_recommendation")
        if not isinstance(raw_facts, list) or not isinstance(raw_recommendation, (str, dict)):
            raise _malformed("segmented final envelope has invalid fields")
        strategy_recommendation: StrategyRecommendation | None = None
        if isinstance(raw_recommendation, dict):
            normalized_recommendation = dict(raw_recommendation)
            raw_refs = normalized_recommendation.get("evidence_refs")
            if isinstance(raw_refs, str):
                normalized_recommendation["evidence_refs"] = [raw_refs]
            if isinstance(normalized_recommendation.get("evidence_refs"), list):
                resolved_refs = [
                    _resolve_alias(ref, state) for ref in normalized_recommendation["evidence_refs"]
                ]
                normalized_recommendation["evidence_refs"] = _strategy_only_refs_when_supported(
                    resolved_refs, state
                )
            try:
                strategy_recommendation = StrategyRecommendation.model_validate(
                    normalized_recommendation
                )
            except PydanticValidationError as exc:
                if not _strategy_refs(state) and isinstance(raw_recommendation.get("text"), str):
                    recommendation = raw_recommendation["text"]
                else:
                    raise _malformed("segmented recommendation has invalid evidence refs") from exc
            else:
                recommendation = strategy_recommendation.text
        else:
            recommendation = raw_recommendation
            refs = _strategy_refs(state)
            if refs:
                strategy_recommendation = StrategyRecommendation(
                    text=recommendation,
                    evidence_refs=refs,
                )
        fact_texts: list[str] = []
        segment_claims: list[NumericClaim] = []
        try:
            for fact in raw_facts:
                if not isinstance(fact, dict) or not isinstance(fact.get("text"), str):
                    raise ValueError("invalid fact segment")
                raw_segment_claims = fact.get("numeric_claims")
                if raw_segment_claims is None:
                    raw_segment_claims = []
                if not isinstance(raw_segment_claims, list):
                    raise ValueError("invalid fact claims")
                fact_texts.append(fact["text"].strip())
                segment_claims.extend(
                    NumericClaim.model_validate(_resolve_claim_aliases(item, state))
                    for item in raw_segment_claims
                )
        except (PydanticValidationError, ValueError) as exc:
            raise _malformed("segmented final envelope contains an invalid fact") from exc
        # The user's decision comes first; verified facts support it instead of burying it.
        answer = "\n\n".join(text for text in [recommendation.strip(), *fact_texts] if text)
        if not answer:
            raise _malformed("segmented final envelope is empty")
        return AgentDecision(
            action=AgentAction.FINAL,
            final_response=FinalResponse(
                recommendation=answer,
                strategy_recommendation=strategy_recommendation,
            ),
            numeric_claims=segment_claims,
            numeric_claims_declared=True,
        )
    if payload.get("action") != AgentAction.FINAL.value:
        return _plain_final(content)
    final_recommendation = payload.get("recommendation")
    if not isinstance(final_recommendation, str) or not final_recommendation.strip():
        raise _malformed("final action requires a recommendation")
    return AgentDecision(
        action=AgentAction.FINAL,
        final_response=FinalResponse(
            recommendation=final_recommendation,
            reasoning=_string_list(payload.get("reasoning")),
            next_steps=_string_list(payload.get("next_steps")),
            alternatives=_string_list(payload.get("alternatives")),
        ),
    )


def _embedded_envelope(content: str, state: AgentState) -> dict[str, Any] | None:
    """Extract only a recognized complete envelope when surrounded by provider prose."""

    decoder = json.JSONDecoder(strict=False)
    matches: list[dict[str, Any]] = []
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and (
            payload.get("action") == "request_context"
            or "supported_facts" in payload
            or "answer" in payload
        ):
            matches.append(payload)
    if len(matches) > 1:
        raise _malformed("DeepSeek returned multiple conflicting final envelopes")
    if matches:
        state.trace.finalization.normalized_wrapped_envelope_count += 1
        return matches[0]
    return None


def _missing_context_field(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict) and isinstance(item.get("field"), str):
        field = item["field"]
        assert isinstance(field, str)
        return field.strip()
    return ""


def _resolve_alias(ref: Any, state: AgentState) -> Any:
    if not isinstance(ref, str):
        return ref
    return build_evidence_alias_registry(state).get(ref, ref)


def _resolve_claim_aliases(item: Any, state: AgentState) -> Any:
    if not isinstance(item, dict):
        return item
    resolved = dict(item)
    resolved["evidence_ref"] = _resolve_alias(resolved.get("evidence_ref"), state)
    derivation = resolved.get("derivation")
    if isinstance(derivation, dict) and isinstance(derivation.get("inputs"), list):
        resolved["derivation"] = {
            **derivation,
            "inputs": [_resolve_alias(value, state) for value in derivation["inputs"]],
        }
    return resolved


def _strategy_only_refs_when_supported(refs: list[Any], state: AgentState) -> list[Any]:
    """Drop auxiliary current-task facts only when valid strategy refs already support the claim."""

    strategy_refs = set(_strategy_refs(state))
    if not any(isinstance(ref, str) and ref in strategy_refs for ref in refs):
        return refs
    return [
        ref
        for ref in refs
        if not isinstance(ref, str)
        or ref in strategy_refs
        or not (ref.startswith("game_state.") or ref.startswith("tool."))
    ]


def _strategy_refs(state: AgentState) -> list[str]:
    refs: list[str] = []
    for record in state.tool_history:
        if record.tool_call.name != "search_game_knowledge" or not isinstance(
            record.result.data, dict
        ):
            continue
        chunks = record.result.data.get("results")
        if not isinstance(chunks, list):
            continue
        refs.extend(
            f"knowledge.{record.tool_call.id}.{chunk['chunk_id']}"
            for chunk in chunks
            if isinstance(chunk, dict)
            and isinstance(chunk.get("chunk_id"), str)
            and chunk.get("source_type") == "community_strategy"
            and chunk.get("document_type") == "strategy_guide"
            and chunk.get("game_patch") in {None, state.current_patch}
        )
    return refs


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
