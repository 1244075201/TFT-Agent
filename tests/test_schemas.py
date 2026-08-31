from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tft_agent.agent.state import AgentState
from tft_agent.core.schemas import (
    AgentAction,
    AgentDecision,
    AgentStatus,
    FinalResponse,
    GameState,
    ToolCall,
    ToolCallRecord,
    ToolError,
    ToolErrorResolution,
    ToolResult,
    UnitState,
)
from tft_agent.tft_mcp_server.schemas import ShopOddsRecord


def test_game_state_accepts_partial_facts_and_nested_units() -> None:
    state = GameState(gold=20, board=[UnitState(name="Test Unit", star_level=2)])

    assert state.patch is None
    assert state.board[0].star_level == 2


def test_game_state_accepts_v02_decision_facts() -> None:
    state = GameState(level=7, hp=35, gold=40, target_champion="Alistar", target_cost=4)

    assert state.target_cost == 4
    assert state.target_champion == "Alistar"


def test_shop_odds_record_requires_a_probability_distribution() -> None:
    payload = {
        "level": 7,
        "cost_1_probability": 0.16,
        "cost_2_probability": 0.30,
        "cost_3_probability": 0.43,
        "cost_4_probability": 0.10,
        "cost_5_probability": 0.01,
        "game_patch": "16.17",
        "source_version": "fixture",
        "source": "fixture",
        "source_type": "test_fixture",
        "retrieved_at": datetime.now(UTC),
    }
    assert ShopOddsRecord.model_validate(payload).cost_4_probability == 0.10
    payload["cost_4_probability"] = 0.20
    with pytest.raises(ValidationError):
        ShopOddsRecord.model_validate(payload)


@pytest.mark.parametrize("field,value", [("gold", -1), ("hp", 101), ("level", 0)])
def test_game_state_rejects_invalid_ranges(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        GameState.model_validate({field: value})


def test_agent_state_requires_final_response_when_completed() -> None:
    with pytest.raises(ValidationError):
        AgentState(game_state=GameState(), goal="answer", status=AgentStatus.COMPLETED)


def test_agent_state_rejects_step_count_above_limit() -> None:
    with pytest.raises(ValidationError):
        AgentState(game_state=GameState(), goal="answer", step_count=3, max_steps=2)


def test_tool_call_rejects_empty_identity() -> None:
    with pytest.raises(ValidationError):
        ToolCall(id="", name="lookup")


def test_agent_decision_enforces_action_payload() -> None:
    with pytest.raises(ValidationError):
        AgentDecision(action=AgentAction.TOOL_CALL)
    with pytest.raises(ValidationError):
        AgentDecision(action=AgentAction.REQUEST_CONTEXT, missing_context=[])

    final = FinalResponse(recommendation="Hold")
    decision = AgentDecision(action=AgentAction.FINAL, final_response=final)
    assert decision.final_response == final


def test_final_response_allows_a_minimal_fact_answer() -> None:
    response = FinalResponse(recommendation="Alistar is a 2-cost champion.")

    assert response.reasoning == []
    assert response.next_steps == []
    assert response.alternatives == []
    assert response.confidence is None


def test_tool_result_requires_error_for_failure() -> None:
    with pytest.raises(ValidationError):
        ToolResult(tool_call_id="call-1", success=False)


def test_tool_result_and_record_capture_structured_failure() -> None:
    call = ToolCall(id="call-1", name="get_item_info", arguments={"name": "X"})
    result = ToolResult(
        tool_call_id=call.id,
        success=False,
        error=ToolError(code="not_found", message="missing", retryable=False),
    )
    now = datetime.now(UTC)
    record = ToolCallRecord(
        tool_call=call,
        result=result,
        started_at=now,
        finished_at=now,
        latency_ms=0,
    )
    assert record.result.error is not None
    assert record.result.error.code == "not_found"


def test_tool_error_resolution_requires_consistent_retryability() -> None:
    context_error = ToolError(
        code="AMBIGUOUS",
        message="choose one",
        retryable=False,
        resolution=ToolErrorResolution.REQUEST_CONTEXT,
    )
    assert context_error.resolution == ToolErrorResolution.REQUEST_CONTEXT

    with pytest.raises(ValidationError):
        ToolError(
            code="TIMEOUT",
            message="try later",
            retryable=False,
            resolution=ToolErrorResolution.RETRY,
        )
