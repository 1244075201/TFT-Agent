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
    ToolResult,
    UnitState,
)


def test_game_state_accepts_partial_facts_and_nested_units() -> None:
    state = GameState(gold=20, board=[UnitState(name="Test Unit", star_level=2)])

    assert state.patch is None
    assert state.board[0].star_level == 2


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
