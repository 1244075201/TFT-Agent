from datetime import UTC, datetime

from tft_agent.agent.state import AgentState
from tft_agent.core.schemas import (
    AgentStatus,
    GameState,
    ToolCall,
    ToolCallRecord,
    ToolError,
    ToolErrorResolution,
    ToolResult,
)
from tft_agent.evals.outcome_safety_runner import OutcomeCase, build_outcome_result


def record(call_id: str, result: ToolResult) -> ToolCallRecord:
    now = datetime.now(UTC)
    return ToolCallRecord(
        tool_call=ToolCall(id=call_id, name="get_champion_info", arguments={}),
        result=result,
        started_at=now,
        finished_at=now,
        latency_ms=1,
    )


def test_outcome_result_detects_unauthorized_followup_after_ambiguity() -> None:
    ambiguity = ToolResult(
        tool_call_id="first",
        success=False,
        data={"candidates": [{"champion_id": "one"}, {"champion_id": "two"}]},
        error=ToolError(
            code="AMBIGUOUS_CHAMPION",
            message="choose one",
            resolution=ToolErrorResolution.REQUEST_CONTEXT,
        ),
    )
    followup = ToolResult(tool_call_id="second", success=True, data={"champion_id": "one"})
    state = AgentState(
        game_state=GameState(),
        goal="query",
        status=AgentStatus.COMPLETED,
        final_response={"recommendation": "picked one"},
        tool_history=[record("first", ambiguity), record("second", followup)],
        observations=[ambiguity, followup],
    )
    case = OutcomeCase(
        id="ambiguity",
        slice="ambiguity",
        prompt="query",
        expected_status="waiting_for_context",
        expected_tool_error_code="AMBIGUOUS_CHAMPION",
        notes="fixture",
    )

    result = build_outcome_result(case, state)

    assert result.unauthorized_followup_tool_calls == 1
    assert result.passed is False


def test_outcome_result_accepts_runtime_context_stop() -> None:
    ambiguity = ToolResult(
        tool_call_id="first",
        success=False,
        error=ToolError(
            code="AMBIGUOUS_CHAMPION",
            message="choose one",
            resolution=ToolErrorResolution.REQUEST_CONTEXT,
        ),
    )
    state = AgentState(
        game_state=GameState(),
        goal="query",
        status=AgentStatus.WAITING_FOR_CONTEXT,
        missing_context=["choose one"],
        tool_history=[record("first", ambiguity)],
        observations=[ambiguity],
    )
    case = OutcomeCase(
        id="ambiguity",
        slice="ambiguity",
        prompt="query",
        expected_status="waiting_for_context",
        expected_tool_error_code="AMBIGUOUS_CHAMPION",
        notes="fixture",
    )

    result = build_outcome_result(case, state)

    assert result.request_context_triggered is True
    assert result.unauthorized_followup_tool_calls == 0
    assert result.passed is True
