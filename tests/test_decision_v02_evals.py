from tft_agent.agent.state import AgentState
from tft_agent.core.schemas import (
    AgentStatus,
    FinalResponse,
    GameState,
    ToolCall,
    ToolCallRecord,
    ToolError,
    ToolResult,
    utc_now,
)
from tft_agent.evals.decision_v02_runner import DecisionCase, DecisionDataset, build_decision_result


def test_decision_dataset_has_required_slices() -> None:
    dataset = DecisionDataset.model_validate_json(
        open("evals/decision_v0_2_cases.json", encoding="utf-8").read()
    )
    assert len(dataset.cases) == 12
    assert {case.slice for case in dataset.cases} == {
        "direct_shop_odds",
        "simple_grounded_decision",
        "missing_context",
        "wrong_tool_temptation",
        "two_tool_useful_chain",
    }


def test_decision_result_checks_exact_tool_sequence() -> None:
    case = DecisionCase(
        id="fixture",
        slice="direct_shop_odds",
        prompt="fixture",
        expected_tool_sequence=["get_shop_odds"],
        expected_status="completed",
        notes="fixture",
    )
    now = utc_now()
    call = ToolCall(id="1", name="get_shop_odds", arguments={"level": 7})
    result = ToolResult(tool_call_id="1", success=True, data={"level": 7})
    state = AgentState(
        game_state=GameState(level=7, user_question="fixture"),
        goal="fixture",
        status=AgentStatus.COMPLETED,
        final_response=FinalResponse(recommendation="fixture"),
        tool_history=[
            ToolCallRecord(
                tool_call=call, result=result, started_at=now, finished_at=now, latency_ms=0
            )
        ],
    )
    state.trace.game_state_snapshot = state.game_state.model_dump(mode="json")

    measured = build_decision_result(case, state)
    assert measured.passed is True
    assert measured.actual_tool_sequence == ["get_shop_odds"]
    assert measured.game_state["level"] == 7


def test_decision_result_rejects_a_correct_sequence_with_tool_failure() -> None:
    case = DecisionCase(
        id="failure",
        slice="two_tool_useful_chain",
        prompt="fixture",
        expected_tool_sequence=["get_shop_odds"],
        expected_status="completed",
        notes="fixture",
    )
    now = utc_now()
    call = ToolCall(id="1", name="get_shop_odds", arguments={"level": 7})
    result = ToolResult(
        tool_call_id="1",
        success=False,
        error=ToolError(code="SHOP_ODDS_NOT_FOUND", message="missing"),
    )
    state = AgentState(
        game_state=GameState(level=7, user_question="fixture"),
        goal="fixture",
        status=AgentStatus.COMPLETED,
        final_response=FinalResponse(recommendation="missing"),
        tool_history=[
            ToolCallRecord(
                tool_call=call,
                result=result,
                started_at=now,
                finished_at=now,
                latency_ms=0,
            )
        ],
    )

    assert build_decision_result(case, state).passed is False
