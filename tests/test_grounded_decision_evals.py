from datetime import UTC, datetime

from tft_agent.agent.state import AgentState
from tft_agent.core.schemas import (
    AgentStatus,
    FinalResponse,
    GameState,
    ToolCall,
    ToolCallRecord,
    ToolResult,
)
from tft_agent.evals.grounded_decision_runner import (
    GroundingCase,
    GroundingDataset,
    evaluate_grounding,
)


def _case() -> GroundingCase:
    return GroundingCase(
        id="fixture",
        slice="unqueried_comparison",
        prompt="fixture",
        expected_status="completed",
        allow_waiting_for_context=False,
        required_tools=["get_shop_odds"],
        allowed_shop_levels=[7, 8],
        forbidden_patterns=[],
        exact_hit_patterns=[],
        notes="fixture",
    )


def _state(final_text: str, queried_levels: list[int]) -> AgentState:
    now = datetime.now(UTC)
    history = [
        ToolCallRecord(
            tool_call=ToolCall(id=str(level), name="get_shop_odds", arguments={"level": level}),
            result=ToolResult(
                tool_call_id=str(level),
                success=True,
                data={"level": level, "odds": {"cost_4": 0.10 if level == 7 else 0.30}},
            ),
            started_at=now,
            finished_at=now,
            latency_ms=0,
        )
        for level in queried_levels
    ]
    return AgentState(
        game_state=GameState(level=7, target_cost=4, user_question="fixture"),
        goal="fixture",
        status=AgentStatus.COMPLETED,
        final_response=FinalResponse(recommendation=final_text),
        tool_history=history,
    )


def test_grounding_dataset_has_required_slices() -> None:
    dataset = GroundingDataset.model_validate_json(
        open("evals/grounded_decision_v0_2_1_cases.json", encoding="utf-8").read()
    )
    assert len(dataset.cases) == 14
    assert len({case.slice for case in dataset.cases}) == 8


def test_eval_flags_unqueried_level_probability() -> None:
    result = evaluate_grounding(_case(), _state("7级是10%，8级是15%。", [7]))

    assert result.unqueried_level_probability_claim is True
    assert result.passed is False


def test_eval_accepts_queried_level_comparison() -> None:
    result = evaluate_grounding(_case(), _state("7级是10%，8级是30%。", [7, 8]))

    assert result.unqueried_level_probability_claim is False
    assert result.passed is True


def test_eval_flags_chinese_quantity_expression() -> None:
    result = evaluate_grounding(_case(), _state("大概还能撑三五波。", [7]))

    assert result.chinese_numeral_unsupported_expression is True
    assert result.passed is False
