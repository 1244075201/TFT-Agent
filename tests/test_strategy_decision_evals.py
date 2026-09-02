from pathlib import Path

from tft_agent.evals.strategy_decision_v031_runner import StrategyDataset


def test_strategy_dataset_is_small_versioned_and_covers_required_slices() -> None:
    dataset = StrategyDataset.model_validate_json(
        Path("evals/strategy_decision_v0_3_1_cases.json").read_text(encoding="utf-8")
    )

    assert 12 <= len(dataset.cases) <= 16
    assert len({case.id for case in dataset.cases}) == len(dataset.cases)
    assert {
        "pure_strategy",
        "structured_only",
        "shop_only",
        "strategy_game_state",
        "structured_strategy",
        "missing_strategy_context",
        "no_answer",
        "hard_negative",
    } <= {case.slice for case in dataset.cases}
    assert len(dataset.manual_quality_rubric) == 7


def test_strategy_cases_forbid_unnecessary_tool_explosion() -> None:
    dataset = StrategyDataset.model_validate_json(
        Path("evals/strategy_decision_v0_3_1_cases.json").read_text(encoding="utf-8")
    )
    by_id = {case.id: case for case in dataset.cases}

    assert by_id["structured_001"].allowed_tools == ["get_champion_info"]
    assert by_id["shop_001"].allowed_tools == ["get_shop_odds"]
    assert by_id["pure_001"].allowed_tools == ["search_game_knowledge"]
    assert by_id["missing_001"].allowed_tools == []
