from tft_agent.agent.parser import GameStateParser
from tft_agent.agent.validator import GameStateValidator


def test_parser_extracts_explicit_v02_decision_facts() -> None:
    state = GameStateParser().parse_text(
        "我现在4-1，7级，35血，40金币，目标是一张4费卡，我该继续D吗？"
    )

    assert state.stage == "4-1"
    assert state.level == 7
    assert state.hp == 35
    assert state.gold == 40
    assert state.target_cost == 4


def test_parser_keeps_missing_decision_facts_unknown() -> None:
    state = GameStateParser().parse_text("我现在要不要D？")

    assert state.level is None
    assert state.hp is None
    assert state.gold is None
    assert state.target_cost is None


def test_parser_extracts_explicit_english_champion_target() -> None:
    state = GameStateParser().parse_text("我7级35血40金币，想找Alistar，我现在该D吗？")

    assert state.target_champion == "Alistar"


def test_parser_extracts_explicit_strategy_context_without_classifying_it() -> None:
    state = GameStateParser().parse_text(
        "我现在4-2，7级，35血，40金币，正在玩Blossom Ahri，核心单位：Ahri、Sett；该D吗？"
    )

    assert state.comp_name == "Blossom Ahri"
    assert state.core_units == ["Ahri", "Sett"]
    assert not hasattr(state, "should_roll")


def test_parser_does_not_treat_a_generic_comp_reference_as_a_name() -> None:
    state = GameStateParser().parse_text(
        "我现在4-2，7级，35血，40金币，正在玩这套阵容，核心质量不足，我该D还是升人口？"
    )

    assert state.comp_name is None
    assert state.core_units == []
    assert GameStateValidator().missing_strategy_context(state) == ["comp_name or core_units"]


def test_parser_extracts_explicit_named_strategy_entity() -> None:
    state = GameStateParser().parse_text("Solar Kayle现在怎么玩？")

    assert state.comp_name == "Solar Kayle"


def test_parser_extracts_named_comp_before_versioned_stage_question() -> None:
    state = GameStateParser().parse_text(
        "Blossom Ahri在18.1的Stage 4一般如何运营？请按收录攻略回答。"
    )

    assert state.comp_name == "Blossom Ahri"


def test_parser_preserves_numeric_suffix_in_explicit_comp_name() -> None:
    state = GameStateParser().parse_text("Draven Fast 9中途经济不够时攻略建议怎么转？")

    assert state.comp_name == "Draven Fast 9"


def test_parser_extracts_named_comp_before_guide_reference() -> None:
    state = GameStateParser().parse_text(
        "查DA_18_Ahri费用，再结合18.1 Blossom Ahri攻略说明这套阵容。"
    )

    assert state.comp_name == "Blossom Ahri"
    assert GameStateValidator().missing_strategy_context(state) == []
