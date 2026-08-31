from tft_agent.agent.parser import GameStateParser


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
