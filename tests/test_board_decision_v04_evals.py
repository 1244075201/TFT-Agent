import json
from pathlib import Path

from tft_agent.agent.parser import GameStateParser
from tft_agent.agent.state import AgentState
from tft_agent.core.evidence import build_evidence_registry
from tft_agent.evals.board_decision_v04_runner import BoardDataset
from tft_agent.llm.evidence import build_evidence_context
from tft_agent.llm.prompts import SYSTEM_PROMPT_VERSION, build_system_prompt


def test_v04_dataset_parses_expected_rich_state_deterministically() -> None:
    dataset = BoardDataset.model_validate_json(
        Path("evals/board_decision_v0_4_cases.json").read_text(encoding="utf-8")
    )

    assert dataset.dataset_version == "v0.4.0"
    assert len(dataset.cases) >= 8
    for case in dataset.cases:
        game_state = GameStateParser().parse_text(case.prompt)
        registry = build_evidence_registry(AgentState(game_state=game_state, goal=case.prompt))
        for path, expected in case.expected_state.items():
            assert registry[f"game_state.{path}"] == expected, json.dumps(
                game_state.model_dump(mode="json"), ensure_ascii=False
            )


def test_v04_no_explosion_cases_allow_no_tools() -> None:
    dataset = BoardDataset.model_validate_json(
        Path("evals/board_decision_v0_4_cases.json").read_text(encoding="utf-8")
    )
    cases = [case for case in dataset.cases if case.slice == "champion_tool_explosion"]

    assert cases
    assert all(case.expected_tools == case.allowed_tools == [] for case in cases)


def test_v04_prompt_exposes_rich_facts_and_forbids_champion_tool_explosion() -> None:
    state = AgentState(
        game_state=GameStateParser().parse_text(
            "棋盘：Ahri 2星（蓝霸符），Sett 2星；备战席：Garen 1星；散件：拳套。"
        ),
        goal="summarize",
    )

    prompt = build_system_prompt([], build_evidence_context(state))

    assert SYSTEM_PROMPT_VERSION == "v0.4.0"
    assert '"board":[{"name":"Ahri","star_level":2,"items":["蓝霸符"]}' in prompt
    assert "do not call\nget_champion_info once per board or bench unit" in prompt
