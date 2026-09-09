import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tft_agent.agent.state import AgentMessage
from tft_agent.app.pipeline import (
    AnswerContext,
    ProductMode,
    QueryKind,
    RetrievedKnowledge,
    build_retrieval_plan,
    context_gaps,
    understand_query,
)
from tft_agent.core.config import Settings
from tft_agent.core.schemas import GameState
from tft_agent.llm.grounded_decision import GroundedDecisionClient, normalize_decision_output
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)


@pytest.mark.parametrize(
    "question",
    ["这套阵容怎么玩", "为什么选它", "推荐阵容", "A和B比较", "该不该搜牌", "怎么转阵"],
)
def test_decision_language_always_uses_strategy_path(question: str) -> None:
    result = understand_query(question, GameState(), ProductMode.MATCH_GUIDANCE)
    assert result.kind == QueryKind.STRATEGY_DECISION


def test_comp_browser_is_deterministic_for_explicit_browsing() -> None:
    result = understand_query("显示当前版本阵容列表", GameState(), ProductMode.COMP_BROWSER)
    assert result.kind == QueryKind.DETERMINISTIC_LOOKUP


def test_unknown_intent_defaults_to_strategy() -> None:
    result = understand_query("黑荆棘裁决", GameState(), ProductMode.MATCH_GUIDANCE)
    assert result.kind == QueryKind.STRATEGY_DECISION


def test_plan_uses_text_retrieval_and_exposes_non_blocking_gaps() -> None:
    state = GameState(level=7, item_components=["反曲弓"])
    understanding = understand_query("我该怎么办", state, ProductMode.MATCH_GUIDANCE)
    plan = build_retrieval_plan(understanding, state, "18.1d")
    assert plan.retrieve_strategy_text is True
    assert "反曲弓" in plan.static_entity_names
    assert "board" not in context_gaps(state)
    assert "stage" in context_gaps(state)


def _context() -> AnswerContext:
    state = GameState(level=7)
    understanding = understand_query("怎么运营", state, ProductMode.MATCH_GUIDANCE)
    return AnswerContext(
        question="怎么运营",
        understanding=understanding,
        game_state=state,
        history=[AgentMessage(role="user", content="我在玩测试阵容")],
        retrieval_plan=build_retrieval_plan(understanding, state, "18.1d"),
        knowledge=RetrievedKnowledge(),
        context_gaps=context_gaps(state),
    )


@pytest.mark.parametrize(
    "content",
    [
        json.dumps({"summary": "保留经济", "options": [{"action": "先不搜"}]}),
        '```json\n{"recommendation":"保留经济"}\n```',
        '说明：{"answer":"保留经济"} 完毕',
        "保留经济并观察下一轮来牌。",
    ],
)
def test_local_normalization_accepts_useful_output(content: str) -> None:
    assert normalize_decision_output(content, _context()).response.recommendation


def test_local_normalization_rejects_empty_output() -> None:
    with pytest.raises(ValueError):
        normalize_decision_output("", _context())


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **_: Any) -> Any:
        self.calls += 1
        message = SimpleNamespace(content='{"summary":"条件式建议","options":[]}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeOpenAI:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


@pytest.mark.asyncio
async def test_decision_client_makes_exactly_one_call() -> None:
    provider = _FakeOpenAI()
    client = GroundedDecisionClient(Settings(), client=provider)  # type: ignore[arg-type]
    result = await client.decide_context(_context())
    assert result.response.recommendation == "条件式建议"
    assert provider.completions.calls == 1


@pytest.mark.asyncio
async def test_sqlite_session_messages_are_isolated(tmp_path: Path) -> None:
    repository = SQLiteGameDataRepository(tmp_path / "state.sqlite3")
    await repository.initialize()
    await repository.append_session_messages("one", [("user", "hello")])
    await repository.append_session_messages("two", [("user", "keep")])
    await repository.clear_session("one")
    assert await repository.load_session_messages("one") == []
    assert await repository.load_session_messages("two") == [("user", "keep")]
