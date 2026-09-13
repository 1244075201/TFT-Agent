from __future__ import annotations

import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import tft_agent.app.service as service_module
from tft_agent.app.pipeline import AnswerContext, ProductMode, RetrievalTrace
from tft_agent.app.service import AgentService
from tft_agent.core.config import Settings
from tft_agent.core.schemas import ChatEvent, ChatRequest, FinalResponse
from tft_agent.llm.grounded_decision import DecisionOutcome, GroundedDecisionClient


class _FakeDecisionClient:
    contexts: list[AnswerContext] = []

    def __init__(self, settings: Settings) -> None:
        pass

    async def decide_context(self, context: AnswerContext) -> DecisionOutcome:
        type(self).contexts.append(context)
        return DecisionOutcome(
            response=FinalResponse(recommendation="仅按当前证据给出回答。"), options=[]
        )

    async def close(self) -> None:
        pass


@pytest.fixture
def polish_settings(tmp_path: Path) -> Settings:
    source = Path(__file__).parents[1] / "data" / "tft.sqlite3"
    target = tmp_path / "query-polish.sqlite3"
    shutil.copy2(source, target)
    return Settings(database_path=target, deepseek_api_key="fake-test-key")


async def _ask(service: AgentService, query: str, session: str) -> list[ChatEvent]:
    events: list[ChatEvent] = []
    request = ChatRequest(
        text=query, session_id=session, product_mode=ProductMode.GENERAL_GUIDE.value
    )
    stream: AsyncIterator[ChatEvent] = service.ask(request)
    async for event in stream:
        events.append(event)
    assert not [event for event in events if event.type == "error"]
    return events


def _trace(service: AgentService) -> RetrievalTrace:
    assert service.last_trace is not None
    return service.last_trace


@pytest.mark.asyncio
async def test_inferno_emblem_and_ashe_are_resolved_as_separate_entities(
    polish_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(polish_settings)
    events = await _ask(service, "地狱火转艾希怎么玩", "multi-entity")
    trace = _trace(service)
    resolved = {(item.entity_type, item.display_name) for item in trace.resolved_entities}

    assert ("champion", "艾希") in resolved
    assert ("emblem", "地狱火纹章") in resolved
    assert trace.understanding.unresolved_terms == []
    assert trace.explicit_unresolved_term is None
    assert any("recognized entities lack exact relation" in gap for gap in trace.knowledge_gaps)
    answer = next(event.message for event in events if event.type == "answer")
    assert "已识别到实体" in answer and "证据没有建立" in answer
    assert "三条候选路线" not in answer
    assert trace.llm_call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "kind", "name"),
    [
        ("裁决转德莱文怎么玩", "emblem", "裁决使纹章"),
        ("神谕转阿狸能玩吗", "emblem", "神谕纹章"),
        ("雷恩加尔拿鱼骨头怎么玩", "item", "鱼骨头"),
        ("德莱文有金币收集者怎么玩", "item", "金币收集者"),
    ],
)
async def test_natural_multi_entity_phrases_preserve_each_known_entity(
    polish_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    kind: str,
    name: str,
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(polish_settings)
    await _ask(service, query, "natural-multi")
    trace = _trace(service)
    assert any(item.entity_type == kind and item.display_name == name
               for item in trace.resolved_entities)
    assert trace.understanding.unresolved_terms == []


@pytest.mark.asyncio
async def test_emblem_only_query_remains_resolved_without_fake_candidates(
    polish_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(polish_settings)
    events = await _ask(service, "地狱火转现在怎么玩好", "emblem-only")
    trace = _trace(service)
    assert any(item.entity_type == "emblem" and item.display_name == "地狱火纹章"
               for item in trace.resolved_entities)
    assert trace.understanding.unresolved_terms == []
    assert trace.understanding.target_comp is None
    assert trace.understanding.candidate_comps == []
    assert "三条候选路线" not in next(
        event.message for event in events if event.type == "answer"
    )


@pytest.mark.asyncio
async def test_followup_core_items_inherits_comp_and_preserves_real_plan_groups(
    polish_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(polish_settings)
    await _ask(service, "重装狮子狗怎么玩", "item-followup")
    events = await _ask(service, "那核心装备是什么", "item-followup")
    trace = _trace(service)
    answer = next(event.message for event in events if event.type == "answer")

    assert trace.subject_inherited is True
    assert trace.understanding.target_comp is not None
    assert trace.understanding.target_comp.display_name == "重装狮子狗"
    assert "雷恩加尔（方案1：" in answer
    assert "方案2：" in answer and "方案3：" in answer
    assert "方案编号仅表示上游分组，不代表优先级" in answer


@pytest.mark.asyncio
async def test_champion_items_and_artifacts_share_neutral_grouped_renderer(
    polish_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(polish_settings)
    for index, query in enumerate(("德莱文出什么装备", "德莱文有什么神器")):
        events = await _ask(service, query, f"grouped-{index}")
        trace = _trace(service)
        answer = next(event.message for event in events if event.type == "answer")
        assert trace.retrieval_plan.target_champion is not None
        assert answer.startswith("德莱文95")
        assert "德莱文（方案" in answer
        assert "不代表优先级" in answer
        assert "最优" not in answer and "第一推荐" not in answer


@pytest.mark.asyncio
async def test_explicit_new_subject_drops_stale_candidate_context(
    polish_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(polish_settings)
    await _ask(service, "艾希95和德莱文95选哪个", "stale-candidates")
    await _ask(service, "重装狮子狗怎么玩", "stale-candidates")
    trace = _trace(service)
    assert trace.subject_inherited is False
    assert trace.understanding.target_comp is not None
    assert trace.understanding.target_comp.display_name == "重装狮子狗"
    assert trace.understanding.candidate_comps == []
    assert "CANDIDATE COMPOSITIONS" not in trace.clean_context_summary
    assert "艾希95" not in trace.clean_context_summary
    assert "德莱文95" not in trace.clean_context_summary


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_request: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.last_request = kwargs
        message = SimpleNamespace(content='{"recommendation":"证据不足。"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.mark.asyncio
async def test_decision_prompt_closes_knowledge_gap_fact_boundary(
    polish_settings: Settings
) -> None:
    completions = _FakeCompletions()
    provider = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = GroundedDecisionClient(polish_settings, client=provider)  # type: ignore[arg-type]
    service = AgentService(polish_settings)
    await service._context()
    state = service_module.GameState(user_question="地狱火转现在怎么玩好")
    understanding = service_module.understand_query(
        state.user_question or "", state, ProductMode.GENERAL_GUIDE
    )
    plan = service_module.build_retrieval_plan(
        understanding, state, "18.2", 18, "Enchanted Wilds", "snapshot"
    )
    context = AnswerContext(
        question=state.user_question or "", understanding=understanding,
        game_state=state, retrieval_plan=plan,
        knowledge_gaps=["recognized entities lack exact relation"],
    )
    await client.decide_context(context)
    prompt = completions.last_request["messages"][0]["content"]
    assert "不得发明候选路线或候选数量" in prompt
    assert "必须直接陈述缺口" in prompt
