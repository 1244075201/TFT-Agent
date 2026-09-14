from __future__ import annotations

import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

import tft_agent.app.service as service_module
from tft_agent.app.pipeline import AnswerContext, ProductMode, RetrievalTrace
from tft_agent.app.service import AgentService
from tft_agent.core.config import Settings
from tft_agent.core.schemas import ChatEvent, ChatRequest, FinalResponse
from tft_agent.llm.grounded_decision import DecisionOutcome


class _FakeDecisionClient:
    calls = 0

    def __init__(self, settings: Settings) -> None:
        pass

    async def decide_context(self, context: AnswerContext) -> DecisionOutcome:
        type(self).calls += 1
        return DecisionOutcome(
            response=FinalResponse(recommendation="基于当前重新检索的候选证据回答。"),
            options=[],
        )

    async def close(self) -> None:
        pass


@pytest.fixture
def continuity_settings(tmp_path: Path) -> Settings:
    source = Path(__file__).parents[1] / "data" / "tft.sqlite3"
    target = tmp_path / "continuity.sqlite3"
    shutil.copy2(source, target)
    return Settings(database_path=target, deepseek_api_key="fake-test-key")


async def _ask(
    service: AgentService,
    query: str,
    session: str,
    mode: ProductMode = ProductMode.GENERAL_GUIDE,
) -> list[ChatEvent]:
    events: list[ChatEvent] = []
    request = ChatRequest(text=query, session_id=session, product_mode=mode.value)
    stream: AsyncIterator[ChatEvent] = service.ask(request)
    async for event in stream:
        events.append(event)
    assert not [event for event in events if event.type == "error"]
    return events


def _trace(service: AgentService) -> RetrievalTrace:
    assert service.last_trace is not None
    return service.last_trace


@pytest.mark.asyncio
async def test_other_routes_reuses_previous_master_candidate_set(
    continuity_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(continuity_settings)
    await _ask(service, "开局主宰转怎么运营", "master-followup")
    first = _trace(service)
    first_ids = [str(item["comp_id"]) for item in first.structured_candidate_comps]
    assert len(first_ids) > 1

    events = await _ask(service, "那其他路线能怎么玩", "master-followup")
    second = _trace(service)
    expected = first_ids[1:]
    evidence_ids = {item.comp_id for item in second.retrieved_knowledge.evidence
                    if item.comp_id}

    assert second.strategy_state_inherited is True
    assert second.inherited_topic == "开局主宰转怎么运营"
    assert second.inherited_candidate_ids == expected
    assert set(expected) <= evidence_ids
    assert second.retrieved_knowledge.evidence
    assert second.llm_call_count == 1
    assert "PREVIOUS TOPIC" in second.clean_context_summary
    assert "CURRENT CANDIDATES" in second.clean_context_summary
    assert "no relevant evidence found" not in second.knowledge_gaps
    assert events[-1].message == "基于当前重新检索的候选证据回答。"


@pytest.mark.asyncio
async def test_ordinal_followup_retrieves_second_ashe_candidate(
    continuity_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(continuity_settings)
    await _ask(service, "艾希怎么玩", "ashe-ordinal")
    first = _trace(service)
    candidate_ids = [item.canonical_id for item in first.understanding.candidate_comps]
    assert len(candidate_ids) > 1

    await _ask(service, "那第二个呢", "ashe-ordinal")
    second = _trace(service)
    evidence_ids = {item.comp_id for item in second.retrieved_knowledge.evidence
                    if item.comp_id}
    assert second.strategy_state_inherited is True
    assert second.inherited_candidate_ids == [candidate_ids[1]]
    assert evidence_ids == {candidate_ids[1]}
    assert second.llm_call_count == 1


@pytest.mark.asyncio
async def test_comparison_followup_rereads_all_ashe_candidates(
    continuity_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(continuity_settings)
    await _ask(service, "艾希怎么玩", "ashe-comparison")
    first_ids = [item.canonical_id for item in _trace(service).understanding.candidate_comps]
    await _ask(service, "这几个有什么区别", "ashe-comparison")
    comparison = _trace(service)
    assert comparison.strategy_state_inherited is True
    assert comparison.inherited_candidate_ids == first_ids
    assert comparison.understanding.decision_intent.value == "composition_comparison"
    assert set(first_ids) <= {item.comp_id for item in comparison.retrieved_knowledge.evidence}


@pytest.mark.asyncio
async def test_explicit_new_topic_overrides_previous_candidates(
    continuity_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(continuity_settings)
    await _ask(service, "主宰转怎么玩", "topic-override")
    await _ask(service, "那森林转呢", "topic-override")
    trace = _trace(service)
    assert trace.strategy_state_inherited is False
    assert trace.inherited_candidate_ids == []
    assert any(item.get("expanded_trait") == "永恒之森"
               for item in trace.structured_expansions)


@pytest.mark.asyncio
async def test_strategy_state_isolated_and_cleared(
    continuity_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(continuity_settings)
    await _ask(service, "主宰转怎么玩", "isolation-a")

    await _ask(service, "那其他路线呢", "isolation-b")
    assert _trace(service).strategy_state_inherited is False

    await _ask(
        service, "那其他路线呢", "isolation-a", ProductMode.MATCH_GUIDANCE
    )
    assert _trace(service).strategy_state_inherited is False

    await service.clear_general_guide_conversation("isolation-a")
    await _ask(service, "那其他路线呢", "isolation-a")
    assert _trace(service).strategy_state_inherited is False


@pytest.mark.asyncio
async def test_patch_mismatch_invalidates_strategy_state(
    continuity_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(continuity_settings)
    await _ask(service, "主宰转怎么玩", "patch-isolation")
    key = "patch-isolation:general-guide"
    previous = service._strategy_turn_states[key]
    service._strategy_turn_states[key] = previous.model_copy(update={"patch": "18.1"})

    await _ask(service, "那其他路线呢", "patch-isolation")
    assert _trace(service).strategy_state_inherited is False
    assert key not in service._strategy_turn_states
