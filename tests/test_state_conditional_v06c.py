from __future__ import annotations

import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

import tft_agent.app.service as service_module
from tft_agent.app.pipeline import (
    AnswerContext,
    ProductMode,
    QueryKind,
    QueryStateUnderstanding,
    RetrievalPlan,
)
from tft_agent.app.semantic_retrieval import (
    DiscoveryQueryType,
    PlannerOutcome,
    RetrievalCapability,
    SemanticRetrievalPlan,
    is_state_aware_strategy_query,
    validate_plan_payload,
)
from tft_agent.app.service import AgentService
from tft_agent.core.config import Settings
from tft_agent.core.schemas import ChatEvent, ChatRequest, FinalResponse, GameState
from tft_agent.llm.grounded_decision import DecisionOutcome, normalize_decision_output


class _StatePlanner:
    calls = 0
    concepts = ["经济", "上9", "血量"]

    def __init__(self, settings: Settings) -> None:
        pass

    async def plan(self, query: str, game_state: object = None,
                   *, state_aware: bool = False) -> PlannerOutcome:
        type(self).calls += 1
        assert state_aware
        return PlannerOutcome(
            model="fake-state-planner", latency_ms=1,
            plan=SemanticRetrievalPlan(
                user_goal="find conditional economic and leveling strategy",
                query_type=DiscoveryQueryType.STATE_AWARE_STRATEGY,
                observed_state=["low resources"],
                decision_questions=["stabilize or level"],
                capabilities=[RetrievalCapability.SEARCH_STRATEGY_TEXT,
                              RetrievalCapability.GET_COMPOSITION_STRATEGY],
                strategy_searches=type(self).concepts,
                requested_fields=["mid_game", "late_game"],
            ),
        )

    async def close(self) -> None:
        pass


class _Answerer:
    contexts: list[AnswerContext] = []
    recommendation = "当前攻略证据只能支持有条件地调整运营。"

    def __init__(self, settings: Settings) -> None:
        pass

    async def decide_context(self, context: AnswerContext) -> DecisionOutcome:
        type(self).contexts.append(context)
        return DecisionOutcome(FinalResponse(recommendation=type(self).recommendation), [])

    async def close(self) -> None:
        pass


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    source = Path(__file__).parents[1] / "data" / "tft.sqlite3"
    target = tmp_path / "state-conditional.sqlite3"
    shutil.copy2(source, target)
    return Settings(database_path=target, deepseek_api_key="fake")


async def _ask(service: AgentService, query: str, session: str,
               mode: ProductMode = ProductMode.GENERAL_GUIDE) -> list[ChatEvent]:
    events: list[ChatEvent] = []
    stream: AsyncIterator[ChatEvent] = service.ask(ChatRequest(
        text=query, session_id=session, product_mode=mode.value,
    ))
    async for event in stream:
        events.append(event)
    assert not [event for event in events if event.type == "error"]
    return events


@pytest.mark.parametrize("query", [
    "我如果大连败开 没血 经济又没办法支撑我直接上9怎么办",
    "血量很低但质量不够怎么办",
    "8级没打过对面还要不要存钱",
    "经济很好但血量很低怎么处理",
    "连败开怎么止血",
])
def test_state_aware_structure_without_state_keyword_rules(query: str) -> None:
    assert is_state_aware_strategy_query(query, set())


def test_state_negation_is_search_concept_not_silent_filter() -> None:
    plan = validate_plan_payload({
        "user_goal": "conditional leveling decision",
        "query_type": "state_aware_strategy",
        "observed_state": ["resources insufficient"],
        "decision_questions": ["level or stabilize"],
        "capabilities": ["SEARCH_STRATEGY_TEXT", "GET_COMPOSITION_STRATEGY"],
        "strategy_searches": ["stabilize"],
        "strategy_exclusions": ["level nine"],
        "requested_fields": ["mid_game"],
    })
    assert plan.strategy_searches == ["stabilize", "level nine"]
    assert plan.strategy_exclusions == []


def test_malformed_json_extracts_only_recommendation_not_raw_blob() -> None:
    context = AnswerContext(
        question="怎么调整",
        understanding=QueryStateUnderstanding(
            kind=QueryKind.STRATEGY_DECISION, product_mode=ProductMode.GENERAL_GUIDE,
        ),
        game_state=GameState(),
        retrieval_plan=RetrievalPlan(game_patch="18.2", set_number=18, set_name="test"),
    )
    outcome = normalize_decision_output(
        '{"recommendation":"先核对当前证据。","next_steps":["再决定",]}', context,
    )
    assert outcome.response.recommendation == "先核对当前证据。"


@pytest.mark.asyncio
async def test_state_aware_retrieves_active_strategy_evidence(
    settings: Settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "RetrievalPlannerClient", _StatePlanner)
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _Answerer)
    _StatePlanner.concepts = ["经济", "上9", "血量"]
    _StatePlanner.calls = 0
    _Answerer.contexts = []
    service = AgentService(settings)
    await _ask(service, "我如果大连败开 没血 经济又没办法支撑我直接上9怎么办", "state")
    trace = service.last_trace
    assert trace is not None
    assert trace.route == "state_aware_strategy"
    assert trace.query_type == "state_aware_strategy"
    assert trace.planner_invoked and trace.llm_call_count == 2
    assert trace.retrieved_knowledge.evidence
    assert all(item.snapshot_id == trace.retrieval_plan.active_snapshot_id
               for item in trace.retrieved_knowledge.evidence)
    assert any(call.operation == "search_state_strategy_text"
               for call in trace.repository_calls)
    assert "OBSERVED STATE" in _Answerer.contexts[-1].readable_summary()


@pytest.mark.asyncio
async def test_state_aware_no_match_preserves_knowledge_gap(
    settings: Settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "RetrievalPlannerClient", _StatePlanner)
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _Answerer)
    _StatePlanner.concepts = ["zzqnonexistent" ]
    service = AgentService(settings)
    events = await _ask(service, "血量很低但质量不够怎么办", "empty")
    trace = service.last_trace
    assert trace is not None and not trace.retrieved_knowledge.evidence
    assert any("does not prove" in gap for gap in trace.knowledge_gaps)
    assert trace.grounding_fallback_used
    assert "没有足够检索证据" in events[-1].message


@pytest.mark.asyncio
@pytest.mark.parametrize("first,second,target,condition", [
    ("厄斐琉斯没有迅射转能玩吗", "那有和没有都怎么构筑呢", "厄斐琉斯", "迅捷射手纹章"),
    ("德莱文没有鱼骨头能玩吗", "那有和没有分别怎么玩", "德莱文", "鱼骨头"),
])
async def test_conditional_followup_keeps_target_and_separate_scenarios(
    settings: Settings, monkeypatch: pytest.MonkeyPatch,
    first: str, second: str, target: str, condition: str,
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _Answerer)
    _Answerer.recommendation = f"没有{condition}也能玩。"
    service = AgentService(settings)
    await _ask(service, first, "conditional")
    initial = service.last_trace
    assert initial is not None and initial.route == "conditional_strategy"
    assert target in initial.target_entities
    assert condition in initial.focal_conditions
    events = await _ask(service, second, "conditional")
    followup = service.last_trace
    assert followup is not None
    assert followup.route == "conditional_followup"
    assert followup.planner_invoked is True and followup.llm_call_count == 2
    assert target in followup.target_entities
    assert condition in followup.focal_conditions
    assert set(followup.scenario_evidence) == {"present", "absent"}
    assert followup.scenario_evidence["absent"] == []
    assert followup.grounding_fallback_used
    assert "不能确认同样可玩" in events[-1].message
    assert all(item.comp_id in set(followup.inherited_candidate_ids)
               for item in followup.retrieved_knowledge.evidence)


@pytest.mark.asyncio
async def test_association_only_does_not_rank_choices(
    settings: Settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _Answerer)
    _Answerer.recommendation = "这三个里优先升级咯。"
    service = AgentService(settings)
    events = await _ask(service, "升级咯 对冲基金 水乳交融 这三个开局选什么", "choices")
    trace = service.last_trace
    assert trace is not None
    assert trace.planner_invoked and trace.llm_call_count == 2
    assert trace.grounding_fallback_used
    assert "优先升级咯" not in events[-1].message


@pytest.mark.asyncio
async def test_descriptive_condition_followup_retains_target(
    settings: Settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _Answerer)
    _Answerer.recommendation = "只能按检索到的证据回答。"
    service = AgentService(settings)
    await _ask(service, "德莱文95有经济强化和没有怎么运营", "concept")
    first = service.last_trace
    assert first is not None and first.route == "conditional_strategy"
    assert first.focal_conditions == ["经济强化"]
    await _ask(service, "那两种情况都说一下", "concept")
    followup = service.last_trace
    assert followup is not None and followup.route == "conditional_followup"
    assert "德莱文95" in followup.target_entities
    assert followup.focal_conditions == ["经济强化"]
