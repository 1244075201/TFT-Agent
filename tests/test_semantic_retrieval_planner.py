from __future__ import annotations

import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import ValidationError

import tft_agent.app.service as service_module
from tft_agent.app.pipeline import AnswerContext, ProductMode
from tft_agent.app.semantic_retrieval import (
    CompositionFilters,
    PlannerOutcome,
    RetrievalCapability,
    SemanticRetrievalPlan,
    validate_plan_payload,
)
from tft_agent.app.service import AgentService
from tft_agent.core.config import Settings
from tft_agent.core.schemas import ChatEvent, ChatRequest, FinalResponse
from tft_agent.llm.grounded_decision import DecisionOutcome


class _Planner:
    calls = 0

    def __init__(self, settings: Settings) -> None:
        pass

    async def plan(self, query: str, game_state: object = None) -> PlannerOutcome:
        type(self).calls += 1
        reroll = "赌狗" in query
        return PlannerOutcome(
            model="fake-planner",
            latency_ms=2,
            plan=SemanticRetrievalPlan(
                user_goal="寻找低费主核且有追三星运营证据的阵容" if reroll else "寻找低费主核阵容",
                capabilities=[
                    RetrievalCapability.LIST_COMPOSITIONS,
                    RetrievalCapability.FILTER_BY_MAIN_UNIT_COST,
                    RetrievalCapability.SEARCH_STRATEGY_TEXT,
                    RetrievalCapability.GET_COMPOSITION_STRATEGY,
                ],
                composition_filters=CompositionFilters(
                    main_unit_cost_max=3, require_strategy_match=reroll
                ),
                strategy_searches=(
                    ["追三星", "三星", "慢D", "D牌", "6级D", "7级D"] if reroll else []
                ),
                requested_fields=[
                    "tier", "main_units", "starting_conditions", "core", "early_game",
                ],
                limit=8,
            ),
        )

    async def close(self) -> None:
        pass


class _Answerer:
    calls = 0
    contexts: list[AnswerContext] = []

    def __init__(self, settings: Settings) -> None:
        pass

    async def decide_context(self, context: AnswerContext) -> DecisionOutcome:
        type(self).calls += 1
        type(self).contexts.append(context)
        return DecisionOutcome(FinalResponse(recommendation="grounded"), [])

    async def close(self) -> None:
        pass


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    source = Path(__file__).parents[1] / "data" / "tft.sqlite3"
    target = tmp_path / "semantic.sqlite3"
    shutil.copy2(source, target)
    return Settings(database_path=target, deepseek_api_key="fake")


async def _ask(service: AgentService, query: str, session: str) -> list[ChatEvent]:
    events: list[ChatEvent] = []
    stream: AsyncIterator[ChatEvent] = service.ask(ChatRequest(
        text=query, session_id=session, product_mode=ProductMode.GENERAL_GUIDE.value,
    ))
    async for event in stream:
        events.append(event)
    assert not [event for event in events if event.type == "error"]
    return events


@pytest.mark.asyncio
async def test_semantic_query_invokes_planner_and_uses_active_snapshot(
    settings: Settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "RetrievalPlannerClient", _Planner)
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _Answerer)
    _Planner.calls = _Answerer.calls = 0
    _Answerer.contexts = []
    service = AgentService(settings)
    await _ask(service, "这版本低费赌狗能玩什么", "semantic")
    trace = service.last_trace
    assert trace is not None
    assert trace.route == "semantic_discovery"
    assert trace.planner_invoked is True
    assert trace.llm_call_count == 2
    assert _Planner.calls == _Answerer.calls == 1
    assert trace.retrieval_plan.game_patch == "18.2"
    assert trace.retrieved_knowledge.evidence
    assert all(row.source_patch == "18.2" for row in trace.retrieved_knowledge.evidence)
    assert all(row.snapshot_id == trace.retrieval_plan.active_snapshot_id
               for row in trace.retrieved_knowledge.evidence)
    summary = _Answerer.contexts[-1].readable_summary()
    assert "RETRIEVAL INTERPRETATION" in summary
    assert "SELECT " not in summary


@pytest.mark.asyncio
async def test_explicit_strategy_and_fact_do_not_invoke_planner(
    settings: Settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "RetrievalPlannerClient", _Planner)
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _Answerer)
    _Planner.calls = _Answerer.calls = 0
    service = AgentService(settings)
    await _ask(service, "主宰转怎么运营", "control-strategy")
    assert service.last_trace is not None
    assert service.last_trace.planner_invoked is False
    assert service.last_trace.llm_call_count == 1
    await _ask(service, "推荐20套阵容", "control-fact")
    assert service.last_trace is not None
    assert service.last_trace.planner_invoked is False
    assert service.last_trace.llm_call_count == 0
    assert _Planner.calls == 0


@pytest.mark.asyncio
async def test_semantic_candidates_support_followup_without_second_planner(
    settings: Settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "RetrievalPlannerClient", _Planner)
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _Answerer)
    _Planner.calls = 0
    service = AgentService(settings)
    await _ask(service, "这版本低费赌狗能玩什么", "continuity")
    first = service.last_trace
    assert first is not None and first.candidate_compositions
    await _ask(service, "那第二个怎么玩", "continuity")
    second = service.last_trace
    assert second is not None
    assert second.strategy_state_inherited is True
    assert second.planner_invoked is False
    assert second.llm_call_count == 1
    assert _Planner.calls == 1


def test_plan_rejects_sql_unknown_capability_and_patch_override() -> None:
    base = {
        "user_goal": "find comps",
        "capabilities": ["LIST_COMPOSITIONS"],
        "requested_fields": ["tier"],
        "limit": 8,
    }
    for illegal in (
        {**base, "sql": "SELECT * FROM datatft_comps"},
        {**base, "capabilities": ["RUN_SQL"]},
        {**base, "patch": "18.1"},
    ):
        with pytest.raises((ValidationError, ValueError)):
            validate_plan_payload(illegal)


def test_low_cost_reroll_plan_differs_from_low_cost_only() -> None:
    reroll = SemanticRetrievalPlan(
        user_goal="low cost reroll",
        capabilities=[RetrievalCapability.FILTER_BY_MAIN_UNIT_COST],
        composition_filters=CompositionFilters(
            main_unit_cost_max=3, require_strategy_match=True
        ),
        strategy_searches=["追三星", "慢D"],
    )
    low_cost = SemanticRetrievalPlan(
        user_goal="low cost",
        capabilities=[RetrievalCapability.FILTER_BY_MAIN_UNIT_COST],
        composition_filters=CompositionFilters(main_unit_cost_max=3),
    )
    assert reroll.composition_filters.require_strategy_match is True
    assert low_cost.composition_filters.require_strategy_match is False
    assert low_cost.strategy_searches == []
