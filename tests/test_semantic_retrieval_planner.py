from __future__ import annotations

import shutil
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

import tft_agent.app.service as service_module
from tft_agent.app.pipeline import AnswerContext, ProductMode
from tft_agent.app.semantic_retrieval import (
    CompositionFilters,
    CostQuantifier,
    PlannerOutcome,
    RetrievalCapability,
    SemanticRetrievalPlan,
    is_semantic_discovery_query,
    validate_plan_payload,
)
from tft_agent.app.service import AgentService
from tft_agent.core.config import Settings
from tft_agent.core.schemas import ChatEvent, ChatRequest, FinalResponse
from tft_agent.llm.grounded_decision import DecisionOutcome, enforce_closed_world_grounding
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)


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
async def test_explicit_strategy_uses_planner_but_factual_browse_does_not(
    settings: Settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "RetrievalPlannerClient", _Planner)
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _Answerer)
    _Planner.calls = _Answerer.calls = 0
    service = AgentService(settings)
    await _ask(service, "主宰转怎么运营", "control-strategy")
    assert service.last_trace is not None
    assert service.last_trace.planner_invoked is True
    assert service.last_trace.llm_call_count == 2
    await _ask(service, "推荐20套阵容", "control-fact")
    assert service.last_trace is not None
    assert service.last_trace.planner_invoked is False
    assert service.last_trace.llm_call_count == 0
    assert _Planner.calls == 1


@pytest.mark.asyncio
async def test_semantic_candidates_support_followup_with_refreshed_intent(
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
    assert second.planner_invoked is True
    assert second.llm_call_count == 2
    assert _Planner.calls == 2


@pytest.mark.asyncio
async def test_descriptive_preference_does_not_become_unresolved_entity(
    settings: Settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "RetrievalPlannerClient", _Planner)
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _Answerer)
    service = AgentService(settings)
    await _ask(service, "我想玩节奏快一点的阵容", "descriptive")
    trace = service.last_trace
    assert trace is not None
    assert trace.route == "semantic_discovery"
    assert trace.planner_invoked is True
    assert trace.explicit_unresolved_term is None
    assert trace.final_resolution_category.value != "unresolved_entity"


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
        {**base, "capabilities": ["FIND_BY_TRAIT"]},
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


@pytest.mark.parametrize("query", [
    "这版本两费主c的阵容有什么",
    "三费主C能玩什么",
    "我想玩节奏快一点的阵容",
    "我想玩稳一点的",
    "后期上限高一点的阵容",
    "有没有不太吃五费卡的",
    "比较容易成型的有什么",
    "我不想赌三星，有什么低费核心阵容",
    "有哪些阵容能吃烂分 哪些能吃鸡",
])
def test_attribute_discovery_does_not_require_preference_keyword_list(query: str) -> None:
    assert is_semantic_discovery_query(query, set()) is True


def test_declared_capability_requires_executable_parameters() -> None:
    base = {"user_goal": "discover", "requested_fields": ["tier"], "limit": 8}
    with pytest.raises(ValueError, match="required parameters"):
        validate_plan_payload({
            **base,
            "capabilities": ["FILTER_BY_MAIN_UNIT_COST"],
        })
    with pytest.raises(ValueError, match="required parameters"):
        validate_plan_payload({
            **base,
            "capabilities": ["SEARCH_STRATEGY_TEXT"],
        })
    normalized = validate_plan_payload({
        **base,
        "capabilities": ["LIST_COMPOSITIONS"],
        "strategy_exclusions": ["excluded concept"],
    })
    assert RetrievalCapability.SEARCH_STRATEGY_TEXT in normalized.capabilities


def test_generic_cost_plans_share_capability_and_vary_only_range() -> None:
    ranges = ((1, 1), (2, 2), (3, 3), (1, 2))
    plans = [validate_plan_payload({
        "user_goal": "cost constrained discovery",
        "capabilities": ["LIST_COMPOSITIONS", "FILTER_BY_MAIN_UNIT_COST"],
        "composition_filters": {
            "main_unit_cost_min": low,
            "main_unit_cost_max": high,
            "cost_quantifier": "ANY",
            "require_strategy_match": False,
        },
        "requested_fields": ["tier", "main_units"],
        "limit": 8,
    }) for low, high in ranges]
    assert {
        tuple(plan.capabilities) for plan in plans
    } == {(
        RetrievalCapability.LIST_COMPOSITIONS,
        RetrievalCapability.FILTER_BY_MAIN_UNIT_COST,
    )}
    assert [(plan.composition_filters.main_unit_cost_min,
             plan.composition_filters.main_unit_cost_max) for plan in plans] == list(ranges)


@pytest.mark.asyncio
async def test_main_carry_cost_quantifier_any_is_not_all(
    settings: Settings,
) -> None:
    with sqlite3.connect(settings.database_path) as connection:
        snapshot = connection.execute(
            "SELECT active_snapshot_id FROM game_patches WHERE status='active'"
        ).fetchone()[0]
        comp_id = connection.execute(
            "SELECT comp_id FROM datatft_comps WHERE snapshot_id=? AND name=?",
            (snapshot, "主宰女警"),
        ).fetchone()[0]
        connection.execute(
            "UPDATE datatft_comp_board_units SET is_main=1 WHERE snapshot_id=? "
            "AND comp_id=? AND display_name IN (?,?)",
            (snapshot, comp_id, "凯特琳", "希维尔"),
        )
    repository = SQLiteGameDataRepository(settings.database_path)
    common = dict(
        snapshot_id=snapshot, game_patch="18.2",
        main_unit_cost_min=2, main_unit_cost_max=2,
        strategy_terms=[], item_terms=[], require_strategy_match=False,
        strategy_exclusions=[],
        requested_fields=["tier", "main_units"], limit=12,
    )
    any_result = await repository.discover_compositions(
        **common, cost_quantifier=CostQuantifier.ANY.value,
    )
    all_result = await repository.discover_compositions(
        **common, cost_quantifier=CostQuantifier.ALL.value,
    )
    any_rows = cast(list[dict[str, object]], any_result["candidate_comps"])
    all_rows = cast(list[dict[str, object]], all_result["candidate_comps"])
    any_names = {row["name"] for row in any_rows}
    all_names = {row["name"] for row in all_rows}
    assert "主宰女警" in any_names
    assert "主宰女警" not in all_names


def test_closed_world_rejects_hallucinated_entity_and_statistics() -> None:
    from tft_agent.app.pipeline import QueryStateUnderstanding, RetrievalPlan

    context = AnswerContext(
        question="有哪些阵容比较稳定",
        understanding=QueryStateUnderstanding(
            kind="strategy_decision", product_mode="general_guide",
        ),
        game_state={},
        retrieval_plan=RetrievalPlan(
            game_patch="18.2", set_number=18, set_name="test",
        ),
        evidence=[],
        retrieval_first=True,
    )
    unsafe = DecisionOutcome(FinalResponse(
        recommendation="纳亚菲利前四率63.7%，平均排名3.89。",
        reasoning=["这套阵容胜率很高。"],
    ), [])
    result = enforce_closed_world_grounding(unsafe, context, [{
        "display_name": "纳亚菲利", "aliases": [], "entity_type": "champion",
    }])
    assert result.used_fallback is True
    assert any("canonical entity" in value for value in result.violations)
    assert any("statistic" in value for value in result.violations)
    assert any("胜率很高" in value for value in result.violations)
    assert "纳亚菲利" not in result.outcome.response.recommendation
    assert "63.7" not in result.outcome.response.recommendation


def test_closed_world_allows_only_statistics_present_in_evidence() -> None:
    from tft_agent.app.pipeline import EvidenceRecord, QueryStateUnderstanding, RetrievalPlan

    evidence = EvidenceRecord(
        evidence_id="stats", source="test", requested_patch="18.2", source_patch="18.2",
        patch_match="exact", entity_type="composition", canonical_entity_id="comp-1",
        display_name="测试阵容", section="composition",
        semantic_content={"top4_rate": "63.7%", "sample_size": 10000,
                          "sampling_window": "2026-09-20/2026-09-21"},
        evidence_kind="statistical_result", relevance_score=1, url="test",
    )
    context = AnswerContext(
        question="表现如何",
        understanding=QueryStateUnderstanding(
            kind="strategy_decision", product_mode="general_guide",
        ), game_state={}, evidence=[evidence],
        retrieval_plan=RetrievalPlan(game_patch="18.2", set_number=18, set_name="test"),
    )
    safe = DecisionOutcome(FinalResponse(recommendation="测试阵容前四率63.7%。"), [])
    result = enforce_closed_world_grounding(safe, context, [{
        "display_name": "测试阵容", "aliases": [], "entity_type": "composition",
    }])
    assert result.used_fallback is False
    evidence.semantic_content.pop("sample_size")
    assert enforce_closed_world_grounding(safe, context, []).used_fallback
