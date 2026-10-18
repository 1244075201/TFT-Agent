"""End-to-end local pipeline tests; model semantics are separately provider-evaluated."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tft_agent.app.live_knowledge import LiveKnowledge, WebPage
from tft_agent.app.pipeline import AnswerContext
from tft_agent.app.semantic_retrieval import (
    PlannerOutcome,
    RetrievalPlannerClient,
    SemanticRetrievalPlan,
)
from tft_agent.app.service import AgentService
from tft_agent.core.config import Settings
from tft_agent.core.schemas import ChatRequest, FinalResponse, GameState
from tft_agent.llm.grounded_decision import DecisionOutcome


class Planner:
    inputs: list[str] = []

    def __init__(self, settings: Settings) -> None:
        pass

    async def plan(
        self, query: str, game_state: GameState | None = None, *, state_aware: bool = False
    ) -> PlannerOutcome:
        self.inputs.append(query)
        current = query.split("当前问题：")[-1]
        followup = "没这个海克斯" in current or "哪个更稳" in current
        names = [name for name in ("艾希", "地狱火", "魔女", "黑暗仪式") if name in query]
        return PlannerOutcome(
            plan=SemanticRetrievalPlan(
                user_goal="比较成型潜力" if "上限" in query else "寻找满足条件的攻略",
                preferences=["追三星", "稳定名次"] if "赌狗" in query else [],
                constraints=["没有黑暗仪式"] if "没这个海克斯" in current else [],
                mentioned_entities=names,
                relationships=["组合玩法"] if names else [],
                strategy_searches=names or ["后期", "运营", "追三星"],
                capabilities=["SEARCH_STRATEGY_TEXT", "GET_COMPOSITION_STRATEGY"],
                requested_fields=["core", "early_game", "late_game"],
                followup=followup,
                freshness_required=True,
                # The service must not promote inferred possession into user facts.
                observed_state=["玩家有黑暗仪式"],
            ),
            model="fixture-planner",
            latency_ms=0,
        )

    async def close(self) -> None:
        pass


class Answerer:
    contexts: list[AnswerContext] = []

    def __init__(self, settings: Settings) -> None:
        pass

    async def decide_context(self, context: AnswerContext) -> DecisionOutcome:
        self.contexts.append(context)
        return DecisionOutcome(
            FinalResponse(
                recommendation="根据已检索攻略，可以在满足条件时尝试；缺失条件时不能直接照搬。",
            ),
            [],
        )

    async def close(self) -> None:
        pass


class Source:
    queries: list[str]

    def __init__(self) -> None:
        self.queries = []

    def search(self, query: str) -> list[str]:
        self.queries.append(query)
        return ["https://datatft.com/emerging"]

    def fetch(self, url: str) -> WebPage:
        return WebPage(
            url,
            "18.2版本新玩法攻略",
            "18.2版本 地狱火艾希 魔女黑暗仪式。"
            + "作者建议先核实开局条件，过渡时观察经济和装备，后期视来牌调整。" * 6,
        )


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentService:
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "guide.sqlite3",
        current_game_patch="18.2",
        deepseek_api_key="fixture",
        knowledge_search_api_key=None,
    )
    service = AgentService(settings)
    service._repo()
    with sqlite3.connect(settings.database_path) as db:
        for kind, name in [
            ("champion", "艾希"),
            ("trait", "地狱火"),
            ("trait", "魔女"),
            ("augment", "黑暗仪式"),
        ]:
            db.execute(
                "INSERT INTO canonical_entities VALUES (?,?,?,?,?,?,?)",
                (name, kind, name, "18.2", 18, "test", "fixture"),
            )
    source = Source()
    service.live_knowledge = LiveKnowledge(settings, source, source)
    monkeypatch.setattr("tft_agent.app.service.RetrievalPlannerClient", Planner)
    monkeypatch.setattr("tft_agent.app.service.GroundedDecisionClient", Answerer)
    Planner.inputs = []
    Answerer.contexts = []
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "什么阵容上限高",
        "什么赌狗能吃烂分",
        "地狱火艾希怎么玩",
        "魔女专属海克斯黑暗仪式怎么玩",
    ],
)
async def test_strategy_goal_and_new_combination_reach_verified_web_evidence(
    service: AgentService,
    query: str,
) -> None:
    events = [
        event
        async for event in service.ask(
            ChatRequest(
                text=query,
                session_id="guide",
                product_mode="general_guide",
            )
        )
    ]
    assert not [event for event in events if event.type == "error"]
    assert events[-1].type == "answer"
    assert "无法唯一识别" not in events[-1].message
    assert "证据有限" in events[-1].message
    trace = service.last_trace
    assert trace and trace.planner_invoked and trace.llm_call_count == 2
    assert trace.understanding.unresolved_terms == []
    assert any(item.section == "web_strategy" for item in trace.retrieved_knowledge.evidence)
    assert not trace.observed_state
    assert not Answerer.contexts[-1].game_state.augments
    assert "publication_unknown" in trace.clean_context_summary
    assert "https://datatft.com/emerging" in trace.clean_context_summary


@pytest.mark.asyncio
async def test_negated_followup_preserves_subject_but_not_possession(service: AgentService) -> None:
    for query in ("魔女专属海克斯黑暗仪式怎么玩", "没这个海克斯呢"):
        events = [
            event
            async for event in service.ask(
                ChatRequest(
                    text=query,
                    session_id="guide",
                    product_mode="general_guide",
                )
            )
        ]
        assert events[-1].type == "answer"
    context = Answerer.contexts[-1]
    assert context.previous_topic and "黑暗仪式" in context.previous_topic
    assert context.retrieval_interpretation and "没有黑暗仪式" in context.retrieval_interpretation
    assert context.game_state.augments == []
    assert context.observed_state == []
    assert "当前问题：没这个海克斯呢" in Planner.inputs[-1]
    await service.clear_general_guide_conversation("guide")
    assert not service._guide_topics


@pytest.mark.asyncio
async def test_named_combo_without_evidence_is_not_replaced_with_another_build(
    service: AgentService,
) -> None:
    service.live_knowledge.search = None
    events = [
        event
        async for event in service.ask(
            ChatRequest(
                text="魔女专属海克斯黑暗仪式怎么玩",
                session_id="missing",
                product_mode="general_guide",
            )
        )
    ]
    assert events[-1].type == "answer"
    assert "尚未找到当前版本中这套组合的可靠运营攻略" in events[-1].message
    assert "这不代表玩法不存在" in events[-1].message
    assert service.last_trace and service.last_trace.grounding_fallback_used


@pytest.mark.asyncio
async def test_provider_guide_routing_and_redundant_field_normalization() -> None:
    class Completions:
        async def create(self, **kwargs: Any) -> Any:
            payload = {
                "user_goal": "魔女与黑暗仪式组合攻略",
                "query_type": "state_aware_strategy",
                "observed_state": [],
                "mentioned_entities": ["魔女", "黑暗仪式"],
                "relationships": [{"source": "魔女", "type": "组合", "target": "黑暗仪式"}],
                "capabilities": ["GET_COMPOSITION_ITEMS", "GET_COMPOSITION_ROSTER"],
                "requested_fields": ["core"],
            }
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(payload, ensure_ascii=False),
                        )
                    )
                ]
            )

    client: Any = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    planner = RetrievalPlannerClient(Settings(_env_file=None), client=client)
    result = await planner.plan("魔女专属海克斯黑暗仪式怎么玩")
    assert not result.used_fallback
    assert result.plan.query_type == "composition_discovery"
    assert result.plan.mentioned_entities == ["魔女", "黑暗仪式"]
    assert result.plan.relationships == ["魔女 组合 黑暗仪式"]
    assert {"items", "roster"} <= set(result.plan.requested_fields)
