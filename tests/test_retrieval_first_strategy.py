from __future__ import annotations

import shutil
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

import tft_agent.app.service as service_module
from tft_agent.app.pipeline import AnswerContext, ProductMode, RetrievalTrace
from tft_agent.app.service import AgentService
from tft_agent.core.config import Settings
from tft_agent.core.schemas import ChatEvent, ChatRequest, FinalResponse, GameState
from tft_agent.llm.grounded_decision import DecisionOutcome


class _FakeDecisionClient:
    contexts: list[AnswerContext] = []
    calls = 0

    def __init__(self, settings: Settings) -> None:
        pass

    async def decide_context(self, context: AnswerContext) -> DecisionOutcome:
        type(self).calls += 1
        type(self).contexts.append(context)
        return DecisionOutcome(
            response=FinalResponse(recommendation="按当前局面和检索证据继续运营。"),
            options=[],
        )

    async def close(self) -> None:
        pass


@pytest.fixture
def retrieval_first_settings(tmp_path: Path) -> Settings:
    source = Path(__file__).parents[1] / "data" / "tft.sqlite3"
    target = tmp_path / "retrieval-first.sqlite3"
    shutil.copy2(source, target)
    return Settings(database_path=target, deepseek_api_key="fake-test-key")


async def _ask(
    service: AgentService,
    text: str,
    session_id: str,
    mode: ProductMode,
    state: GameState | None = None,
) -> list[ChatEvent]:
    events: list[ChatEvent] = []
    request = ChatRequest(
        text=text,
        session_id=session_id,
        product_mode=mode.value,
        confirmed_game_state=state,
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
async def test_match_strategy_proceeds_with_zero_resolved_entities(
    retrieval_first_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    _FakeDecisionClient.calls = 0
    _FakeDecisionClient.contexts = []
    service = AgentService(retrieval_first_settings)
    state = GameState(
        stage="5-1", level=7, gold=30, hp=26,
        completed_items=["鬼索的狂暴之刃", "巨人杀手"],
        emblems=["约德尔人和朋友纹章"],
    )

    events = await _ask(
        service, "我现在该怎么玩", "zero-entity", ProductMode.MATCH_GUIDANCE, state
    )
    trace = _trace(service)

    assert trace.retrieval_first_strategy is True
    assert trace.resolved_entities == []
    assert trace.final_resolution_category.value == "resolved_strategy"
    assert trace.original_query == "我现在该怎么玩"
    assert trace.retrieval_queries[0] == "我现在该怎么玩"
    assert trace.game_state_present is True
    assert any(call.operation == "hybrid_normalized_retrieval"
               for call in trace.repository_calls)
    assert trace.retrieved_knowledge.evidence
    assert all(item.source_patch == "18.2" for item in trace.retrieved_knowledge.evidence)
    assert "CURRENT GAMESTATE" in trace.clean_context_summary
    assert "stage: 5-1" in trace.clean_context_summary
    assert trace.llm_call_count == 1 == _FakeDecisionClient.calls
    assert not any("ENTITY_UNRESOLVED" in event.message for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    ["现在怎么玩", "怎么办", "下一步干什么", "我现在该干嘛", "这把怎么运营"],
)
async def test_natural_match_questions_share_open_strategy_path(
    retrieval_first_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    query: str,
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(retrieval_first_settings)
    await _ask(service, query, "natural-" + query, ProductMode.MATCH_GUIDANCE, GameState(level=7))
    trace = _trace(service)
    assert trace.retrieval_first_strategy is True
    assert trace.original_query == query
    assert trace.llm_call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "重装狮子狗怎么玩", "艾希怎么玩", "花妖转怎么玩", "森林转怎么玩", "地狱火艾希怎么玩",
])
async def test_general_strategy_uses_broad_current_patch_retrieval(
    retrieval_first_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    query: str,
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(retrieval_first_settings)
    await _ask(service, query, "general-" + query, ProductMode.GENERAL_GUIDE)
    trace = _trace(service)
    assert trace.retrieval_first_strategy is True
    assert trace.retrieval_queries[0] == query
    assert trace.retrieved_knowledge.evidence
    assert all(item.source_patch == "18.2" for item in trace.retrieved_knowledge.evidence)
    if query == "森林转怎么玩":
        assert any(item.entity_type == "composition"
                   for item in trace.retrieved_knowledge.evidence)
    assert "CURRENT GAMESTATE" not in trace.clean_context_summary
    assert trace.llm_call_count == 1


@pytest.mark.asyncio
async def test_general_guide_does_not_receive_saved_match_state(
    retrieval_first_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(retrieval_first_settings)
    await service.save_game_state(
        "isolated-match", GameState(stage="5-1", gold=30, comp_name="重装狮子狗")
    )
    await _ask(service, "重装狮子狗怎么玩", "isolated-match", ProductMode.GENERAL_GUIDE)
    trace = _trace(service)
    assert trace.game_state_present is False
    assert "5-1" not in trace.clean_context_summary
    assert "30金币" not in trace.clean_context_summary


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expanded_trait"),
    [
        ("开局有森林转怎么构筑", "永恒之森"),
        ("开局有地狱火纹章怎么构筑", "地狱火"),
    ],
)
async def test_trait_and_emblem_reverse_retrieval_generate_candidates(
    retrieval_first_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    expanded_trait: str,
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(retrieval_first_settings)
    await _ask(service, query, "trait-reverse-" + expanded_trait, ProductMode.GENERAL_GUIDE)
    trace = _trace(service)
    assert any(item.get("expanded_trait") == expanded_trait
               for item in trace.structured_expansions)
    assert trace.structured_candidate_comps
    assert any(item.section == "candidate_composition"
               for item in trace.retrieved_knowledge.evidence)
    assert any(item.section in {"starting_conditions", "core", "early_game", "mid_game",
                                "late_game"}
               for item in trace.retrieved_knowledge.evidence)
    assert trace.llm_call_count == 1


@pytest.mark.asyncio
async def test_artifact_reverse_retrieval_includes_holders_and_compositions(
    retrieval_first_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(retrieval_first_settings)
    await _ask(
        service, "开局有鱼骨头怎么构筑", "artifact-reverse", ProductMode.GENERAL_GUIDE
    )
    trace = _trace(service)
    assert any(item.get("relation") == "item_to_holder_to_composition"
               for item in trace.structured_expansions)
    assert trace.structured_candidate_comps
    relations = [item for item in trace.retrieved_knowledge.evidence
                 if item.section == "structured_relation"]
    assert relations and any("鱼骨头" in str(item.semantic_content) for item in relations)
    assert any("holder" in str(item.get("signals"))
               for item in trace.structured_candidate_comps)


@pytest.mark.asyncio
async def test_augment_reverse_retrieval_generates_associated_compositions(
    retrieval_first_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    with sqlite3.connect(retrieval_first_settings.database_path) as connection:
        row = connection.execute(
            "SELECT display_name FROM datatft_comp_hexes "
            "WHERE snapshot_id=(SELECT active_snapshot_id FROM game_patches "
            "WHERE game_patch='18.2') AND canonical_augment_id IS NOT NULL "
            "GROUP BY canonical_augment_id,display_name HAVING COUNT(DISTINCT comp_id)>0 LIMIT 1"
        ).fetchone()
    assert row is not None
    augment_name = str(row[0])
    service = AgentService(retrieval_first_settings)
    await _ask(
        service, f"开局有{augment_name}可以玩什么", "augment-reverse",
        ProductMode.GENERAL_GUIDE,
    )
    trace = _trace(service)
    assert any(item.get("relation") == "augment_to_composition"
               for item in trace.structured_expansions)
    assert trace.structured_candidate_comps
    assert trace.llm_call_count == 1


@pytest.mark.asyncio
async def test_recommend_twenty_comps_uses_structured_list_without_strategy_noise(
    retrieval_first_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(retrieval_first_settings)
    events = await _ask(
        service, "现在这个版本你推荐20套阵容", "twenty-comps", ProductMode.GENERAL_GUIDE
    )
    trace = _trace(service)
    comps = [item for item in trace.retrieved_knowledge.evidence
             if item.section == "composition"]
    assert trace.retrieval_first_strategy is False
    assert len(comps) == 20
    assert {item.section for item in trace.retrieved_knowledge.evidence} == {"composition"}
    assert trace.llm_call_count == 0
    assert "按数据库 tier 优先列出的阵容" in events[-1].message


@pytest.mark.asyncio
async def test_ambiguous_fire_emblem_does_not_pull_unrelated_fallback_comps(
    retrieval_first_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(retrieval_first_settings)
    await _ask(
        service, "开局有火转怎么构筑", "ambiguous-fire", ProductMode.GENERAL_GUIDE
    )
    trace = _trace(service)
    assert trace.retrieval_first_strategy is True
    assert trace.structured_seeds == []
    assert trace.structured_candidate_comps == []
    assert trace.retrieved_knowledge.evidence == []
    assert trace.knowledge_gaps == [
        "no relevant evidence found in active normalized knowledge",
        "no unique structured entity seed was resolved; request the full entity name",
    ]
    assert trace.llm_call_count == 1
