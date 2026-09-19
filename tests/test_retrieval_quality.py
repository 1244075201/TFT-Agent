from __future__ import annotations

import re
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

import tft_agent.app.service as service_module
from tft_agent.app.entity_resolution import CanonicalEntityResolver
from tft_agent.app.pipeline import AnswerContext, ProductMode, RetrievalTrace, comp_evidence
from tft_agent.app.service import AgentService
from tft_agent.core.config import Settings
from tft_agent.core.schemas import ChatEvent, ChatRequest, FinalResponse, GameState
from tft_agent.llm.grounded_decision import DecisionOutcome
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)


class _FakeDecisionClient:
    calls = 0
    contexts: list[AnswerContext] = []

    def __init__(self, _: Settings) -> None:
        pass

    async def decide_context(self, context: AnswerContext) -> DecisionOutcome:
        type(self).calls += 1
        type(self).contexts.append(context)
        return DecisionOutcome(
            response=FinalResponse(
                recommendation="基于已检索证据给出条件式建议。",
                reasoning=["缺失信息已作为知识缺口保留。"],
                next_steps=["根据下一回合信息再调整。"],
            ),
            options=[],
        )

    async def close(self) -> None:
        pass


@pytest.fixture
def retrieval_settings(tmp_path: Path) -> Settings:
    source = Path(__file__).parents[1] / "data" / "tft.sqlite3"
    target = tmp_path / "retrieval.sqlite3"
    shutil.copy2(source, target)
    return Settings(database_path=target, deepseek_api_key="fake-test-key")


async def _ask(service: AgentService, query: str, session: str) -> list[ChatEvent]:
    events: list[ChatEvent] = []
    stream: AsyncIterator[ChatEvent] = service.ask(ChatRequest(text=query, session_id=session))
    async for event in stream:
        events.append(event)
    assert not [event for event in events if event.type == "error"]
    return events


async def _ask_general(service: AgentService, query: str, session: str) -> list[ChatEvent]:
    events: list[ChatEvent] = []
    request = ChatRequest(
        text=query, session_id=session, product_mode=ProductMode.GENERAL_GUIDE.value
    )
    async for event in service.ask(request):
        events.append(event)
    assert not [event for event in events if event.type == "error"]
    return events


def _trace(service: AgentService) -> RetrievalTrace:
    assert service.last_trace is not None
    return service.last_trace


@pytest.mark.asyncio
async def test_five_retrieval_acceptance_cases(
    retrieval_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    _FakeDecisionClient.calls = 0
    _FakeDecisionClient.contexts = []
    service = AgentService(retrieval_settings)

    await _ask(service, "高迅射厄斐怎么玩？", "case-one")
    first = _trace(service)
    assert first.understanding.target_comp is not None
    assert first.understanding.target_comp.display_name == "高迅射厄斐"
    assert first.retrieval_plan.retrieve_all_comps is False
    assert {item.display_name for item in first.retrieved_knowledge.evidence
            if item.entity_type == "composition"} <= {"高迅射厄斐"}
    assert first.llm_call_count == 1

    await _ask(service, "我玩高迅射，爆了一个迅射转，应该怎么调整？", "case-two")
    second = _trace(service)
    resolved = {(item.entity_type, item.display_name) for item in second.resolved_entities}
    assert ("composition", "高迅射厄斐") in resolved
    assert ("emblem", "迅捷射手纹章") in resolved
    assert second.retrieval_plan.retrieve_all_comps is False
    assert not any(call.operation == "browse_comps" for call in second.repository_calls)
    assert second.retrieval_first_strategy is True
    assert second.retrieved_knowledge.evidence
    assert second.llm_call_count == 1

    augment_names = await _three_unambiguous_augments(retrieval_settings)
    augment_query = (
        f"我2-1有{augment_names[0]}、{augment_names[1]}、{augment_names[2]}，"
        "选哪个？现在还没定阵容。"
    )
    await _ask(service, augment_query, "case-three")
    third = _trace(service)
    resolved_augments = [item for item in third.resolved_entities
                         if item.entity_type == "augment"]
    assert {item.display_name for item in resolved_augments} == set(augment_names)
    assert third.understanding.target_comp is None
    assert third.retrieval_plan.retrieve_all_comps is False
    broad_call = next(call for call in third.repository_calls
                      if call.operation == "hybrid_normalized_retrieval")
    assert broad_call.result_count > 0
    assert not any(call.operation == "browse_comps" for call in third.repository_calls)
    assert third.llm_call_count == 1

    events = await _ask(service, "德莱文95核心棋子是谁？", "case-four")
    fourth = _trace(service)
    answer = next(event for event in events if event.type == "answer")
    assert fourth.understanding.target_comp is not None
    assert fourth.understanding.target_comp.display_name == "德莱文95"
    assert fourth.understanding.kind.value == "deterministic_lookup"
    assert fourth.llm_call_count == 0
    assert answer.response is not None and "德莱文" in answer.response.recommendation
    assert {item.display_name for item in fourth.retrieved_knowledge.evidence
            if item.entity_type == "composition"} == {"德莱文95"}

    await _ask(
        service,
        "我2-1还没定阵容，现在有青龙刀、羊刀、狂徒，应该怎么过渡？",
        "case-five",
    )
    fifth = _trace(service)
    item_names = {item.display_name for item in fifth.resolved_entities
                  if item.entity_type == "item"}
    assert item_names == {"朔极之矛", "鬼索的狂暴之刃", "狂徒铠甲"}
    assert fifth.understanding.target_comp is None
    assert fifth.retrieval_plan.retrieve_all_comps is False
    assert not any(call.operation == "browse_comps" for call in fifth.repository_calls)
    assert fifth.retrieved_knowledge.evidence
    assert fifth.llm_call_count == 1
    assert _FakeDecisionClient.calls == 4

    for context in _FakeDecisionClient.contexts:
        summary = context.readable_summary()
        assert "BronzeForLife_2_" not in summary
        assert "DA_Riftbeast18" not in summary
        assert "DA_18_" not in summary
        assert not re.search(r"@[A-Za-z0-9_.*:]+@", summary)

    for trace in (first, second, third, fourth, fifth):
        assert trace.original_query
        assert trace.repository_calls
        assert trace.clean_context_summary
        assert {"entity_resolution", "planning", "retrieval", "context", "local_total"} <= (
            trace.stage_latency_ms.keys()
        )


async def _three_unambiguous_augments(settings: Settings) -> list[str]:
    repository = SQLiteGameDataRepository(settings.database_path.resolve())
    await repository.initialize()
    resolver = CanonicalEntityResolver(repository, settings)
    await resolver.resolve("初始化当前版本实体目录")
    catalog = await repository.canonical_catalog(
        settings.current_game_patch, settings.current_set_number
    )
    augment_rows = [row for row in catalog if row["entity_type"] == "augment"]
    name_counts: dict[str, int] = {}
    for row in augment_rows:
        name = str(row["display_name"])
        name_counts[name] = name_counts.get(name, 0) + 1
    records = await repository.find_static_entities_by_ids(
        [str(row["canonical_id"]) for row in augment_rows]
    )
    definitions = {record.entity_id: record.description for record in records}
    names = [
        str(row["display_name"])
        for row in augment_rows
        if name_counts[str(row["display_name"])] == 1
        and re.search(r"[\u4e00-\u9fff]", str(row["display_name"]))
        and definitions.get(str(row["canonical_id"]), "").strip()
    ]
    assert len(names) >= 3
    return names[:3]


@pytest.mark.asyncio
async def test_resolver_refuses_same_type_ambiguity(
    retrieval_settings: Settings
) -> None:
    repository = SQLiteGameDataRepository(retrieval_settings.database_path.resolve())
    await repository.initialize()
    resolver = CanonicalEntityResolver(repository, retrieval_settings)
    await resolver.resolve("初始化当前版本实体目录")
    catalog = await repository.canonical_catalog(
        retrieval_settings.current_game_patch, retrieval_settings.current_set_number
    )
    duplicate = next(
        name for name in {str(row["display_name"]) for row in catalog}
        if sum(row["entity_type"] == "augment" and row["display_name"] == name
               for row in catalog) > 1
    )
    result = await resolver.resolve(f"{duplicate}这个强化怎么样？")
    assert not [item for item in result.entities
                if item.entity_type == "augment" and item.display_name == duplicate]


@pytest.mark.asyncio
async def test_composition_canonical_name_outranks_cross_type_alias_generically() -> None:
    class CollisionCatalog:
        async def canonical_catalog(
            self, game_patch: str, set_number: int
        ) -> list[dict[str, object]]:
            base: dict[str, object] = {
                "game_patch": game_patch,
                "set_number": set_number,
                "set_name": "Test Set",
                "source": "test",
            }
            return [
                {**base, "canonical_id": "comp-night", "entity_type": "composition",
                 "display_name": "夜行者", "aliases": []},
                {**base, "canonical_id": "champ-shadow", "entity_type": "champion",
                 "display_name": "暗影英雄", "aliases": ["夜行者"]},
            ]

    resolver = CanonicalEntityResolver(  # type: ignore[arg-type]
        CollisionCatalog(),
        Settings(current_game_patch="18.2", current_set_number=18),
    )
    result = await resolver.resolve("夜行者有哪些棋子")
    assert [(item.entity_type, item.display_name) for item in result.entities] == [
        ("composition", "夜行者"),
    ]


@pytest.mark.asyncio
async def test_strategy_text_uses_recorded_patch_family_fallback(
    retrieval_settings: Settings
) -> None:
    repository = SQLiteGameDataRepository(retrieval_settings.database_path.resolve())
    await repository.initialize()
    chunks = await repository.search_knowledge_text(
        "Vanguard Aphelios", retrieval_settings.current_game_patch, 3
    )
    assert chunks
    patch_family = re.sub(r"[a-z]+$", "", retrieval_settings.current_game_patch)
    assert {chunk["game_patch"] for chunk in chunks} == {patch_family}
    assert {chunk["patch_match"] for chunk in chunks} == {"family_fallback"}


@pytest.mark.asyncio
async def test_meta_lookup_is_exact_patch_and_source_identity_is_not_url(
    retrieval_settings: Settings
) -> None:
    repository = SQLiteGameDataRepository(retrieval_settings.database_path.resolve())
    await repository.initialize()
    current = await repository.search_comps(
        None, retrieval_settings.current_game_patch, 2
    )
    assert len(current) == 2
    patch_family = re.sub(r"[a-z]+$", "", retrieval_settings.current_game_patch)
    family_records = await repository.search_comps(None, patch_family, 2)
    assert all(record.snapshot.game_patch == patch_family for record in family_records)
    assert all(record.snapshot.game_patch == retrieval_settings.current_game_patch
               for record in current)
    records = [comp_evidence(record, retrieval_settings.current_game_patch) for record in current]
    assert current[0].snapshot.source_url == current[1].snapshot.source_url
    identities = {item.evidence_id for group in records for item in group}
    assert len(identities) == sum(len(group) for group in records)


@pytest.mark.asyncio
async def test_state_aware_retrieval_regression_cases(
    retrieval_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    _FakeDecisionClient.calls = 0
    _FakeDecisionClient.contexts = []
    service = AgentService(retrieval_settings)

    await _ask(service, "天使怎么玩", "state-case-one")
    first = _trace(service)
    assert first.understanding.target_comp is not None
    assert first.understanding.target_comp.display_name == "森林天使"
    assert _composition_names(first) == {"森林天使"}
    assert "USER QUESTION\n天使怎么玩" in first.clean_context_summary

    await service.save_game_state(
        "state-case-two",
        GameState(
            comp_name="高迅射厄斐", active_comp_id="16271",
            emblems=["迅捷射手纹章"],
        ),
    )
    await service._repo().append_session_messages(
        "state-case-two", [("user", "黑荆棘裁决怎么玩？"), ("assistant", "旧回答")]
    )
    await _ask(service, "我玩蛇女阵容 但是D了半天就两张怎么办", "state-case-two")
    second = _trace(service)
    assert second.game_state_after_merge.comp_name == "6护卫蛇女"
    assert second.game_state_after_merge.emblems == ["迅捷射手纹章"]
    assert second.understanding.state_mutation.value == "replace"
    assert "6护卫蛇女" in _composition_names(second)
    assert second.structured_candidate_comps
    assert "6护卫蛇女" in second.clean_context_summary
    assert "黑荆棘裁决" not in second.clean_context_summary

    await _ask(
        service,
        "我现在艾希95和德莱文95的牌都来了一些，我应该怎么判断最后玩哪套？",
        "state-case-three",
    )
    third = _trace(service)
    assert [item.display_name for item in third.understanding.candidate_comps] == [
        "艾希95", "德莱文95",
    ]
    assert third.understanding.target_comp is None
    assert third.game_state_after_merge.comp_name is None
    assert any(call.operation == "hybrid_normalized_retrieval"
               for call in third.repository_calls)
    assert {"艾希95", "德莱文95"} <= _composition_names(third)

    await _ask(
        service,
        "我现在玩艾希95，4-1八级，35块，50血，要不要D牌，还是硬扛上9。",
        "state-case-four",
    )
    fourth = _trace(service)
    state = fourth.game_state_after_merge
    assert (state.comp_name, state.stage, state.level, state.gold, state.hp) == (
        "艾希95", "4-1", 8, 35, 50,
    )
    assert "early_game" not in fourth.retrieval_plan.knowledge_needs
    assert "starting_conditions" not in fourth.retrieval_plan.knowledge_needs
    assert not {"early_game", "starting_conditions"} & {
        item.section for item in fourth.retrieved_knowledge.evidence
    }
    assert "level_8_stabilization" in fourth.retrieval_plan.knowledge_needs
    assert fourth.original_query in fourth.clean_context_summary

    await _ask(service, "森林天使怎么玩", "state-case-five")
    fifth = _trace(service)
    assert "early_game" in fifth.retrieval_plan.knowledge_needs
    early = [str(item.semantic_content) for item in fifth.retrieved_knowledge.evidence
             if item.section == "early_game"]
    assert early and all("连败" in value for value in early)
    assert all("保血" not in value for value in early)
    assert "RETRIEVED EVIDENCE" in fifth.clean_context_summary
    assert "KNOWN GAPS" in fifth.clean_context_summary
    facts_block = fifth.clean_context_summary.split(
        "CURRENT GAMESTATE\n", 1
    )[1].split("\n\n", 1)[0]
    assert "体系卡数量多" not in facts_block

    await _ask(service, "我玩艾希95", "state-case-six")
    await _ask(service, "我还没定阵容", "state-case-six")
    sixth = _trace(service)
    assert sixth.understanding.state_mutation.value == "clear"
    assert sixth.game_state_after_merge.comp_name is None
    assert not [call for call in sixth.repository_calls if call.operation == "find_comp_by_id"]
    assert "艾希95" not in sixth.clean_context_summary

    await service.save_game_state(
        "state-case-seven",
        GameState(
            comp_name="德莱文95", active_comp_id="16241",
            emblems=["迅捷射手纹章"], item_components=["暴风大剑"],
        ),
    )
    await _ask(service, "我不玩德莱文了，换艾希95", "state-case-seven")
    seventh = _trace(service)
    assert seventh.understanding.state_mutation.value == "replace"
    assert seventh.game_state_after_merge.comp_name == "艾希95"
    assert seventh.understanding.candidate_comps == []
    assert seventh.game_state_after_merge.emblems == ["迅捷射手纹章"]
    assert seventh.game_state_after_merge.item_components == ["暴风大剑"]
    assert _composition_names(seventh) == {"艾希95"}

    await _ask(service, "我玩艾希95", "state-retain")
    await _ask(service, "那现在要不要D？", "state-retain")
    retained = _trace(service)
    assert retained.understanding.state_mutation.value == "retain"
    assert retained.understanding.target_comp is not None
    assert retained.understanding.target_comp.display_name == "艾希95"
    assert retained.injected_history == []
    assert "艾希95" in retained.retrieval_queries
    assert retained.game_state_after_merge.comp_name == "艾希95"
    assert _FakeDecisionClient.calls == 10


def _composition_names(trace: RetrievalTrace) -> set[str]:
    return {item.display_name for item in trace.retrieved_knowledge.evidence
            if item.entity_type == "composition"}


@pytest.mark.asyncio
async def test_general_guide_conversational_subject_precedence(
    retrieval_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(retrieval_settings)

    await _ask_general(service, "德莱文95怎么玩", "guide-followup-items")
    await _ask_general(service, "神器开局选什么比较好", "guide-followup-items")
    inherited = _trace(service)
    assert inherited.subject_inherited is True
    assert inherited.conversational_subject is not None
    assert inherited.conversational_subject.display_name == "德莱文95"
    assert inherited.understanding.target_comp is not None
    assert inherited.understanding.target_comp.display_name == "德莱文95"
    assert "德莱文95" in _composition_names(inherited)

    await _ask_general(service, "德莱文95怎么玩", "guide-followup-equipment")
    await _ask_general(service, "那要什么装备？", "guide-followup-equipment")
    equipment = _trace(service)
    assert equipment.subject_inherited is True
    assert equipment.understanding.target_comp is not None
    assert equipment.understanding.target_comp.display_name == "德莱文95"

    await _ask_general(service, "德莱文95怎么玩", "guide-global-browse")
    await _ask_general(service, "当前版本有哪些S级阵容？", "guide-global-browse")
    browse = _trace(service)
    assert browse.subject_inherited is False
    assert browse.understanding.target_comp is None
    assert browse.retrieval_plan.retrieve_all_comps is True

    await _ask_general(service, "德莱文95怎么玩", "guide-explicit-switch")
    await _ask_general(service, "那艾希95呢？", "guide-explicit-switch")
    switched = _trace(service)
    assert switched.subject_inherited is False
    assert switched.explicit_entities
    assert switched.understanding.target_comp is not None
    assert switched.understanding.target_comp.display_name == "艾希95"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_query", "second_query", "attempt", "forbidden"),
    [
        ("德莱文95怎么玩", "XXXXX怎么玩", "XXXXX", ["德莱文95"]),
    ],
)
async def test_explicit_unresolved_entity_blocks_subject_inheritance(
    retrieval_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    first_query: str,
    second_query: str,
    attempt: str,
    forbidden: list[str],
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(retrieval_settings)
    await _ask_general(service, first_query, "explicit-unresolved")
    events = await _ask_general(service, second_query, "explicit-unresolved")
    trace = _trace(service)

    assert trace.explicit_entity_attempt == attempt
    assert trace.explicit_unresolved_term == attempt
    assert trace.subject_inherited is False
    assert trace.inheritance_blocked_reason == "explicit_unresolved_entity_attempt"
    assert trace.conversational_subject is None
    assert trace.final_resolution_category.value == "resolved_strategy"
    assert trace.retrieval_first_strategy is True
    assert trace.llm_call_count == 1
    answer = next(event.message for event in events if event.type == "answer")
    assert all(value not in answer for value in forbidden)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_query", "forbidden"),
    [
        ("螳螂怎么玩", ["卡兹克"]),
        ("狮子狗怎么玩", ["雷恩加尔", "重装狮子狗"]),
    ],
)
async def test_explicit_trait_blocks_unrelated_subject_inheritance(
    retrieval_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    first_query: str,
    forbidden: list[str],
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(retrieval_settings)
    await _ask_general(service, first_query, "explicit-trait")
    events = await _ask_general(service, "地狱火怎么玩", "explicit-trait")
    trace = _trace(service)

    assert any(item.entity_type == "trait" and item.display_name == "地狱火"
               for item in trace.resolved_entities)
    assert trace.explicit_unresolved_term is None
    assert trace.subject_inherited is False
    assert trace.inheritance_blocked_reason == "current_explicit_resolved_entity"
    assert trace.conversational_subject is None
    answer = next(event.message for event in events if event.type == "answer")
    assert all(value not in answer for value in forbidden)


@pytest.mark.asyncio
async def test_explicit_new_entity_replaces_previous_subject_even_without_coverage(
    retrieval_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(retrieval_settings)
    await _ask_general(service, "螳螂怎么玩", "explicit-replacement")
    await _ask_general(service, "我想玩婕拉怎么运营", "explicit-replacement")
    trace = _trace(service)

    champion = next(item for item in trace.explicit_entities
                    if item.entity_type == "champion")
    assert champion.display_name == "婕拉"
    assert trace.subject_inherited is False
    assert trace.inheritance_blocked_reason == "current_explicit_resolved_entity"
    assert trace.conversational_subject is not None
    assert trace.conversational_subject.display_name == "婕拉"
    assert trace.final_resolution_category.value != "unresolved_entity"


@pytest.mark.asyncio
async def test_new_general_guide_conversation_clears_only_guide_scope(
    retrieval_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(retrieval_settings)
    await service._repo().clear_session("desktop-match")
    match_state = GameState(comp_name="森林天使", stage="4-1", hp=40)
    await service.save_game_state("desktop-match", match_state)
    await service._repo().append_session_messages(
        "desktop-match", [("user", "本局怎么运营"), ("assistant", "保留这条对局记录")]
    )

    await _ask_general(service, "德莱文95怎么玩", "desktop-guide")
    await _ask_general(service, "神器怎么选", "desktop-guide")
    before = _trace(service)
    assert before.subject_inherited is True
    assert before.understanding.target_comp is not None
    assert before.understanding.target_comp.display_name == "德莱文95"

    await service.clear_general_guide_conversation("desktop-guide")
    assert await service._repo().load_session_messages(
        "desktop-guide:general-guide"
    ) == []
    events = await _ask_general(service, "神器怎么选", "desktop-guide")
    after = _trace(service)
    assert after.conversation_reset is True
    assert after.conversational_subject is None
    assert after.subject_inherited is False
    assert after.injected_history == []
    assert after.understanding.target_comp is None
    assert all("德莱文95" not in event.message for event in events)

    assert await service.load_game_state("desktop-match") == match_state
    assert await service._repo().load_session_messages("desktop-match") == [
        ("user", "本局怎么运营"), ("assistant", "保留这条对局记录")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "entity_type", "canonical_id", "target", "candidates", "category"),
    [
        ("狮子狗怎么玩", "champion", "DA_18_Rengar", None,
         {"世纪和解", "重装狮子狗"}, "resolved_strategy"),
        ("螳螂怎么玩", "champion", "DA_18_KhaZix", "世纪和解", set(),
         "resolved_strategy"),
        ("剑圣怎么玩", "composition", "16253", "剑圣", set(), "resolved_strategy"),
        ("易怎么玩", "champion", "DA_18_MasterYi_AD", "剑圣", set(),
         "resolved_strategy"),
        ("索拉卡怎么玩", "champion", "DA_18_Soraka", None,
         {"6护卫蛇女", "永森莲华ez", "赌芸阿娜", "高森林EZ", "黑荆棘裁决"},
         "resolved_strategy"),
        ("墨菲特怎么玩", "champion", "DA_18_Malphite", None,
         {"猎手豹女", "黑荆棘裁决", "黑荆棘赌狼人"}, "resolved_strategy"),
        ("德莱文怎么玩", "champion", "DA_Draven18", "德莱文95", set(),
         "resolved_strategy"),
        ("德莱文95怎么玩", "composition", "16241", "德莱文95", set(),
         "resolved_strategy"),
    ],
)
async def test_champion_centric_strategy_routes(
    retrieval_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    entity_type: str,
    canonical_id: str,
    target: str | None,
    candidates: set[str],
    category: str,
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(retrieval_settings)
    events = await _ask(service, query, "champion-route")
    trace = _trace(service)

    assert any(item.entity_type == entity_type and item.canonical_id == canonical_id
               for item in trace.resolved_entities)
    assert trace.understanding.unresolved_terms == []
    assert trace.final_resolution_category.value == category
    assert (trace.understanding.target_comp.display_name
            if trace.understanding.target_comp else None) == target
    assert {item.display_name for item in trace.understanding.candidate_comps} == candidates
    if entity_type == "champion":
        assert any(call.operation == "find_compositions_by_champion_signals"
                   for call in trace.repository_calls)
    answer = next(event for event in events if event.type == "answer")
    if category == "resolved_entity_no_knowledge":
        assert answer.response is not None
        assert "已识别到英雄" in answer.response.recommendation
        assert "无法唯一识别" not in answer.response.recommendation


@pytest.mark.asyncio
async def test_cross_type_short_name_ambiguity_is_unresolved(
    retrieval_settings: Settings
) -> None:
    repository = SQLiteGameDataRepository(retrieval_settings.database_path.resolve())
    await repository.initialize()
    resolver = CanonicalEntityResolver(repository, retrieval_settings)
    result = await resolver.resolve("世纪和解怎么玩")
    assert not [item for item in result.entities if item.display_name == "世纪和解"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    ["剑圣核心棋子是谁？", "剑圣怎么玩", "剑圣有哪些棋子"],
)
async def test_composition_alias_collision_routes_to_exact_composition(
    retrieval_settings: Settings, monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(retrieval_settings)
    events = await _ask_general(service, query, f"composition-collision-{query}")
    trace = _trace(service)

    assert trace.understanding.target_comp is not None
    assert trace.understanding.target_comp.display_name == "剑圣"
    assert {(item.entity_type, item.display_name) for item in trace.resolved_entities} == {
        ("composition", "剑圣"),
    }
    assert trace.subject_inherited is False
    assert any(
        item.entity_type == "composition" and item.display_name == "剑圣"
        for item in trace.retrieved_knowledge.evidence
    )
    assert not [event for event in events if event.type == "error"]


@pytest.mark.asyncio
async def test_new_composition_attempt_never_inherits_previous_subject(
    retrieval_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "GroundedDecisionClient", _FakeDecisionClient)
    service = AgentService(retrieval_settings)
    await _ask_general(service, "拉露恩95核心棋子是谁？", "collision-continuity")
    events = await _ask_general(service, "剑圣核心棋子是谁？", "collision-continuity")
    trace = _trace(service)

    assert trace.explicit_entity_attempt == "剑圣"
    assert trace.subject_inherited is False
    assert trace.inheritance_blocked_reason == "current_explicit_resolved_entity"
    assert trace.conversational_subject is not None
    assert trace.conversational_subject.display_name == "剑圣"
    assert trace.understanding.target_comp is not None
    assert trace.understanding.target_comp.display_name == "剑圣"
    assert all(
        item.display_name != "拉露恩95" for item in trace.retrieved_knowledge.evidence
    )
    answer = next(event.message for event in events if event.type == "answer")
    assert "拉露恩95" not in answer


@pytest.mark.asyncio
async def test_unresolved_fact_subject_attempt_blocks_previous_composition(
    retrieval_settings: Settings
) -> None:
    service = AgentService(retrieval_settings)
    await _ask_general(service, "拉露恩95核心棋子是谁？", "unresolved-fact-subject")
    events = await _ask_general(
        service, "不存在阵容核心棋子是谁？", "unresolved-fact-subject"
    )
    trace = _trace(service)

    assert trace.explicit_entity_attempt == "不存在阵容"
    assert trace.explicit_unresolved_term == "不存在阵容"
    assert trace.subject_inherited is False
    assert trace.inheritance_blocked_reason == "explicit_unresolved_entity_attempt"
    assert trace.conversational_subject is None
    assert trace.understanding.target_comp is None
    answer = next(event.message for event in events if event.type == "answer")
    assert "拉露恩95" not in answer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "composition",
    ["艾希95", "主宰女警", "德莱文95", "拉露恩95", "希维尔拼多多"],
)
async def test_existing_composition_fact_resolution_regression(
    retrieval_settings: Settings, composition: str
) -> None:
    service = AgentService(retrieval_settings)
    await _ask_general(service, f"{composition}核心棋子是谁？", f"fact-{composition}")
    trace = _trace(service)
    assert trace.understanding.target_comp is not None
    assert trace.understanding.target_comp.display_name == composition
    assert trace.llm_call_count == 0
