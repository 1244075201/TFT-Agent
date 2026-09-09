from __future__ import annotations

import re
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

import tft_agent.app.service as service_module
from tft_agent.app.entity_resolution import CanonicalEntityResolver
from tft_agent.app.pipeline import AnswerContext, RetrievalTrace, comp_evidence
from tft_agent.app.service import AgentService
from tft_agent.core.config import Settings
from tft_agent.core.schemas import ChatEvent, ChatRequest, FinalResponse
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
    assert any("missing" in gap for gap in second.knowledge_gaps)
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
    static_call = next(call for call in third.repository_calls
                       if call.operation == "find_static_entities_by_ids")
    assert set(static_call.filters["canonical_ids"]) == {
        item.canonical_id for item in resolved_augments
    }
    assert static_call.result_count == 3
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
        "我2-1还没定阵容，现在有大剑、反曲弓、腰带，应该怎么过渡？",
        "case-five",
    )
    fifth = _trace(service)
    item_names = {item.display_name for item in fifth.resolved_entities
                  if item.entity_type == "item"}
    assert item_names == {"暴风大剑", "反曲之弓", "巨人腰带"}
    assert fifth.understanding.target_comp is None
    assert fifth.retrieval_plan.retrieve_all_comps is False
    assert not any(call.operation == "browse_comps" for call in fifth.repository_calls)
    assert any("candidate index" in gap for gap in fifth.knowledge_gaps)
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
