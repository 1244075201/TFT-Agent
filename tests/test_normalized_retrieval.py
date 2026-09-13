from __future__ import annotations

from typing import Any

import pytest

from tft_agent.app.entity_resolution import ResolvedEntity
from tft_agent.app.pipeline import ProductMode, build_retrieval_plan, understand_query
from tft_agent.app.service import AgentService
from tft_agent.core.config import Settings
from tft_agent.core.schemas import GameState
from tft_agent.tft_mcp_server.schemas import CompStatsRecord, DataSnapshotMetadata


def _entity(identifier: str, kind: str, name: str) -> ResolvedEntity:
    return ResolvedEntity(
        canonical_id=identifier, entity_type=kind, display_name=name, game_patch="18.2",
        set_number=18, set_name="Set 18", matched_alias=name, match_kind="exact",
        match_score=1.0,
    )


def _plan(question: str, entities: list[ResolvedEntity]) -> Any:
    state = GameState(user_question=question)
    understanding = understand_query(
        question, state, ProductMode.GENERAL_GUIDE,
        type("Resolution", (), {"entities": entities, "unresolved_terms": []})(),
    )
    return build_retrieval_plan(understanding, state, "18.2", 18, "Set 18", "snapshot-18-2")


class _NormalizedRepository:
    async def get_comp_board_units(
        self, snapshot_id: str, comp_id: str, phase: str
    ) -> list[dict[str, Any]]:
        return [
            {"display_name": "德莱文", "canonical_champion_id": "DA_Draven18",
             "x": 3, "y": 3, "stars": 2, "resolution_status": "EXACT",
             "source_response_id": "detail"},
            {"display_name": "布隆", "canonical_champion_id": "DA_18_Braum",
             "x": 2, "y": 0, "stars": 2, "resolution_status": "EXACT",
             "source_response_id": "detail"},
        ]

    async def get_comp_items(
        self, snapshot_id: str, comp_id: str
    ) -> list[dict[str, Any]]:
        return await self.get_comp_items_by_category(snapshot_id, comp_id, "NORMAL")

    async def get_comp_items_by_category(
        self, snapshot_id: str, comp_id: str, category: str
    ) -> list[dict[str, Any]]:
        if category == "RADIANT":
            return []
        return [{
            "holder_display_name": "德莱文",
            "holder_canonical_champion_id": "DA_Draven18",
            "item_display_name": "智慧末刃" if category == "ARTIFACT" else "无尽之刃",
            "canonical_item_id": "DA_Artifact_WitsEnd",
            "item_category": category, "plan_index": 1, "slot_index": 1,
            "identity_resolution_status": "EXACT", "source_response_id": "detail",
            "definition": {"description": "静态效果"},
        }]

    async def get_comp_augments(
        self, snapshot_id: str, comp_id: str
    ) -> list[dict[str, Any]]:
        return [{
            "display_name": "坚守防线", "canonical_augment_id": "DA_Augment",
            "source_kind": "TEAM_URL", "identity_resolution_status": "EXACT",
            "source_response_id": "detail", "definition": {"description": "获得防御收益"},
        }]

    async def get_comp_conditions(
        self, snapshot_id: str, comp_id: str
    ) -> list[dict[str, Any]]:
        return []

    async def get_comp_strategy_sections(
        self, snapshot_id: str, comp_id: str
    ) -> list[dict[str, Any]]:
        return [{"section_type": section, "ordinal": 0, "text": section,
                 "source_response_id": "detail"} for section in (
                     "starting_conditions", "core", "early_game", "mid_game", "late_game",
                 )]


def _comp() -> CompStatsRecord:
    metadata = DataSnapshotMetadata(
        snapshot_id="snapshot-18-2", server="cn", game_patch="18.2",
        source_url="https://example.test", retrieved_at="2026-09-13T00:00:00Z",
        content_hash="a" * 64,
    )
    return CompStatsRecord(comp_id="16241", name="德莱文95", snapshot=metadata)


@pytest.mark.parametrize(
    ("question", "need"),
    [
        ("德莱文95有哪些棋子？", "composition_roster"),
        ("德莱文怎么站？", "positioning"),
        ("德莱文有什么神器？", "artifact"),
        ("德莱文95有什么光明装备？", "radiant"),
        ("重装狮子狗有哪些强化？", "augment"),
    ],
)
def test_planner_requests_only_relevant_normalized_kind(question: str, need: str) -> None:
    plan = _plan(question, [_entity("16241", "composition", "德莱文95")])
    assert need in plan.knowledge_needs
    unrelated = {"composition_roster", "positioning", "artifact", "radiant", "augment"} - {need}
    assert not unrelated & set(plan.knowledge_needs)


@pytest.mark.asyncio
async def test_champion_position_and_artifact_are_holder_filtered() -> None:
    service = AgentService(Settings())
    service._repository = _NormalizedRepository()  # type: ignore[assignment]
    champion = _entity("DA_Draven18", "champion", "德莱文")
    for question, section in (("德莱文怎么站？", "positioning"),
                              ("德莱文有什么神器？", "artifact")):
        plan = _plan(question, [_entity("16241", "composition", "德莱文95"), champion])
        evidence, calls = await service._normalized_comp_evidence(_comp(), plan)
        assert [item.section for item in evidence] == [section]
        assert evidence[0].snapshot_id == "snapshot-18-2"
        assert evidence[0].source_response_id == "detail"
        assert calls[0].filters["snapshot_id"] == "snapshot-18-2"
        assert "布隆" not in str(evidence[0].semantic_content)


@pytest.mark.asyncio
async def test_radiant_gap_and_augment_no_ranking_semantics() -> None:
    service = AgentService(Settings())
    service._repository = _NormalizedRepository()  # type: ignore[assignment]
    comp = _entity("16241", "composition", "德莱文95")
    radiant, _ = await service._normalized_comp_evidence(
        _comp(), _plan("德莱文95有什么光明装备？", [comp])
    )
    assert radiant == []
    augments, _ = await service._normalized_comp_evidence(
        _comp(), _plan("德莱文95有哪些强化？", [comp])
    )
    content = augments[0].semantic_content
    assert isinstance(content, dict) and content["association_is_not_ranking"] is True
    row = content["augments"][0]
    assert row["rank"] is row["priority"] is row["score"] is None


@pytest.mark.asyncio
async def test_full_guide_uses_exact_normalized_sections() -> None:
    service = AgentService(Settings())
    service._repository = _NormalizedRepository()  # type: ignore[assignment]
    evidence, _ = await service._normalized_comp_evidence(
        _comp(), _plan("德莱文95怎么玩？", [_entity("16241", "composition", "德莱文95")])
    )
    sections = {item.section for item in evidence}
    assert {"starting_conditions", "core", "early_game", "mid_game", "late_game"} <= sections
    assert all(item.source_patch == "18.2" and item.patch_match == "exact" for item in evidence)
