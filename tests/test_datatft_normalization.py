import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tft_agent.ingestion.datatft_normalization import (
    RuntimeCatalog,
    build_runtime_catalog,
    canonical_knowledge,
    decode_team_url,
)
from tft_agent.ingestion.sources.datatft import DataTFTSource
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)
from tft_agent.tft_mcp_server.schemas import CompStatsRecord, DataSnapshotMetadata


def _catalog() -> RuntimeCatalog:
    return build_runtime_catalog(
        {
            "heros18": [
                {"chessId": 918001, "key": "DA_18_One", "displayName": "一号", "code": "AA"},
                {"chessId": 918002, "key": "DA_18_Two", "displayName": "二号", "code": "AB"},
            ],
            "equips18": [
                {"equipId": 535, "key": "DA_InfinityEdge", "name": "无尽之刃", "code": "IE"},
                {"equipId": 539, "key": "DA_GuinsoosRageblade", "name": "鬼索", "code": "GR"},
            ],
            "hexs18": [[{"hexId": 100, "key": "DA_Hex", "name": "强化", "code": "HX"}]],
        },
        18,
    )


def test_official_team_url_semantics_decode_board_stars_items_and_hex() -> None:
    decoded = decode_team_url(
        "https://www.datatft.com/team-builder/AAA2IEGR*3HAB-x|HXs18", _catalog()
    )

    assert decoded["issues"] == []
    assert decoded["units"][0] == {
        "board_index": 0,
        "x": 0,
        "y": 0,
        "external_hero_id": "918001",
        "external_hero_key": "DA_18_One",
        "display_name": "一号",
        "resolution_status": "EXACT",
        "unresolved_raw_token": None,
        "stars": 3,
        "is_main": None,
        "replace_id": None,
        "replace_key": None,
        "extra_traits": [],
        "items": [
            {
                "external_item_id": "535",
                "external_item_key": "DA_InfinityEdge",
                "display_name": "无尽之刃",
                "resolution_status": "EXACT",
                "unresolved_raw_token": None,
            },
            {
                "external_item_id": "539",
                "external_item_key": "DA_GuinsoosRageblade",
                "display_name": "鬼索",
                "resolution_status": "EXACT",
                "unresolved_raw_token": None,
            },
        ],
        "powerups": {},
    }
    assert decoded["units"][1]["board_index"] == 7
    assert (decoded["units"][1]["x"], decoded["units"][1]["y"]) == (0, 1)
    assert decoded["units"][1]["is_main"] is True
    assert decoded["hexes"][0]["external_entity_id"] == "100"


def test_unknown_fixed_width_token_is_preserved_without_false_identity() -> None:
    decoded = decode_team_url("https://www.datatft.com/team-builder/AZZ", _catalog())
    assert decoded["units"][0]["external_hero_id"] is None
    assert decoded["units"][0]["unresolved_raw_token"] == "AZZ"
    assert decoded["issues"] == ["unknown hero token 'ZZ'"]


def test_official_set18_pseudo_unit_fallback_code_is_decoded() -> None:
    decoded = decode_team_url("https://www.datatft.com/team-builder/AzC", _catalog())
    assert decoded["issues"] == []
    assert decoded["units"][0]["external_hero_id"] == "999001"
    assert decoded["units"][0]["external_hero_key"] == "999001"


def test_early_and_final_boards_and_holder_without_name_are_normalized() -> None:
    detail = {
        "teamUrl": "https://www.datatft.com/team-builder/AAA",
        "earlyTeamUrl": "https://www.datatft.com/team-builder/HAB",
        "equips": [{"heroId": "918001", "equips": [["535", "539"]]}],
        "conditions": json.dumps(
            {
                "filterOptionResults": [
                    {"type": "hero", "id": "918001", "key": "DA_18_One", "name": "一号"},
                    {"type": "trait", "id": "DA_Trait", "traitLevelText": "3+"},
                    {"type": "equip", "id": "535"},
                ],
                "exclusionFilterOptionResults": [{"type": "hero", "id": "918002"}],
            }
        ),
        "open": "开局",
        "core": "核心",
        "earlyDesc": "前期",
        "midDesc": "中期",
        "finalDesc": "后期",
        "tips": [
            {
                "description": "提示",
                "author": "作者",
                "labels": [{"type": "hex", "id": "100", "key": "DA_Hex"}],
            }
        ],
    }
    knowledge = canonical_knowledge({"id": 1}, detail, _catalog())
    assert len(knowledge["boards"]["FINAL"]["units"]) == 1
    assert len(knowledge["boards"]["EARLY"]["units"]) == 1
    assert len(knowledge["item_plans"]) == 2
    assert knowledge["item_plans"][0]["holder"]["display_name"] == "一号"
    assert [item["condition_type"] for item in knowledge["conditions"]] == [
        "hero",
        "trait",
        "equip",
        "hero",
    ]
    assert [item["condition_kind"] for item in knowledge["conditions"]][-1] == "EXCLUSION"
    assert knowledge["hexes"][0]["source_kind"] == "TIP_LABEL"
    assert knowledge["hexes"][0]["rank"] is None
    assert [item["section_type"] for item in knowledge["strategy_sections"]] == [
        "starting_conditions",
        "core",
        "early_game",
        "mid_game",
        "late_game",
        "tip",
    ]


def test_v2_hash_changes_for_team_url_and_item_plan() -> None:
    source = DataTFTSource("https://example.test")

    def payload(team: str, item: str) -> str:
        knowledge = canonical_knowledge(
            {"id": "one"},
            {
                "title": "阵容",
                "teamUrl": team,
                "equips": [{"heroId": "918001", "equips": [[item]]}],
            },
            _catalog(),
        )
        return json.dumps(
            {
                "canonical_format_version": 2,
                "game_patch": "18.2",
                "comps": [{"id": "one", "name": "阵容", "knowledge_v2": knowledge}],
            },
            sort_keys=True,
        )

    base = source.parse(payload("https://www.datatft.com/team-builder/AAA", "535"))[0]
    board_changed = source.parse(payload("https://www.datatft.com/team-builder/HAA", "535"))[0]
    item_changed = source.parse(payload("https://www.datatft.com/team-builder/AAA", "539"))[0]
    assert len({base.content_hash, board_changed.content_hash, item_changed.content_hash}) == 3


@pytest.mark.asyncio
async def test_v2_storage_read_api_and_v1_compatibility(tmp_path: Path) -> None:
    database = tmp_path / "tft.sqlite3"
    repository = SQLiteGameDataRepository(database)
    await repository.initialize()
    now = datetime.now(UTC)
    await repository.create_sync_run("run", "18.2", "cn", 18, "Set 18", now)
    knowledge = canonical_knowledge(
        {"id": "one"},
        {
            "title": "阵容",
            "teamUrl": "https://www.datatft.com/team-builder/AAA",
            "equips": [{"heroId": "918001", "equips": [["535"]]}],
        },
        _catalog(),
    )
    payload = json.dumps(
        {
            "canonical_format_version": 2,
            "game_patch": "18.2",
            "comps": [{"id": "one", "name": "阵容", "knowledge_v2": knowledge}],
        }
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    metadata = DataSnapshotMetadata(
        snapshot_id="v2",
        server="cn",
        game_patch="18.2",
        source_url="https://example.test",
        retrieved_at=now,
        content_hash=digest,
    )
    await repository.store_candidate_snapshot(
        metadata,
        [CompStatsRecord(comp_id="one", name="阵容", snapshot=metadata)],
        payload,
        "run",
        {},
        "CANONICAL_INPUT_ONLY",
    )
    assert len(await repository.get_comp_board_units("v2", "one", "FINAL")) == 1
    assert len(await repository.get_comp_item_plans("v2", "one")) == 1
    assert await repository.get_comp_conditions("v2", "one") == []

    await repository.create_sync_run("legacy-run", "18.2", "cn", 18, "Set 18", now)
    legacy = metadata.model_copy(update={"snapshot_id": "v1"})
    assert await repository.store_candidate_snapshot(
        legacy,
        [CompStatsRecord(comp_id="one", name="阵容", snapshot=legacy)],
        json.dumps({"game_patch": "18.2", "comps": [{"id": "one", "name": "阵容"}]}),
        "legacy-run",
        {},
        "CANONICAL_INPUT_ONLY",
    ) == ("v1", True)

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT snapshot_id,canonical_format_version FROM datatft_snapshots "
            "ORDER BY canonical_format_version"
        ).fetchall() == [("v1", 1), ("v2", 2)]
        assert connection.execute("SELECT COUNT(*) FROM datatft_comp_conditions").fetchone() == (0,)
