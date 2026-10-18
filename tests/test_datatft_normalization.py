import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tft_agent.ingestion.datatft_normalization import (
    ItemCategory,
    RuntimeCatalog,
    build_runtime_catalog,
    canonical_knowledge,
    classify_item,
    decode_team_url,
)
from tft_agent.ingestion.sources.datatft import DataTFTSource
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)
from tft_agent.tft_mcp_server.schemas import (
    CompStatsRecord,
    DataSnapshotMetadata,
    StaticEntityRecord,
)


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
                {
                    "equipId": 30017,
                    "key": "DA_Artifact_WitsEnd",
                    "name": "智慧末刃",
                    "code": "WI",
                    "type2": "aoen",
                },
                {
                    "equipId": 1519,
                    "key": "DA_DeathbladeRadiant",
                    "name": "光明死亡之刃",
                    "code": "RD",
                    "type2": "guangming",
                },
                {
                    "equipId": 33032,
                    "key": "DA_18_EmblemVanguard",
                    "name": "重装战士纹章",
                    "code": "EM",
                    "type2": "zhuanzhi",
                },
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
        "runtime_description": None,
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
                "runtime_description": None,
                "runtime_type": None,
                "item_category": "OTHER",
            },
            {
                "external_item_id": "539",
                "external_item_key": "DA_GuinsoosRageblade",
                "display_name": "鬼索",
                "resolution_status": "EXACT",
                "unresolved_raw_token": None,
                "runtime_description": None,
                "runtime_type": None,
                "item_category": "OTHER",
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


def test_item_classification_uses_upstream_metadata_not_display_names() -> None:
    assert classify_item("DA_Normal", "common") is ItemCategory.NORMAL
    assert classify_item("DA_Artifact_Example") is ItemCategory.ARTIFACT
    assert classify_item("DA_ExampleRadiant") is ItemCategory.RADIANT
    assert classify_item("DA_18_EmblemExample") is ItemCategory.EMBLEM
    assert classify_item("DA_Unknown") is ItemCategory.OTHER
    assert classify_item("DA_Unknown", static_tags=["{44ace175}"]) is ItemCategory.ARTIFACT
    assert classify_item("DA_Unknown", static_tags=["{ec243f6b}"]) is ItemCategory.RADIANT


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
            "teamUrl": "https://www.datatft.com/team-builder/AAA|HX",
            "equips": [
                {
                    "heroId": "918001",
                    "equips": [["535", "30017", "1519", "33032"]],
                }
            ],
            "tips": [
                {
                    "description": "强化提示",
                    "labels": [{"type": "hex", "id": "100", "key": "DA_Hex"}],
                }
            ],
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
    await repository.replace_static_entities(
        [
            StaticEntityRecord(
                entity_id=key,
                entity_type=entity_type,
                name=name,
                description=f"{name}效果",
                tags=tags,
                data_version="test",
                source="communitydragon",
                source_url="https://example.test/static",
                retrieved_at=now,
            )
            for key, entity_type, name, tags in (
                ("DA_InfinityEdge", "item", "无尽之刃", []),
                ("DA_Artifact_WitsEnd", "item", "智慧末刃", ["{44ace175}"]),
                ("DA_DeathbladeRadiant", "item", "光明死亡之刃", ["{ec243f6b}"]),
                ("DA_18_EmblemVanguard", "emblem", "重装战士纹章", []),
                ("DA_Hex", "augment", "强化", []),
            )
        ]
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
    assert len(await repository.get_comp_item_plans("v2", "one")) == 4
    assert await repository.get_comp_conditions("v2", "one") == []
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE datatft_comp_metadata SET representative_display_name='代表英雄' "
            "WHERE snapshot_id='v2' AND comp_id='one'"
        )
    assert await repository.get_comp_presentation_core_units("v2", "one") == ["代表英雄"]
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE datatft_comps SET core_units='[\"装备主核\", \"前排主核\"]' "
            "WHERE snapshot_id='v2' AND comp_id='one'"
        )
    assert await repository.get_comp_presentation_core_units("v2", "one") == [
        "装备主核", "前排主核",
    ]
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE datatft_comp_board_units SET is_main=1,display_name='明确主核' "
            "WHERE snapshot_id='v2' AND comp_id='one'"
        )
    assert await repository.get_comp_presentation_core_units("v2", "one") == ["明确主核"]
    await repository.resolve_datatft_normalized_entities("v2", "18.2", 18)
    assert len(await repository.get_comp_items("v2", "one")) == 4
    artifact = await repository.get_comp_items_by_category("v2", "one", "ARTIFACT")
    radiant = await repository.get_comp_items_by_category("v2", "one", "RADIANT")
    emblem = await repository.get_comp_items_by_category("v2", "one", "EMBLEM")
    assert [
        (row["holder_external_hero_id"], row["plan_index"], row["slot_index"]) for row in artifact
    ] == [("918001", 1, 2)]
    assert artifact[0]["definition"]["description"] == "智慧末刃效果"
    assert radiant[0]["definition"]["description"] == "光明死亡之刃效果"
    assert emblem[0]["definition"]["description"] == "重装战士纹章效果"
    augments = await repository.get_comp_augments("v2", "one")
    assert {row["source_kind"] for row in augments} == {"TEAM_URL", "TIP_LABEL"}
    assert all(row["rank"] is None and row["priority"] is None for row in augments)
    assert augments[0]["definition"]["description"] == "强化效果"
    item_definition = await repository.get_item_definition("v2", "30017")
    augment_definition = await repository.get_augment_definition("v2", "DA_Hex")
    assert item_definition is not None and item_definition["item_category"] == "ARTIFACT"
    assert augment_definition is not None and augment_definition["display_name"] == "强化"

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


def test_new_replacement_encoding_preserves_the_next_board_position() -> None:
    decoded = decode_team_url(
        "https://www.datatft.com/simulator/AAA_ABHABs18", _catalog()
    )
    assert decoded["issues"] == []
    assert [unit["board_index"] for unit in decoded["units"]] == [0, 7]
    assert decoded["units"][0]["replace_id"] == "918002"
    assert decoded["units"][1]["external_hero_id"] == "918002"
