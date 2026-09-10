from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tft_agent.app.aliases import AliasEntry, AliasRegistry, EntityTextNormalizer
from tft_agent.app.service import AgentService
from tft_agent.core.config import Settings


def test_normalizer_handles_nfkc_case_space_and_punctuation() -> None:
    assert EntityTextNormalizer.normalize(" 蓝 Ｂｕｆｆ！") == "蓝buff"
    assert EntityTextNormalizer.normalize("蓝 Buff") == "蓝buff"
    assert EntityTextNormalizer.normalize("蓝BUFF") == "蓝buff"


def test_registry_materializes_scope_and_rejects_missing_conflicts() -> None:
    entries = [
        AliasEntry(aliases=["奶妈"], entity_type="champion", canonical_name="索拉卡",
                   scope="global"),
        AliasEntry(aliases=["天使"], entity_type="composition", canonical_name="森林天使",
                   canonical_id="comp", scope="set", set_number=18),
        AliasEntry(aliases=["奶妈"], entity_type="composition", canonical_name="奶妈阵容",
                   canonical_id="other", scope="set", set_number=18),
        AliasEntry(aliases=["大头"], entity_type="champion", canonical_name="黑默丁格",
                   scope="global"),
    ]
    registry = AliasRegistry(entries)
    accepted = registry.materialize([
        ("unit", "champion", "索拉卡", "test"),
        ("comp", "composition", "森林天使", "test"),
        ("other", "composition", "奶妈阵容", "test"),
    ], 18, "18.1d")
    assert [(item.alias, item.canonical_id) for item in accepted] == [("天使", "comp")]
    assert {item.reason for item in registry.rejections()} == {
        "alias_conflict", "target_missing",
    }


def test_registry_default_contains_reviewed_alias_sets() -> None:
    registry = AliasRegistry.load_default()
    approved = [entry for entry in registry.entries if entry.review_status == "approved"]
    assert sum(entry.entity_type == "champion" for entry in approved) == 39
    assert sum(entry.entity_type == "item" for entry in approved) == 18
    assert sum(entry.entity_type == "composition" for entry in approved) == 3
    assert not any("generated_prefix" in entry.notes or "generated_suffix" in entry.notes
                   for entry in registry.entries)


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["狮子狗", "螳螂", "奶妈", "石头人"])
async def test_ui_and_chat_share_champion_alias_identity(alias: str, tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "data" / "tft.sqlite3"
    database = tmp_path / "aliases.sqlite3"
    shutil.copy2(source, database)
    service = AgentService(Settings(database_path=database))
    resolved = await service._resolver().resolve(alias)
    chat = next(item for item in resolved.entities if item.entity_type == "champion")
    catalog = await service.entity_catalog()
    ui = next(item for item in catalog.search(alias) if item.entity_type == "champion")
    assert ui.canonical_id == chat.canonical_id


@pytest.mark.asyncio
async def test_existing_item_alias_still_uses_the_same_catalog(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "data" / "tft.sqlite3"
    database = tmp_path / "item-alias.sqlite3"
    shutil.copy2(source, database)
    service = AgentService(Settings(database_path=database))
    resolved = await service._resolver().resolve("青龙刀")
    chat = next(item for item in resolved.entities if item.entity_type == "item")
    catalog = await service.entity_catalog()
    ui = next(item for item in catalog.search("青龙刀") if item.entity_type == "item")
    assert ui.canonical_id == chat.canonical_id


@pytest.mark.asyncio
async def test_single_character_champion_requires_explicit_entity_query(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "data" / "tft.sqlite3"
    database = tmp_path / "single-character.sqlite3"
    shutil.copy2(source, database)
    service = AgentService(Settings(database_path=database))

    for query in ("易怎么玩", "易出什么", "玩易"):
        resolution = await service._resolver().resolve(query)
        champion = next(item for item in resolution.entities
                        if item.entity_type == "champion")
        assert champion.display_name == "易"
        assert champion.canonical_id == "DA_18_MasterYi_AD"

    for query in ("这个交易很容易完成", "局势容易判断", "收益可能变化"):
        resolution = await service._resolver().resolve(query)
        assert not [item for item in resolution.entities
                    if item.canonical_id == "DA_18_MasterYi_AD"]
