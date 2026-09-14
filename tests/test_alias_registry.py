from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tft_agent.app.aliases import AliasEntry, AliasRegistry, EntityTextNormalizer
from tft_agent.app.service import AgentService
from tft_agent.core.config import Settings

EMBLEMABLE_TRAITS = [
    ("主宰", "DA_18_EmblemJuggernaut"),
    ("地狱火", "DA_18_EmblemInferno"),
    ("护卫", "DA_18_EmblemDefender"),
    ("斗士", "DA_18_EmblemBrawler"),
    ("月蚀骑士", "DA_18_EmblemLunar"),
    ("永恒之森", "DA_18_EmblemElderwood"),
    ("法师", "DA_18_EmblemSpellweaver"),
    ("灵魂莲华", "DA_18_EmblemBlossom"),
    ("狂战士", "DA_18_EmblemSlayer"),
    ("猎人", "DA_18_EmblemHunter"),
    ("神谕", "DA_18_EmblemInvoker"),
    ("约德尔人和朋友", "DA_18_EmblemSprykin"),
    ("绝命花妖", "DA_18_EmblemFloraFatalis"),
    ("花仙子", "DA_18_EmblemFae"),
    ("裁决使", "DA_18_EmblemExecutioner"),
    ("迅捷射手", "DA_18_EmblemRapidfire"),
    ("重装战士", "DA_18_EmblemVanguard"),
    ("野兽之灵", "DA_18_EmblemPrimal"),
    ("魔女", "DA_18_EmblemCoven"),
    ("黑荆棘", "DA_18_EmblemBlackthorn"),
]

APPROVED_EMBLEM_STEMS = [
    ("花妖", "DA_18_EmblemFloraFatalis"),
    ("森林", "DA_18_EmblemElderwood"),
    ("永森", "DA_18_EmblemElderwood"),
    ("迅射", "DA_18_EmblemRapidfire"),
    ("裁决", "DA_18_EmblemExecutioner"),
    ("猎手", "DA_18_EmblemHunter"),
    ("莲华", "DA_18_EmblemBlossom"),
    ("重装", "DA_18_EmblemVanguard"),
]


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
    assert {
        alias
        for entry in approved
        if entry.entity_type == "trait"
        for alias in entry.aliases
    } == {"重装", "花妖", "森林", "永森", "迅射", "裁决", "猎手", "莲华", "野怪"}
    assert not any("generated_prefix" in entry.notes or "generated_suffix" in entry.notes
                   for entry in registry.entries)


def test_emblem_expressions_require_a_unique_real_trait_relation() -> None:
    registry = AliasRegistry([
        AliasEntry(aliases=["森林"], entity_type="trait", canonical_name="永恒之森",
                   canonical_id="trait-forest", scope="set", set_number=18),
        AliasEntry(aliases=["野怪"], entity_type="trait", canonical_name="峡谷野怪",
                   canonical_id="trait-monster", scope="set", set_number=18),
    ])
    accepted = registry.materialize([
        ("trait-forest", "trait", "永恒之森", "test"),
        ("trait-monster", "trait", "峡谷野怪", "test"),
        ("emblem-forest", "emblem", "永恒之森纹章", "test"),
        ("phantom-one", "emblem", "幻影纹章", "test"),
        ("phantom-two", "emblem", "幻影纹章", "test"),
    ], 18, "18.2")
    aliases = {(item.alias, item.canonical_id, item.entity_type) for item in accepted}

    assert {
        ("森林", "trait-forest", "trait"),
        ("野怪", "trait-monster", "trait"),
        ("永恒之森转", "emblem-forest", "emblem"),
        ("永恒之森转职", "emblem-forest", "emblem"),
        ("森林转", "emblem-forest", "emblem"),
        ("森林纹章", "emblem-forest", "emblem"),
        ("森林转职", "emblem-forest", "emblem"),
    } <= aliases
    assert not any(item.alias.startswith("野怪") and item.entity_type == "emblem"
                   for item in accepted)
    assert not any(item.alias == "幻影转" for item in accepted)


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
@pytest.mark.parametrize(("trait_name", "emblem_id"), EMBLEMABLE_TRAITS)
async def test_all_emblemable_traits_support_canonical_product_expressions(
    trait_name: str, emblem_id: str, tmp_path: Path
) -> None:
    source = Path(__file__).parents[1] / "data" / "tft.sqlite3"
    database = tmp_path / "canonical-emblems.sqlite3"
    shutil.copy2(source, database)
    service = AgentService(Settings(database_path=database, current_game_patch="18.2"))

    trait = await service._resolver().resolve(trait_name)
    assert [(item.entity_type, item.display_name) for item in trait.entities] == [
        ("trait", trait_name)
    ]
    for phrase in (trait_name + "转", trait_name + "纹章", trait_name + "转职"):
        resolution = await service._resolver().resolve(phrase)
        assert [(item.entity_type, item.canonical_id) for item in resolution.entities] == [
            ("emblem", emblem_id)
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("stem", "emblem_id"), APPROVED_EMBLEM_STEMS)
async def test_approved_trait_stems_support_product_expressions(
    stem: str, emblem_id: str, tmp_path: Path
) -> None:
    source = Path(__file__).parents[1] / "data" / "tft.sqlite3"
    database = tmp_path / "stem-emblems.sqlite3"
    shutil.copy2(source, database)
    service = AgentService(Settings(database_path=database, current_game_patch="18.2"))

    trait = await service._resolver().resolve(stem)
    assert [(item.entity_type, item.match_kind) for item in trait.entities] == [
        ("trait", "exact_alias")
    ]
    for phrase in (stem + "转", stem + "纹章", stem + "转职"):
        resolution = await service._resolver().resolve(phrase)
        assert [(item.entity_type, item.canonical_id) for item in resolution.entities] == [
            ("emblem", emblem_id)
        ]


@pytest.mark.asyncio
async def test_non_emblemable_stem_does_not_create_fake_emblem_alias(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "data" / "tft.sqlite3"
    database = tmp_path / "non-emblemable.sqlite3"
    shutil.copy2(source, database)
    service = AgentService(Settings(database_path=database, current_game_patch="18.2"))

    assert [(item.entity_type, item.display_name)
            for item in (await service._resolver().resolve("野怪")).entities] == [
        ("trait", "峡谷野怪")
    ]
    unsupported = await service._resolver().resolve("野怪转")
    assert not any(item.entity_type == "emblem" for item in unsupported.entities)


@pytest.mark.asyncio
@pytest.mark.parametrize("phrase", ["幻影纹章", "幻影转"])
async def test_ambiguous_phantom_emblem_stays_unresolved(
    phrase: str, tmp_path: Path
) -> None:
    source = Path(__file__).parents[1] / "data" / "tft.sqlite3"
    database = tmp_path / "phantom.sqlite3"
    shutil.copy2(source, database)
    service = AgentService(Settings(database_path=database, current_game_patch="18.2"))

    assert (await service._resolver().resolve(phrase)).entities == []


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
