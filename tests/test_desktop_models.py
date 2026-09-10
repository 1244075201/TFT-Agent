from pathlib import Path

import pytest

from tft_agent.app.pipeline import ProductMode
from tft_agent.app.service import AgentService
from tft_agent.app.session_scope import session_scope
from tft_agent.core.config import Settings
from tft_agent.core.schemas import ChatRequest, GameState
from tft_agent.desktop.catalog import (
    CanonicalEntityCatalog,
    EntityOption,
    is_component,
    search_match_reason,
)
from tft_agent.desktop.requests import game_state_summary, guide_request, match_request
from tft_agent.desktop.selectors import AugmentChoiceModel, EntitySelectorModel


def _catalog() -> CanonicalEntityCatalog:
    return CanonicalEntityCatalog([
        EntityOption(
            canonical_id="comp-1",
            display_name="森林天使",
            entity_type="composition",
            aliases=["天使"],
        ),
        EntityOption(
            canonical_id="item-1",
            display_name="暴风大剑",
            entity_type="item",
            aliases=["大剑"],
            tags=["component"],
        ),
    ])


def test_catalog_maps_only_presentation_fields() -> None:
    catalog = CanonicalEntityCatalog.from_repository([{
        "canonical_id": "comp-1",
        "display_name": "森林天使",
        "entity_type": "composition",
        "aliases": ["天使"],
        "source": "internal-source",
        "game_patch": "18.1d",
    }])
    assert catalog.options == [EntityOption(
        canonical_id="comp-1",
        display_name="森林天使",
        entity_type="composition",
        aliases=["天使"],
    )]
    assert "source" not in catalog.options[0].model_dump()


def test_selector_searches_aliases_and_enforces_type() -> None:
    model = EntitySelectorModel(_catalog(), ["composition"])
    assert [item.display_name for item in model.suggestions("天使")] == ["森林天使"]
    model.select(model.suggestions("天使")[0])
    assert model.display_values() == ["森林天使"]
    assert search_match_reason(model.selected[0], "天使") == "匹配别名：天使"
    with pytest.raises(ValueError):
        model.select(_catalog().options[1])


def test_multi_selector_deduplicates_and_removes_values() -> None:
    model = EntitySelectorModel(_catalog(), ["item"], multiple=True, predicate=is_component)
    item = model.suggestions("大剑")[0]
    model.select(item)
    model.select(item)
    assert model.display_values() == ["暴风大剑"]
    model.remove(item.canonical_id)
    assert model.display_values() == []


def test_quantity_selector_increments_and_decrements_one_at_a_time() -> None:
    model = EntitySelectorModel(_catalog(), ["item"], multiple=True, count_duplicates=True)
    option = model.suggestions("大剑")[0]
    model.select(option)
    model.select(option)
    assert model.selected[0].quantity == 2
    assert model.display_values() == ["暴风大剑", "暴风大剑"]
    model.remove(option.canonical_id)
    assert model.selected[0].quantity == 1


def test_general_guide_has_an_isolated_stateless_scope() -> None:
    scope = session_scope("desktop", ProductMode.GENERAL_GUIDE)
    assert scope.session_id == "desktop:general-guide"
    assert scope.uses_game_state is False
    assert scope.accepts_screenshot is False


def test_workspace_requests_use_independent_sessions_and_modes() -> None:
    guide = guide_request("神谕转怎么玩")
    state = GameState(stage="4-1", level=8)
    match = match_request("神谕转怎么玩", state)
    assert (guide.session_id, guide.product_mode, guide.confirmed_game_state) == (
        "desktop-guide", "general_guide", None
    )
    assert (match.session_id, match.product_mode, match.confirmed_game_state) == (
        "desktop-match", "match_guidance", state
    )


def test_augment_choices_keep_slots_stable_and_reject_duplicates() -> None:
    first = EntityOption(canonical_id="a", display_name="宝宝学院", entity_type="augment")
    second = EntityOption(canonical_id="b", display_name="三幻神", entity_type="augment")
    choices = AugmentChoiceModel()
    choices.choose(0, first)
    choices.choose(2, second)
    choices.clear_slot(0)
    assert choices.slots == [None, None, second]
    with pytest.raises(ValueError):
        choices.choose(1, second)


def test_game_state_summary_is_compact() -> None:
    assert game_state_summary(GameState(
        stage="4-1", level=8, gold=35, hp=50, comp_name="艾希95"
    )) == "4-1 · 8级 · 35金 · 50血 · 艾希95"


@pytest.mark.asyncio
async def test_general_guide_does_not_read_or_overwrite_match_state(tmp_path: Path) -> None:
    service = AgentService(Settings(database_path=tmp_path / "scope.sqlite3"))
    match_state = GameState(comp_name="艾希95", level=8, gold=35)
    await service.save_game_state("desktop", match_state)

    events = [event async for event in service.ask(ChatRequest(
        text="核心棋子有哪些棋子",
        session_id="desktop",
        product_mode="general_guide",
        confirmed_game_state=GameState(comp_name="不应保存"),
    ))]

    assert events[-1].type == "answer"
    assert await service.load_game_state("desktop") == match_state
    assert await service.load_game_state("desktop:general-guide") == GameState()
    assert await service._repo().load_session_messages("desktop") == []
    assert await service._repo().load_session_messages("desktop:general-guide")


@pytest.mark.asyncio
async def test_match_inventory_categories_persist_without_schema_migration(tmp_path: Path) -> None:
    service = AgentService(Settings(database_path=tmp_path / "inventory.sqlite3"))
    state = GameState(
        item_components=["反曲之弓"], completed_items=["朔极之矛"],
        artifacts=["死亡之蔑"], radiant_items=["光明版蓝霸符"],
        emblems=["神谕者纹章"],
    )
    await service.save_game_state("desktop-match", state)
    assert await service.load_game_state("desktop-match") == state
