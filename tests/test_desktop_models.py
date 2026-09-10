from pathlib import Path

import pytest

from tft_agent.app.pipeline import ProductMode
from tft_agent.app.service import AgentService
from tft_agent.app.session_scope import session_scope
from tft_agent.core.config import Settings
from tft_agent.core.schemas import ChatRequest, GameState
from tft_agent.desktop.catalog import CanonicalEntityCatalog, EntityOption, is_component
from tft_agent.desktop.selectors import EntitySelectorModel


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


def test_general_guide_has_an_isolated_stateless_scope() -> None:
    scope = session_scope("desktop", ProductMode.GENERAL_GUIDE)
    assert scope.session_id == "desktop:general-guide"
    assert scope.uses_game_state is False
    assert scope.accepts_screenshot is False


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
