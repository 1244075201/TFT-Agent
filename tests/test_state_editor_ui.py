from __future__ import annotations

import os
from typing import cast

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel

from tft_agent.core.schemas import GameState
from tft_agent.desktop.catalog import CanonicalEntityCatalog, EntityOption
from tft_agent.desktop.selectors import EntitySelectorModel, searchable_selector_class
from tft_agent.desktop.state_editor import GameStateEditor


def _app() -> QApplication:
    instance = QApplication.instance()
    return cast(QApplication, instance) if instance else QApplication([])


def _catalog() -> CanonicalEntityCatalog:
    return CanonicalEntityCatalog([
        EntityOption(canonical_id="comp", display_name="森林天使", entity_type="composition",
                     aliases=["天使"]),
        EntityOption(canonical_id="bow", display_name="反曲之弓", entity_type="item",
                     tags=["component"], item_category="component"),
        EntityOption(canonical_id="glove", display_name="拳套", entity_type="item",
                     tags=["component"], item_category="component"),
        EntityOption(canonical_id="sword", display_name="暴风大剑", entity_type="item",
                     tags=["component"], item_category="component"),
        EntityOption(canonical_id="shojin", display_name="朔极之矛", entity_type="item",
                     aliases=["青龙刀"], components=["暴风大剑", "女神之泪"],
                     item_category="completed"),
        EntityOption(canonical_id="artifact", display_name="死亡之蔑", entity_type="item",
                     item_category="artifact"),
        EntityOption(canonical_id="radiant", display_name="光明版蓝霸符", entity_type="item",
                     item_category="radiant"),
        EntityOption(canonical_id="emblem", display_name="神谕者纹章", entity_type="emblem"),
        EntityOption(canonical_id="malphite", display_name="墨菲特", entity_type="champion",
                     aliases=["石头人"]),
        EntityOption(canonical_id="augment", display_name="宝宝学院", entity_type="augment"),
    ])


def _choose(selector: object, query: str) -> None:
    selector.search.setText(query)  # type: ignore[attr-defined]
    selector._refresh(query)  # type: ignore[attr-defined]
    assert selector._candidate_options  # type: ignore[attr-defined]
    selector._activate(selector.candidates.model().index(0, 0))  # type: ignore[attr-defined]


def test_search_and_selected_state_are_separate() -> None:
    _app()
    Selector = searchable_selector_class()
    selector = Selector(EntitySelectorModel(
        _catalog(), ["item"], multiple=True, count_duplicates=True,
        predicate=lambda option: option.item_category == "component",
    ))
    _choose(selector, "弓")
    assert selector.search.text() == ""
    assert selector.model.display_values() == ["反曲之弓"]
    _choose(selector, "拳套")
    assert selector.model.display_values() == ["反曲之弓", "拳套"]
    _choose(selector, "暴风大剑")
    _choose(selector, "暴风大剑")
    assert selector.model.selected[-1].quantity == 2
    assert "暴风大剑 ×2" in selector.chips_layout.itemAt(2).widget().text()


def test_editor_fields_and_state_round_trip_without_choice_ui() -> None:
    _app()
    editor = GameStateEditor(_catalog())
    labels = {label.text() for label in editor.findChildren(QLabel)}
    assert {"奥恩神器", "光明装备"} <= labels
    assert "强化三选一" not in labels
    assert not any(value.startswith("Choice ") for value in labels)

    state = GameState(
        comp_name="森林天使", active_comp_id="comp", augments=["宝宝学院"],
        augment_choices=["旧三选一"],
        item_components=["反曲之弓", "暴风大剑", "暴风大剑"],
        completed_items=["朔极之矛"], artifacts=["死亡之蔑"],
        radiant_items=["光明版蓝霸符"], emblems=["神谕者纹章"], core_units=["墨菲特"],
    )
    editor.set_game_state(state)
    restored = editor.game_state()
    assert restored.model_dump(exclude={"user_question"}) == state.model_dump(
        exclude={"user_question"}
    )
    assert editor.comp.search.text() == ""
    assert editor.components.model.selected[1].quantity == 2


def test_aliases_add_to_the_correct_selected_areas() -> None:
    _app()
    editor = GameStateEditor(_catalog())
    _choose(editor.completed, "青龙刀")
    _choose(editor.units, "石头人")
    _choose(editor.comp, "天使")
    assert editor.completed.text() == "朔极之矛"
    assert editor.units.text() == "墨菲特"
    assert editor.comp.text() == "森林天使"
    assert {
        editor.completed.search.text(), editor.units.search.text(), editor.comp.search.text()
    } == {""}
