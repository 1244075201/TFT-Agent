"""Typed, scrollable editor for user-confirmed match facts."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tft_agent.core.schemas import GameState
from tft_agent.desktop.catalog import CanonicalEntityCatalog, is_completed_item, is_component
from tft_agent.desktop.selectors import EntitySelectorModel, searchable_selector_class


class NumericStateEditor(QSpinBox):
    def __init__(self, maximum: int) -> None:
        super().__init__()
        self.setRange(-1, maximum)
        self.setSpecialValueText("未填写")

    def optional_value(self) -> int | None:
        return None if self.value() < 0 else self.value()


class StageSelector(QComboBox):
    def __init__(self) -> None:
        super().__init__()
        self.addItem("未填写", None)
        for stage in range(2, 8):
            for round_number in range(1, 8):
                value = f"{stage}-{round_number}"
                self.addItem(value, value)

    def optional_value(self) -> str | None:
        value = self.currentData()
        return str(value) if value else None


class AugmentChoiceSelector(QWidget):
    changed = Signal()

    def __init__(self, catalog: CanonicalEntityCatalog) -> None:
        super().__init__()
        Selector = searchable_selector_class()
        self.selectors: list[Any] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for index in range(3):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Choice {index + 1}"))
            selector = Selector(EntitySelectorModel(catalog, ["augment"]))
            selector.values_changed.connect(lambda i=index: self._deduplicate(i))
            self.selectors.append(selector)
            row.addWidget(selector, 1)
            layout.addLayout(row)

    def _deduplicate(self, changed_index: int) -> None:
        changed = self.selectors[changed_index].model.selected
        if not changed:
            self.changed.emit()
            return
        canonical_id = changed[0].canonical_id
        for index, selector in enumerate(self.selectors):
            if index != changed_index and any(
                item.canonical_id == canonical_id for item in selector.model.selected
            ):
                selector.clear()
        self.changed.emit()

    def values(self) -> list[str]:
        return [selector.text() for selector in self.selectors if selector.text().strip()]

    def set_values(self, values: list[str]) -> None:
        for index, selector in enumerate(self.selectors):
            selector.clear()
            if index < len(values):
                selector.setText(values[index])

    def clear(self) -> None:
        for selector in self.selectors:
            selector.clear()


class GameStateEditor(QScrollArea):
    dirty_changed = Signal(bool)
    save_requested = Signal()
    clear_requested = Signal()
    screenshot_requested = Signal()

    def __init__(self, catalog: CanonicalEntityCatalog) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self._dirty = False
        body = QWidget()
        self.setWidget(body)
        layout = QVBoxLayout(body)
        form = QFormLayout()
        layout.addLayout(form)
        Selector = searchable_selector_class()
        self.comp = Selector(EntitySelectorModel(catalog, ["composition"]))
        self.stage = StageSelector()
        self.level = NumericStateEditor(10)
        self.gold = NumericStateEditor(200)
        self.hp = NumericStateEditor(100)
        self.augments = Selector(EntitySelectorModel(catalog, ["augment"], multiple=True))
        self.augment_choices = AugmentChoiceSelector(catalog)
        self.components = Selector(EntitySelectorModel(
            catalog, ["item"], multiple=True, count_duplicates=True, predicate=is_component
        ))
        self.completed = Selector(EntitySelectorModel(
            catalog, ["item"], multiple=True, count_duplicates=True, predicate=is_completed_item
        ))
        self.emblems = Selector(EntitySelectorModel(
            catalog, ["emblem"], multiple=True, count_duplicates=True
        ))
        self.units = Selector(EntitySelectorModel(catalog, ["champion"], multiple=True))
        self.notes = QLineEdit()
        for label, widget in (
            ("目标阵容", self.comp), ("阶段", self.stage), ("等级", self.level),
            ("金币", self.gold), ("生命", self.hp), ("已有强化", self.augments),
            ("强化三选一", self.augment_choices), ("散件", self.components),
            ("成装", self.completed), ("转职", self.emblems),
            ("关键棋子", self.units), ("备注", self.notes),
        ):
            form.addRow(label, widget)
        self.conflict = QLabel()
        self.conflict.setStyleSheet("color:#b7791f")
        layout.addWidget(self.conflict)
        actions = QHBoxLayout()
        save = QPushButton("保存局面")
        clear = QPushButton("清空本局")
        screenshot = QPushButton("截图识别")
        actions.addWidget(save)
        actions.addWidget(clear)
        actions.addWidget(screenshot)
        layout.addLayout(actions)
        layout.addStretch()
        save.clicked.connect(self.save_requested.emit)
        clear.clicked.connect(self.clear_requested.emit)
        screenshot.clicked.connect(self.screenshot_requested.emit)
        self._connect_dirty_signals()

    def _connect_dirty_signals(self) -> None:
        for selector in (
            self.comp, self.augments, self.components, self.completed, self.emblems, self.units
        ):
            selector.values_changed.connect(self.mark_dirty)
        self.augment_choices.changed.connect(self.mark_dirty)
        self.stage.currentIndexChanged.connect(self.mark_dirty)
        for spinbox in (self.level, self.gold, self.hp):
            spinbox.valueChanged.connect(self.mark_dirty)
        self.notes.textChanged.connect(self.mark_dirty)

    def mark_dirty(self) -> None:
        if not self._dirty:
            self._dirty = True
            self.dirty_changed.emit(True)

    def game_state(self, question: str = "") -> GameState:
        return GameState(
            comp_name=self.comp.text().strip() or None,
            stage=self.stage.optional_value(),
            level=self.level.optional_value(),
            gold=self.gold.optional_value(),
            hp=self.hp.optional_value(),
            augments=_values(self.augments.text()),
            augment_choices=self.augment_choices.values(),
            item_components=_values(self.components.text()),
            completed_items=_values(self.completed.text()),
            emblems=_values(self.emblems.text()),
            core_units=_values(self.units.text()),
            notes=self.notes.text().strip() or None,
            user_question=question or None,
        )

    def set_game_state(self, state: GameState) -> None:
        self.comp.setText(state.comp_name or "")
        self.stage.setCurrentText(state.stage or "未填写")
        self.level.setValue(state.level if state.level is not None else -1)
        self.gold.setValue(state.gold if state.gold is not None else -1)
        self.hp.setValue(state.hp if state.hp is not None else -1)
        self.augments.setText("，".join(state.augments))
        self.augment_choices.set_values(state.augment_choices)
        self.components.setText("，".join(state.item_components))
        self.completed.setText("，".join(state.completed_items))
        self.emblems.setText("，".join(state.emblems))
        self.units.setText("，".join(state.core_units))
        self.notes.setText(state.notes or "")
        self._dirty = False
        self.dirty_changed.emit(False)

    def clear_state(self) -> None:
        self.set_game_state(GameState())


def _values(text: str) -> list[str]:
    return [value.strip() for value in text.replace("，", ",").split(",") if value.strip()]
